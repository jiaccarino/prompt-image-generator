import time
import torch
import psutil
import threading
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel, BigGANModel, BigGANConfig
from torchvision.transforms import ToPILImage
import numpy as np

class BigGANBenchmark:
    def __init__(self, config_file):
        self.load_config(config_file)
        self.setup_logging()
        self.setup_monitoring()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            self.prompts = config['prompts']
            self.target_size = tuple(config.get('target_size', (512, 512)))
            self.truncation = config.get('truncation', 0.4)
            self.batch_size = config.get('batch_size', 1)
            self.model_size = config.get('model_size', '512')  # Default to 512
            
    def setup_logging(self):
        Path('logs').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/biggan_benchmark_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging
        
    def setup_monitoring(self):
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.vram_usage = []
        self.timestamps = []
        self.prompt_end_times = []
        self.monitoring = True
        
    def monitor_resources(self):
        while self.monitoring:
            current_time = time.time() - self.start_time
            self.cpu_usage.append(psutil.cpu_percent())
            self.ram_usage.append(psutil.virtual_memory().used / (1024 ** 3))
            
            if torch.cuda.is_available():
                try:
                    self.gpu_usage.append(torch.cuda.utilization())
                    self.vram_usage.append(torch.cuda.memory_allocated() / (1024 ** 3))
                except Exception:
                    self.gpu_usage.append(0)
                    self.vram_usage.append(0)
            else:
                self.gpu_usage.append(0)
                self.vram_usage.append(0)
                
            self.timestamps.append(current_time)
            time.sleep(0.1)
            
    def get_baseline_usage(self):
        baseline = {
            'cpu': psutil.cpu_percent(),
            'ram': psutil.virtual_memory().used / (1024 ** 3),
            'gpu': 0,
            'vram': 0
        }
        
        if torch.cuda.is_available():
            try:
                baseline['gpu'] = torch.cuda.utilization()
                baseline['vram'] = torch.cuda.memory_allocated() / (1024 ** 3)
            except Exception:
                pass
                
        return baseline
        
    def load_models(self):
        # Create cache directory
        Path('./model_cache').mkdir(exist_ok=True)
        
        # Load BigGAN
        self.logger.info("Loading BigGAN model...")
        try:
            config = BigGANConfig.from_pretrained(
                f'biggan-deep-{self.model_size}',
                cache_dir='./model_cache'
            )
            self.biggan = BigGANModel.from_pretrained(
                f'biggan-deep-{self.model_size}',
                config=config,
                cache_dir='./model_cache'
            )
            self.biggan.to(self.device)
            self.biggan.eval()
        except Exception as e:
            self.logger.error(f"Error loading BigGAN: {str(e)}")
            raise RuntimeError("Failed to load BigGAN model")
        
        # Load CLIP
        self.logger.info("Loading CLIP model...")
        try:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir='./model_cache'
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir='./model_cache'
            )
            self.clip_model.to(self.device)
            self.clip_model.eval()
        except Exception as e:
            self.logger.error(f"Error loading CLIP: {str(e)}")
            raise RuntimeError("Failed to load CLIP model")
        
        # Load ImageNet classes
        self.load_imagenet_classes()
        
    def load_imagenet_classes(self):
        self.logger.info("Loading ImageNet classes...")
        try:
            import requests
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            response = requests.get(url)
            self.class_names = response.text.splitlines()
        except Exception:
            self.logger.warning("Could not load ImageNet classes from URL. Using CLIP only.")
            self.class_names = None
        
    def get_class_vector_from_text(self, text):
        # Process text through CLIP
        inputs = self.clip_processor(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            
            if self.class_names:
                # Calculate similarity with ImageNet classes
                class_inputs = self.clip_processor(
                    self.class_names, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                class_inputs = {k: v.to(self.device) for k, v in class_inputs.items()}
                class_features = self.clip_model.get_text_features(**class_inputs)
                
                # Calculate similarities
                similarities = torch.nn.functional.cosine_similarity(
                    text_features.unsqueeze(1),
                    class_features.unsqueeze(0),
                    dim=2
                )
                
                # Create class vector
                class_vector = torch.zeros(1000, device=self.device)
                top_k = 5
                values, indices = similarities[0].topk(top_k)
                values = torch.nn.functional.softmax(values, dim=0)
                
                for val, idx in zip(values, indices):
                    class_vector[idx] = val
            else:
                # Fallback: project CLIP features to 1000-dim space
                class_vector = torch.nn.functional.linear(
                    text_features, 
                    self.clip_model.text_projection
                )
                class_vector = torch.nn.functional.normalize(class_vector, dim=-1)
                class_vector = torch.nn.functional.pad(
                    class_vector, 
                    (0, 1000 - class_vector.size(1))
                )
            
            return class_vector
        
    def generate_image(self, prompt):
        try:
            # Get class vector from text
            class_vector = self.get_class_vector_from_text(prompt)
            
            # Generate random noise vector
            noise_vector = torch.randn(
                self.batch_size, 
                128,
                device=self.device
            )
            
            # Generate image
            with torch.no_grad():
                output = self.biggan(
                    noise_vector,
                    class_vector.unsqueeze(0),
                    self.truncation
                ).images
            
            # Convert to PIL Image
            output = output.cpu()
            output = (output + 1) / 2
            output = output.clamp(0, 1)
            image = ToPILImage()(output[0])
            
            return image
        
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise RuntimeError(f"Failed to generate image for prompt: {prompt}")
        
    def save_metrics(self):
        metrics = {
            'cpu': (min(self.cpu_usage), max(self.cpu_usage)),
            'ram': (min(self.ram_usage), max(self.ram_usage)),
            'gpu': (min(self.gpu_usage), max(self.gpu_usage)),
            'vram': (min(self.vram_usage), max(self.vram_usage))
        }
        
        self.logger.info("\nMetrics for BigGAN:")
        self.logger.info("--------------------")
        for resource, (min_val, max_val) in metrics.items():
            self.logger.info(
                f"{resource.upper()} - Min: {min_val:.2f}{'%' if resource in ['cpu', 'gpu'] else 'GB'}, "
                f"Max: {max_val:.2f}{'%' if resource in ['cpu', 'gpu'] else 'GB'}"
            )
        
    def plot_utilization(self):
        plt.figure(figsize=(12, 8))
        
        plots = {
            1: ('CPU (%)', self.cpu_usage, '%'),
            2: ('RAM (GB)', self.ram_usage, 'GB'),
            3: ('GPU (%)', self.gpu_usage, '%'),
            4: ('VRAM (GB)', self.vram_usage, 'GB')
        }
        
        for i, (title, data, unit) in plots.items():
            plt.subplot(2, 2, i)
            plt.plot(self.timestamps, data, label=f'{title} Usage')
            plt.axhline(y=self.baseline_usage[title.split()[0].lower()], 
                       color='r', linestyle='--', label='Baseline')
            plt.axvline(x=self.models_loading_time - self.start_time, 
                       color='b', linestyle='--', label='Models Loaded')
            
            # Add prompt completion lines
            for end_time in self.prompt_end_times:
                plt.axvline(x=end_time, color='g', linestyle='--')
                
            plt.xlabel('Time (s)')
            plt.ylabel(unit)
            plt.title(title)
            plt.legend()
            
        plt.tight_layout()
        
        # Save plot
        Path('plots').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'plots/biggan_utilization_{timestamp}.png')
        plt.close()
        
    def run_benchmark(self):
        self.logger.info("Starting BigGAN benchmark")
        
        # Start resource monitoring
        self.start_time = time.time()
        self.baseline_usage = self.get_baseline_usage()
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.start()
        
        try:
            # Load models
            self.load_models()
            self.models_loading_time = time.time()
            
            # Generate images for each prompt
            times = []
            for i, prompt in enumerate(self.prompts, 1):
                start_prompt_time = time.time()
                
                self.logger.info(f"Generating image {i}/{len(self.prompts)} for prompt: '{prompt}'")
                image = self.generate_image(prompt)
                
                end_prompt_time = time.time()
                elapsed_time = end_prompt_time - start_prompt_time
                times.append(elapsed_time)
                
                # Save image
                Path('images').mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'images/biggan_image_{i}_{timestamp}.png'
                image = image.resize(self.target_size)
                image.save(filename)
                
                self.logger.info(f"Generated image {i}/{len(self.prompts)} in {elapsed_time:.2f} seconds")
                self.prompt_end_times.append(end_prompt_time - self.start_time)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log average generation time
            avg_time = sum(times) / len(times)
            self.logger.info(f"Average generation time: {avg_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error during benchmark: {str(e)}")
            raise
            
        finally:
            # Stop monitoring and save metrics
            self.monitoring = False
            monitor_thread.join()
            self.save_metrics()
            self.plot_utilization()
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(self, 'biggan'):
                del self.biggan
            if hasattr(self, 'clip_model'):
                del self.clip_model

def main():
    config_file = 'biggan_config.json'
    
    # Create config file
    config = {
        "prompts": [
            "a red rose in full bloom",
            "a snowy mountain peak at sunset",
            "a golden retriever puppy playing"
        ],
        "target_size": [512, 512],
        "truncation": 0.4,
        "batch_size": 1
    }
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run benchmark
    benchmark = BigGANBenchmark(config_file)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
