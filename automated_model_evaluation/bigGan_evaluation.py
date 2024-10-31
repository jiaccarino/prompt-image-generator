import time
import torch
import psutil
import threading
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
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
            self.model_size = config.get('model_size', '512')  # Can be '128', '256', or '512'
            
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
                self.gpu_usage.append(torch.cuda.utilization())
                self.vram_usage.append(torch.cuda.memory_allocated() / (1024 ** 3))
            else:
                self.gpu_usage.append(0)
                self.vram_usage.append(0)
            self.timestamps.append(current_time)
            time.sleep(0.1)
            
    def get_baseline_usage(self):
        return {
            'cpu': psutil.cpu_percent(),
            'ram': psutil.virtual_memory().used / (1024 ** 3),
            'gpu': torch.cuda.utilization() if torch.cuda.is_available() else 0,
            'vram': torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        }
        
    def load_models(self):
        # Load BigGAN using torch.hub
        self.logger.info("Loading BigGAN model...")
        model_name = f'biggan-deep-{self.model_size}'
        self.biggan = torch.hub.load('huggingface/pytorch-pretrained-biggan', 
                                   model_name, 
                                   pretrained=True)
        self.biggan.to(self.device)
        
        # Load CLIP for text encoding
        self.logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # Load ImageNet class mappings
        self.load_imagenet_classes()
        
    def load_imagenet_classes(self):
        # Load ImageNet class names
        self.logger.info("Loading ImageNet classes...")
        classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            import urllib.request
            with urllib.request.urlopen(classes_url) as f:
                self.class_names = [line.decode('utf-8').strip() for line in f.readlines()]
        except:
            self.logger.warning("Could not load ImageNet classes from URL. Using CLIP only.")
            self.class_names = None
        
    def get_class_vector_from_text(self, text):
        # Process text through CLIP
        inputs = self.clip_processor(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text features from CLIP
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        
        if self.class_names:
            # Calculate similarity with ImageNet classes
            class_inputs = self.clip_processor(self.class_names, return_tensors="pt", padding=True)
            class_inputs = {k: v.to(self.device) for k, v in class_inputs.items()}
            
            with torch.no_grad():
                class_features = self.clip_model.get_text_features(**class_inputs)
                
            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                text_features.unsqueeze(1),
                class_features.unsqueeze(0),
                dim=2
            )
            
            # Create class vector
            class_vector = torch.zeros(1000, device=self.device)
            top_k = 5  # Use top 5 most similar classes
            values, indices = similarities[0].topk(top_k)
            values = torch.nn.functional.softmax(values, dim=0)
            
            for val, idx in zip(values, indices):
                class_vector[idx] = val
        else:
            # Fallback: project CLIP features directly
            class_vector = torch.nn.functional.linear(
                text_features, 
                self.clip_model.text_projection
            )
            class_vector = torch.nn.functional.normalize(class_vector, dim=-1)
            # Expand to 1000 classes
            class_vector = torch.nn.functional.pad(class_vector, (0, 1000 - class_vector.size(1)))
        
        return class_vector
        
    def generate_image(self, prompt):
        # Get class vector from text
        class_vector = self.get_class_vector_from_text(prompt)
        
        # Generate random noise vector
        noise_vector = torch.randn(
            self.batch_size, 
            128,  # BigGAN's noise dimension
            device=self.device
        )
        
        # Generate image
        with torch.no_grad():
            output = self.biggan(
                noise_vector,
                class_vector,
                self.truncation
            )
        
        # Convert to PIL Image
        output = output.cpu()
        output = (output + 1) / 2  # Convert from [-1, 1] to [0, 1]
        output = output.clamp(0, 1)
        image = ToPILImage()(output[0])
        
        return image
        
    def save_metrics(self):
        min_cpu, max_cpu = min(self.cpu_usage), max(self.cpu_usage)
        min_ram, max_ram = min(self.ram_usage), max(self.ram_usage)
        min_gpu, max_gpu = min(self.gpu_usage), max(self.gpu_usage)
        min_vram, max_vram = min(self.vram_usage), max(self.vram_usage)
        
        self.logger.info("\nMetrics for BigGAN:")
        self.logger.info("--------------------")
        self.logger.info(f"CPU - Min: {min_cpu:.2f}%, Max: {max_cpu:.2f}%")
        self.logger.info(f"GPU - Min: {min_gpu:.2f}%, Max: {max_gpu:.2f}%")
        self.logger.info(f"RAM - Min: {min_ram:.2f}GB, Max: {max_ram:.2f}GB")
        self.logger.info(f"VRAM - Min: {min_vram:.2f}GB, Max: {max_vram:.2f}GB")
        
    def plot_utilization(self):
        plt.figure(figsize=(12, 8))
        
        # CPU
        plt.subplot(2, 2, 1)
        plt.plot(self.timestamps, self.cpu_usage, label='CPU Usage')
        plt.axhline(y=self.baseline_usage['cpu'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.models_loading_time - self.start_time, color='b', linestyle='--', label='Models Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('% Utilization')
        plt.title('CPU (%)')
        
        # RAM
        plt.subplot(2, 2, 2)
        plt.plot(self.timestamps, self.ram_usage, label='RAM Usage')
        plt.axhline(y=self.baseline_usage['ram'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.models_loading_time - self.start_time, color='b', linestyle='--', label='Models Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('GB')
        plt.title('RAM (GB)')
        
        # GPU
        plt.subplot(2, 2, 3)
        plt.plot(self.timestamps, self.gpu_usage, label='GPU Usage')
        plt.axhline(y=self.baseline_usage['gpu'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.models_loading_time - self.start_time, color='b', linestyle='--', label='Models Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('% Utilization')
        plt.title('GPU (%)')
        
        # VRAM
        plt.subplot(2, 2, 4)
        plt.plot(self.timestamps, self.vram_usage, label='VRAM Usage')
        plt.axhline(y=self.baseline_usage['vram'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.models_loading_time - self.start_time, color='b', linestyle='--', label='Models Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('GB')
        plt.title('VRAM (GB)')
        
        # Add prompt completion lines
        for end_time in self.prompt_end_times:
            for i in range(1, 5):
                plt.subplot(2, 2, i).axvline(x=end_time, color='g', linestyle='--')
                
        # Add legends
        for i in range(1, 5):
            plt.subplot(2, 2, i).legend()
            
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
            for i, prompt in enumerate(self.prompts):
                start_prompt_time = time.time()
                
                image = self.generate_image(prompt)
                
                end_prompt_time = time.time()
                elapsed_time = end_prompt_time - start_prompt_time
                times.append(elapsed_time)
                
                # Save image
                Path('images').mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'images/biggan_image_{i+1}_{timestamp}.png'
                image = image.resize(self.target_size)
                image.save(filename)
                
                self.logger.info(f"Generated image {i+1}/{len(self.prompts)} in {elapsed_time:.2f} seconds")
                self.prompt_end_times.append(end_prompt_time - self.start_time)
                torch.cuda.empty_cache()
            
            # Log average generation time
            avg_time = sum(times) / len(times)
            self.logger.info(f"Average generation time: {avg_time:.2f} seconds")
            
            # Stop monitoring and save metrics
            self.monitoring = False
            monitor_thread.join()
            self.save_metrics()
            self.plot_utilization()
            
        except Exception as e:
            self.logger.error(f"Error during benchmark: {str(e)}")
            self.monitoring = False
            monitor_thread.join()
            
        finally:
            # Clean up
            torch.cuda.empty_cache()
            if hasattr(self, 'biggan'):
                del self.biggan
            if hasattr(self, 'clip_model'):
                del self.clip_model

def main():
    benchmark = BigGANBenchmark('biggan_config.json')
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()