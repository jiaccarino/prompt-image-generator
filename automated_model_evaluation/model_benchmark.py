import time
import torch
import psutil
import threading
import matplotlib.pyplot as plt
import yaml
import logging
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, AmusedPipeline
from PIL import Image

class ModelBenchmark:
    def __init__(self, model_config_file, prompts_file):
        self.load_config(model_config_file)
        self.load_prompts(prompts_file)
        self.setup_logging()
        self.setup_monitoring()
        
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.models = config['models']
            self.target_size = tuple(config.get('target_size', (768, 768)))
            
    def load_prompts(self, prompts_file):
        with open(prompts_file, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
            
    def setup_logging(self):
        Path('logs').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/benchmark_{timestamp}.log'
        
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
            self.gpu_usage.append(torch.cuda.utilization())
            self.vram_usage.append(torch.cuda.memory_allocated() / (1024 ** 3))
            self.timestamps.append(current_time)
            time.sleep(0.1)
            
    def get_baseline_usage(self):
        return {
            'cpu': psutil.cpu_percent(),
            'ram': psutil.virtual_memory().used / (1024 ** 3),
            'gpu': torch.cuda.utilization(),
            'vram': torch.cuda.memory_allocated() / (1024 ** 3)
        }
        
    def get_pipeline(self, model_config):
        model_name = model_config['name']
        pipeline_type = model_config['type']
        
        if pipeline_type == 'stable-diffusion':
            return StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=model_config.get('torch_dtype', torch.float32),
                variant=model_config.get('variant', None)
            )
        elif pipeline_type == 'sdxl':
            return AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=model_config.get('torch_dtype', torch.float16),
                variant=model_config.get('variant', 'fp16')
            )
        elif pipeline_type == 'amused':
            return AmusedPipeline.from_pretrained(
                model_name,
                variant=model_config.get('variant', 'fp16'),
                torch_dtype=model_config.get('torch_dtype', torch.float16)
            )
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
            
    def save_metrics(self, model_name):
        min_cpu, max_cpu = min(self.cpu_usage), max(self.cpu_usage)
        min_ram, max_ram = min(self.ram_usage), max(self.ram_usage)
        min_gpu, max_gpu = min(self.gpu_usage), max(self.gpu_usage)
        min_vram, max_vram = min(self.vram_usage), max(self.vram_usage)
        
        self.logger.info(f"\nMetrics for model: {model_name}")
        self.logger.info("--------------------")
        self.logger.info(f"CPU - Min: {min_cpu:.2f}%, Max: {max_cpu:.2f}%")
        self.logger.info(f"GPU - Min: {min_gpu:.2f}%, Max: {max_gpu:.2f}%")
        self.logger.info(f"RAM - Min: {min_ram:.2f}GB, Max: {max_ram:.2f}GB")
        self.logger.info(f"VRAM - Min: {min_vram:.2f}GB, Max: {max_vram:.2f}GB")
        
    def plot_utilization(self, model_name):
        plt.figure(figsize=(12, 8))
        
        # CPU
        plt.subplot(2, 2, 1)
        plt.plot(self.timestamps, self.cpu_usage, label='CPU Usage')
        plt.axhline(y=self.baseline_usage['cpu'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.pipeline_loading_time - self.start_time, color='b', linestyle='--', label='Pipeline Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('% Utilization')
        plt.title('CPU (%)')
        
        # RAM
        plt.subplot(2, 2, 2)
        plt.plot(self.timestamps, self.ram_usage, label='RAM Usage')
        plt.axhline(y=self.baseline_usage['ram'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.pipeline_loading_time - self.start_time, color='b', linestyle='--', label='Pipeline Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('GB')
        plt.title('RAM (GB)')
        
        # GPU
        plt.subplot(2, 2, 3)
        plt.plot(self.timestamps, self.gpu_usage, label='GPU Usage')
        plt.axhline(y=self.baseline_usage['gpu'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.pipeline_loading_time - self.start_time, color='b', linestyle='--', label='Pipeline Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('% Utilization')
        plt.title('GPU (%)')
        
        # VRAM
        plt.subplot(2, 2, 4)
        plt.plot(self.timestamps, self.vram_usage, label='VRAM Usage')
        plt.axhline(y=self.baseline_usage['vram'], color='r', linestyle='--', label='Baseline')
        plt.axvline(x=self.pipeline_loading_time - self.start_time, color='b', linestyle='--', label='Pipeline Loaded')
        plt.xlabel('Time (s)')
        plt.ylabel('GB')
        plt.title('VRAM (GB)')
        
        for end_time in self.prompt_end_times:
            for i in range(1, 5):
                plt.subplot(2, 2, i).axvline(x=end_time, color='g', linestyle='--')
                
        for i in range(1, 5):
            plt.subplot(2, 2, i).legend()
            
        plt.tight_layout()
        
        Path('plots').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'plots/utilization_{model_name.replace("/", "_")}_{timestamp}.png')
        plt.close()
        
    def run_benchmark(self):
        for model_config in self.models:
            model_name = model_config['name']
            self.logger.info(f"\nStarting benchmark for model: {model_name}")
            
            self.setup_monitoring()
            
            self.start_time = time.time()
            self.baseline_usage = self.get_baseline_usage()
            monitor_thread = threading.Thread(target=self.monitor_resources)
            monitor_thread.start()
            
            try:
                pipe = self.get_pipeline(model_config)
                pipe.to("cuda")
                self.pipeline_loading_time = time.time()
                
                times = []
                for i, prompt in enumerate(self.prompts):
                    start_prompt_time = time.time()
                    
                    if model_config['type'] == 'amused':
                        image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
                    else:
                        image = pipe(
                            prompt, 
                            height=self.target_size[1], 
                            width=self.target_size[0]
                        ).images[0]
                        
                    end_prompt_time = time.time()
                    elapsed_time = end_prompt_time - start_prompt_time
                    times.append(elapsed_time)
                    
                    Path('images').mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'images/{model_name.replace("/", "_")}_image_{i+1}_{timestamp}.png'
                    image = image.resize(self.target_size, Image.LANCZOS)
                    image.save(filename)
                    
                    self.logger.info(f"Generated image {i+1}/{len(self.prompts)} in {elapsed_time:.2f} seconds")
                    self.prompt_end_times.append(end_prompt_time - self.start_time)
                    torch.cuda.empty_cache()
                
                avg_time = sum(times) / len(times)
                self.logger.info(f"Average generation time: {avg_time:.2f} seconds")
                
                self.monitoring = False
                monitor_thread.join()
                self.save_metrics(model_name)
                self.plot_utilization(model_name)
                
            except Exception as e:
                self.logger.error(f"Error processing model {model_name}: {str(e)}")
                self.monitoring = False
                monitor_thread.join()
                
            finally:
                torch.cuda.empty_cache()
                if 'pipe' in locals():
                    del pipe

def main():
    benchmark = ModelBenchmark('model_config.yaml', 'prompts.txt')
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
