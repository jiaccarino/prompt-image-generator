#!/usr/bin/env python3

import subprocess
import sys
import platform
from pathlib import Path
import logging
import json
import os
import shutil

class ClusterSetupManager:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging
        
    def run_command(self, command, error_message):
        try:
            self.logger.info(f"Running: {' '.join(command)}")
            subprocess.check_call(command)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{error_message}: {str(e)}")
            return False
            
    def clean_torch_install(self):
        """Remove existing PyTorch installations to avoid conflicts"""
        self.logger.info("Cleaning existing PyTorch installations...")
        
        # Remove pip-installed torch
        pip_cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision']
        self.run_command(pip_cmd, "Warning: Could not uninstall pip torch")
        
        # Remove conda-installed torch
        conda_cmd = ['conda', 'remove', '-y', '--force', 'pytorch', 'torchvision', 'cudatoolkit']
        self.run_command(conda_cmd, "Warning: Could not uninstall conda torch")
        
    def get_cuda_version(self):
        """Get CUDA version from nvidia-smi"""
        try:
            output = subprocess.check_output(['nvidia-smi']).decode()
            import re
            match = re.search(r'CUDA Version: (\d+\.\d+)', output)
            if match:
                return match.group(1)
        except:
            pass
        return None
        
    def setup_conda_env(self):
        """Setup clean conda environment"""
        env_name = 'biggan'
        
        # Check if environment exists
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if env_name in result.stdout:
            self.logger.info(f"Removing existing {env_name} environment...")
            subprocess.run(['conda', 'env', 'remove', '-n', env_name, '-y'])
        
        # Create new environment
        self.logger.info(f"Creating new {env_name} environment...")
        create_cmd = ['conda', 'create', '-n', env_name, 'python=3.11', '-y']
        if not self.run_command(create_cmd, "Error creating conda environment"):
            return False
            
        # Get the path to the conda executable in the new environment
        if platform.system() == 'Windows':
            python_path = f"conda run -n {env_name} python"
        else:
            python_path = f"{os.path.dirname(sys.executable)}/conda run -n {env_name} python"
            
        return python_path
        
    def install_packages(self, python_path):
        """Install required packages"""
        cuda_version = self.get_cuda_version()
        self.logger.info(f"Detected CUDA version: {cuda_version}")
        
        # Install PyTorch with correct CUDA version
        if cuda_version:
            cuda_major = cuda_version.split('.')[0]
            if cuda_major == '12':
                torch_cmd = [
                    'conda', 'install', '-n', 'biggan', '-y',
                    'pytorch', 'torchvision', 'pytorch-cuda=11.8',
                    '-c', 'pytorch', '-c', 'nvidia'
                ]
            else:
                torch_cmd = [
                    'conda', 'install', '-n', 'biggan', '-y',
                    'pytorch', 'torchvision', 'cudatoolkit=11.8',
                    '-c', 'pytorch'
                ]
        else:
            torch_cmd = [
                'conda', 'install', '-n', 'biggan', '-y',
                'pytorch', 'torchvision', 'cpuonly',
                '-c', 'pytorch'
            ]
            
        if not self.run_command(torch_cmd, "Error installing PyTorch"):
            return False
            
        # Install other requirements
        pip_cmd = [
            'conda', 'run', '-n', 'biggan', 'pip', 'install',
            'transformers', 'pillow', 'matplotlib', 'psutil'
        ]
        if not self.run_command(pip_cmd, "Error installing pip packages"):
            return False
            
        return True
        
    def create_config_file(self):
        """Create default configuration file"""
        config = {
            "prompts": [
                "a red rose in full bloom",
                "a snowy mountain peak at sunset",
                "a golden retriever puppy playing"
            ],
            "target_size": [512, 512],
            "truncation": 0.4,
            "batch_size": 1,
            "model_size": "512"
        }
        
        config_path = Path('biggan_config.json')
        if not config_path.exists():
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info("Created default configuration file: biggan_config.json")
        else:
            self.logger.info("Configuration file already exists")
            
    def create_activation_script(self):
        """Create activation script"""
        activate_content = """#!/bin/bash
module purge
module load anaconda
module load cuda/11.8
conda activate biggan
"""
        
        with open('activate_env.sh', 'w') as f:
            f.write(activate_content)
        
        # Make it executable
        os.chmod('activate_env.sh', 0o755)
        self.logger.info("Created activation script: activate_env.sh")
        
    def setup(self):
        try:
            self.logger.info("Starting setup process...")
            
            # Clean existing installations
            self.clean_torch_install()
            
            # Setup conda environment
            python_path = self.setup_conda_env()
            if not python_path:
                return
                
            # Install packages
            if not self.install_packages(python_path):
                return
                
            # Create config and activation files
            self.create_config_file()
            self.create_activation_script()
            
            self.logger.info("\nSetup completed successfully!")
            self.logger.info("\nTo use the environment:")
            self.logger.info("1. source activate_env.sh")
            self.logger.info("2. Run your BigGAN script")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            sys.exit(1)

def main():
    setup_manager = ClusterSetupManager()
    setup_manager.setup()

if __name__ == "__main__":
    main()