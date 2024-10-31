#!/bin/bash

# Load required modules
module purge
module load anaconda
module load cuda/12.1.1

# Activate environment
source activate biggan

# Verify environment is working
echo "Verifying CUDA setup..."
nvidia-smi
echo "Verifying PyTorch setup..."
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
if [ $? -eq 0 ]; then
    python -c "import torch; print('CUDA version:', torch.version.cuda)"
    python -c "import torch; print('Device count:', torch.cuda.device_count())"
    python -c "import torch; print('Device name:', torch.cuda.get_device_name(0))"
fi
