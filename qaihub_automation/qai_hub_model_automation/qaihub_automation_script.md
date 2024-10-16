# MobileNet V2 Inference on QAI Hub

This script demonstrates how to use the QAI Hub SDK to compile, profile, and run inference with a MobileNet V2 model on a Samsung Galaxy S24 device.

## Prerequisites

Before running the script, ensure you have the following installed:

1. Python 3.7 or later
2. PyTorch and torchvision
3. QAI Hub SDK
4. Pillow
5. NumPy
6. Requests

You can install these dependencies using pip:

```
pip install torch torchvision qai_hub pillow numpy requests
```

If you encounter issues installing `qai_hub`, try installing it directly from the GitHub repository:

```
pip install git+https://github.com/qai-hub/qai-hub-sdk.git
```

## Script Overview

The script performs the following steps:

1. Imports necessary libraries and defines helper functions.
2. Loads a pre-trained MobileNet V2 model.
3. Compiles the model for the target device (Samsung Galaxy S24).
4. Profiles the compiled model.
5. Runs inference on a sample image.
6. Processes and displays the top 5 predictions.
7. Downloads the compiled model.

## Key Functions

- `compile_model()`: Compiles the PyTorch model for the target device.
- `profile_model()`: Profiles the compiled model on the target device.
- `run_inference()`: Runs inference using the compiled model.
- `process_image()`: Preprocesses the input image for inference.

## Running the Script

1. Save the script to a file (e.g., `qaihub_automation_script.py`).
2. Open a terminal and navigate to the directory containing the script.
3. Run the script using Python:

```
python qaihub_automation_script.py
```

## Expected Output

The script will provide progress updates as it runs:

1. "Compiling model..."
2. "Model compiled successfully."
3. "Profiling model..."
4. "Profile job submitted. Job ID: [ID]"
5. "Profiling completed."
6. "Running inference..."
7. "Inference completed."
8. Top 5 predictions for the input image.
9. "Compiled model downloaded as 'mobilenet_v2.tflite'"

## Customization

- To use a different model, replace `mobilenet_v2(pretrained=True)` with your desired model.
- To target a different device, change the `Device` parameter in the `main()` function.
- To use a different input image, update the `sample_image_url` in the `main()` function.

## Troubleshooting

If you encounter any errors:

1. Ensure all dependencies are correctly installed.
2. Check that you have an active internet connection for downloading the model and sample image.
3. Verify that you have the necessary permissions to write files in the script's directory.
4. If you receive API-related errors, ensure you have proper authentication set up for the QAI Hub SDK.
