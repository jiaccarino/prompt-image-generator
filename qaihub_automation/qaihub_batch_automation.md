# QAI Hub Model Processing Script

## Overview

This script automates the process of fetching, compiling, and running inference on compatible models using the Qualcomm AI Hub (QAI Hub) platform. It performs the following main tasks:

1. Fetches available models from QAI Hub
2. Retrieves available devices
3. Checks compatibility between models and devices
4. Compiles compatible models for their respective devices
5. Runs inference on compiled models using a sample image
6. Displays the top 5 predictions for each model's inference results

## Requirements

- Python 3.6+
- `qai_hub` package (Qualcomm AI Hub SDK)
- `numpy`
- `requests`
- `Pillow` (PIL)

To install the required packages, run:

```
pip install qai_hub numpy requests Pillow
```

Note: The `qai_hub` package might need to be installed from a specific source provided by Qualcomm.

## Usage

1. Ensure you have the necessary credentials and permissions set up for QAI Hub.
2. Run the script:

```
python qai_hub_model_processing.py
```

## Script Components

### Main Functions

1. `fetch_models(batch_size=5)`: 
   - Fetches models from QAI Hub in batches.
   - Returns a list of model objects.

2. `fetch_devices()`: 
   - Retrieves available devices from QAI Hub.
   - Returns a list of device objects.

3. `compile_model(model, device, input_shape)`: 
   - Submits a compile job for a given model and device.
   - Returns the compiled model.

4. `run_inference(model, device, input_data)`: 
   - Submits an inference job for a compiled model.
   - Returns the inference results.

5. `process_image(url, input_shape)`: 
   - Downloads and processes an image from a given URL.
   - Resizes and normalizes the image to match the required input shape.

6. `is_device_compatible(model_device, available_device)`: 
   - Checks if a model's device is compatible with an available device.

### Main Workflow

1. Fetch models and devices from QAI Hub.
2. Process a sample image for inference.
3. Check each model for compatibility with available devices.
4. For each compatible model-device pair:
   - Compile the model for the device.
   - Run inference using the compiled model and sample image.
   - Process and display the top 5 predictions from the inference results.

## Notes

- The script assumes all models expect an input shape of (1, 3, 224, 224). Adjust if necessary.
- Compilation and inference can be time-consuming, especially for larger models or numerous compatible pairs.
- Ensure you have sufficient quota and permissions on QAI Hub to perform these operations.
- The script currently doesn't load specific class labels, so predictions are displayed as class indices.

## Troubleshooting

If you encounter issues:

1. Verify your QAI Hub credentials and permissions.
2. Check your internet connection, as the script requires online access to QAI Hub.
3. Ensure you have the latest version of the `qai_hub` package installed.
4. If specific models or devices cause errors, try running the script with a subset of models or devices for debugging.

## Customization

- To use a different sample image, modify the `sample_image_url` in the `main()` function.
- To process a different number of models at once, adjust the `batch_size` in the `main()` function.
- To add support for different model output formats, modify the output processing section in the main loop.

For further assistance or to report issues, please contact the Qualcomm AI Hub support team.
