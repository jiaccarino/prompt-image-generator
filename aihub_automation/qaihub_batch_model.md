# QAI Hub Batch Inference Script Documentation

## Overview

This script demonstrates how to use the QAI Hub SDK to fetch multiple model IDs, compile these models, and run inference in batches. It's designed to process a single input image across multiple models concurrently, providing a way to benchmark or compare different models efficiently.

## Prerequisites

Before running the script, ensure you have the following installed:

1. Python 3.7 or later
2. QAI Hub SDK
3. NumPy
4. Pillow
5. Requests

You can install these dependencies using pip:

```bash
pip install qai_hub numpy pillow requests
```

## Script Structure

The script is structured into several key functions:

1. `fetch_model_ids(limit=10)`
2. `process_image(url, input_shape)`
3. `compile_and_run_inference(model_id, device, input_data)`
4. `batch_inference(model_ids, device_name, input_data, batch_size=5)`
5. `main()`

### Function Descriptions

#### `fetch_model_ids(limit=10)`

This function retrieves a list of available model IDs from QAI Hub.

- **Parameters:**
  - `limit` (int, optional): The maximum number of model IDs to fetch. Default is 10.
- **Returns:**
  - A list of model ID strings.

#### `process_image(url, input_shape)`

This function downloads and processes an image from a given URL to match the required input shape for the models.

- **Parameters:**
  - `url` (str): The URL of the image to process.
  - `input_shape` (tuple): The required shape of the input (e.g., `(1, 224, 224, 3)`).
- **Returns:**
  - A numpy array containing the processed image data.

#### `compile_and_run_inference(model_id, device, input_data)`

This function compiles a single model and runs inference on it.

- **Parameters:**
  - `model_id` (str): The ID of the model to compile and run.
  - `device` (qai_hub.Device): The target device for compilation and inference.
  - `input_data` (numpy.ndarray): The input data for inference.
- **Returns:**
  - A tuple containing the model ID and the inference result (or None if an error occurred).

#### `batch_inference(model_ids, device_name, input_data, batch_size=5)`

This function runs inference on multiple models in batches.

- **Parameters:**
  - `model_ids` (list): A list of model IDs to process.
  - `device_name` (str): The name of the target device.
  - `input_data` (numpy.ndarray): The input data for inference.
  - `batch_size` (int, optional): The number of models to process concurrently. Default is 5.
- **Returns:**
  - A dictionary mapping model IDs to their respective inference results.

#### `main()`

The main function that orchestrates the entire process. It fetches model IDs, sets up the input data, runs batch inference, and prints the results.

## Usage

1. Save the script to a file (e.g., `qai_hub_batch_inference.py`).
2. Open a terminal and navigate to the directory containing the script.
3. Run the script using Python:

```bash
python qai_hub_batch_inference.py
```

## Customization

You can customize the script by modifying the following parameters in the `main()` function:

- `limit` in `fetch_model_ids(limit=20)`: Change the number of model IDs to fetch.
- `device_name`: Set this to match your target device.
- `input_shape`: Adjust based on the requirements of your models.
- `sample_image_url`: Change to use a different input image.
- `batch_size` in `batch_inference(...)`: Adjust the number of concurrent compilations/inferences.

## Expected Output

The script will provide progress updates as it runs:

1. Number of model IDs fetched
2. Compilation progress for each model
3. Inference progress for each model
4. Results for each successful inference, including:
   - Model ID
   - Output shape
   - First few values of the output

## Error Handling

The script includes basic error handling to prevent a single failed model from stopping the entire batch process. If a model fails to compile or run inference, an error message will be printed, and the script will continue with the next model.

## Performance Considerations

- The script uses threading to run compilations and inferences concurrently. Adjust the `batch_size` based on your system's capabilities and any API rate limits.
- Processing many models or using a large batch size may take considerable time and computational resources.

## Limitations

- The script assumes all models can accept the same input shape. You may need to modify the script if working with models that have different input requirements.
- The current implementation processes a single input image across all models. Modify the script if you need to use different inputs for different models.

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are correctly installed.
2. Check that you have an active internet connection for downloading the model and sample image.
3. Verify that you have the necessary permissions and authentication set up for the QAI Hub SDK.
4. If specific models are failing, try running them individually to isolate the issue.

For more information on the QAI Hub SDK and its capabilities, refer to the official QAI Hub documentation.
