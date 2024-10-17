# Model ID Fetcher and Runner Script Documentation

## Overview

This script automates the process of fetching model IDs from QAI Hub and running them through the main processing script (`qaihub_batch_model_inputs.py`). It's designed to work in conjunction with the existing batch processing script, providing a streamlined way to process multiple models without manually specifying each model ID.

## Features

- Fetches model IDs from QAI Hub in batches
- Allows limiting the number of models to process
- Supports specifying a particular device for processing
- Integrates with the existing `qaihub_batch_model_inputs.py` script
- Provides detailed logging for tracking progress and troubleshooting

## Requirements

- Python 3.6+
- `qai_hub` library
- Access to QAI Hub
- The `qaihub_batch_model_inputs.py` script in the same directory

## Usage

1. Ensure that both `model_id_fetcher.py` and `qaihub_batch_model_inputs.py` are in the same directory.

2. Run the script from the command line:

   ```
   python model_id_fetcher.py [--max_models MAX_MODELS] [--device DEVICE]
   ```

   Options:
   - `--max_models`: (Optional) Maximum number of models to process. If not specified, all available models will be processed.
   - `--device`: (Optional) Specific device to use for all models. If not specified, the script will use compatible devices as determined by the main processing script.

## Function Descriptions

### `fetch_model_ids(batch_size=5, max_models=None)`

Fetches model IDs from QAI Hub.

- Parameters:
  - `batch_size` (int): Number of models to fetch in each API call. Default is 5.
  - `max_models` (int, optional): Maximum number of model IDs to fetch. If None, fetches all available models.
- Returns:
  - List of model IDs (str)
- Raises:
  - Exception: If there's an error fetching model IDs

### `run_main_script(model_ids, device=None)`

Runs the main processing script (`qaihub_batch_model_inputs.py`) with the fetched model IDs.

- Parameters:
  - `model_ids` (list): List of model IDs to process
  - `device` (str, optional): Specific device to use for processing all models
- Raises:
  - subprocess.CalledProcessError: If the main script execution fails

### `main()`

Main function that parses command-line arguments, fetches model IDs, and runs the main processing script.

- Parses command-line arguments for `max_models` and `device`
- Calls `fetch_model_ids()` and `run_main_script()`
- Handles exceptions and logs errors

## Logging

The script logs information, warnings, and errors to both a file (`model_id_fetcher.log`) and the console. This helps in tracking the script's progress and troubleshooting any issues.

## Notes

1. The script assumes that `qaihub_batch_model_inputs.py` is in the same directory and accepts `--model_ids` and `--device` as command-line arguments.

2. If the `--max_models` option is not specified, the script will attempt to process all available models, which may take a considerable amount of time depending on the number of models in QAI Hub.

3. The script fetches models in batches to optimize API calls and manage memory usage, especially when dealing with a large number of models.

4. Error handling is implemented to catch and log issues during model ID fetching and main script execution.

5. Make sure you have the necessary permissions and API access to fetch models from QAI Hub before running this script.

## Troubleshooting

- If you encounter authentication issues, ensure that you're properly logged in to QAI Hub and have the necessary permissions.
- Check the `model_id_fetcher.log` file for detailed error messages if the script fails.
- Verify that `qaihub_batch_model_inputs.py` is in the same directory and accepts the expected command-line arguments.

For further assistance or to report issues, please contact the script maintainer or QAI Hub support.
