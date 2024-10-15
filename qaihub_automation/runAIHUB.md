# AIHub Model Workflow Script Documentation

This document provides instructions on how to set up, run, and use the AIHub Model Workflow script.

## Table of Contents
1. Prerequisites
2. Installation
3. Configuration
4. Running the Script
5. Script Workflow
6. Troubleshooting

## 1. Prerequisites

Before using the script, ensure you have the following:

- Python 3.6 or higher installed on your system
- An active AIHub account
- Your AIHub API key

## 2. Installation

1. Clone or download the script to your local machine.

2. Open a terminal or command prompt and navigate to the directory containing the script.

3. Install the required Python libraries by running:

   ```
   pip install requests python-dotenv
   ```

## 3. Configuration

1. In the same directory as the script, create a file named `.env`.

2. Open the `.env` file in a text editor and add your AIHub API key:

   ```
   AIHUB_API_KEY=your_api_key_here
   ```

   Replace `your_api_key_here` with your actual AIHub API key.

## 4. Running the Script

To run the script:

1. Open a terminal or command prompt.

2. Navigate to the directory containing the script.

3. Run the script using Python:

   ```
   python aihub_model_workflow.py
   ```

   (Replace `aihub_model_workflow.py` with the actual name of the script file if different.)

4. Follow the prompts in the terminal to interact with the script.

## 5. Script Workflow

The script follows this workflow:

1. **Submit Model**: 
   - You'll be prompted to enter the path to your model file.
   - The script will upload the model to AIHub and receive a submission ID.

2. **Retrieve Model ID**: 
   - The script will automatically poll AIHub to check the status of your submitted model.
   - Once the model is ready, it will retrieve the model ID.

3. **Run Model**: 
   - You'll be prompted to enter input data as a JSON string.
   - The script will start an inference job with your model and input data.
   - It will then poll for results and display them once complete.

## 6. Troubleshooting

- If you encounter an "Authorization" error, double-check that your API key in the `.env` file is correct and that the file is in the same directory as the script.

- If the script fails to submit a model, ensure that your model file exists at the path you specified and that you have permission to read it.

- For JSON input errors, make sure your input is valid JSON. You can use an online JSON validator to check your input.

- If the script seems to hang, it may be because AIHub is taking longer than expected to process your model or run inference. The script includes wait times between status checks, so be patient.

- For any other errors, check the error message printed by the script. It often contains helpful information about what went wrong.

If you continue to experience issues, consult the AIHub documentation or contact their support team for assistance.
