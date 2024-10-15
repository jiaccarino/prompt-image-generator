import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AIHub API endpoint and your API key from the environment variable
API_ENDPOINT = "https://app.aihub.qualcomm.com/api/v1"
API_KEY = os.getenv("AIHUB_API_KEY")

# Headers for API requests
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def submit_model(model_file_path):
    """
    Submit a model to AIHub and return the submission ID.
    """
    try:
        with open(model_file_path, 'rb') as model_file:
            files = {'file': model_file}
            response = requests.post(
                f"{API_ENDPOINT}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                files=files
            )
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()["id"]
    except Exception as e:
        raise Exception(f"Error submitting model: {str(e)}")

def get_model_id(submission_id):
    """
    Retrieve the model ID using the submission ID. Polls until the model is ready.
    """
    while True:
        try:
            response = requests.get(
                f"{API_ENDPOINT}/models/{submission_id}",
                headers=headers
            )
            response.raise_for_status()
            response_data = response.json()
            status = response_data["status"]

            if status == "READY":
                return response_data["id"]
            elif status in ["FAILED", "REJECTED"]:
                raise Exception(f"Model submission failed with status: {status}")
        except Exception as e:
            raise Exception(f"Error retrieving model ID: {str(e)}")
        
        time.sleep(10)  # Wait 10 seconds before checking status again

def run_model(model_id, input_data):
    """
    Run a model on AIHub using its ID and input data, returning the result.
    """
    try:
        response = requests.post(
            f"{API_ENDPOINT}/inferences",
            headers=headers,
            json={"model_id": model_id, "input": input_data}
        )
        response.raise_for_status()
        inference_id = response.json()["id"]

        # Polling until inference is complete
        while True:
            response = requests.get(
                f"{API_ENDPOINT}/inferences/{inference_id}",
                headers=headers
            )
            response.raise_for_status()
            inference_data = response.json()
            status = inference_data["status"]

            if status == "COMPLETED":
                return inference_data["output"]
            elif status == "FAILED":
                raise Exception("Model inference failed")

            time.sleep(5)  # Wait 5 seconds before checking inference status again
    except Exception as e:
        raise Exception(f"Error during model inference: {str(e)}")

def main():
    model_file_path = input("Enter the path to your model file: ")

    try:
        print("Submitting model...")
        submission_id = submit_model(model_file_path)
        print(f"Model submitted successfully. Submission ID: {submission_id}")

        print("Retrieving model ID...")
        model_id = get_model_id(submission_id)
        print(f"Model ID retrieved: {model_id}")

        input_data_str = input("Enter the input data (as a JSON string): ")
        input_data = json.loads(input_data_str)

        print("Running model inference...")
        result = run_model(model_id, input_data)
        print("Model inference completed. Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
