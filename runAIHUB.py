import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AIHub API endpoint and your API key
API_ENDPOINT = "https://app.aihub.qualcomm.com/api/v1"
API_KEY = os.getenv("AIHUB_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def submit_model(model_path):
    """
    Submit a model to AIHub and return the submission ID.
    """
    with open(model_path, 'rb') as model_file:
        files = {'file': model_file}
        response = requests.post(
            f"{API_ENDPOINT}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files=files
        )
    response.raise_for_status()
    return response.json()["id"]

def get_model_id(submission_id):
    """
    Retrieve the model ID using the submission ID.
    """
    while True:
        response = requests.get(
            f"{API_ENDPOINT}/models/{submission_id}",
            headers=headers
        )
        response.raise_for_status()
        status = response.json()["status"]
        
        if status == "READY":
            return response.json()["id"]
        elif status in ["FAILED", "REJECTED"]:
            raise Exception(f"Model submission failed with status: {status}")
        
        time.sleep(10)  # Wait for 10 seconds before checking again

def run_model(model_id, input_data):
    """
    Run a model on AIHub given its ID and input data.
    """
    response = requests.post(
        f"{API_ENDPOINT}/inferences",
        headers=headers,
        json={"model_id": model_id, "input": input_data}
    )
    response.raise_for_status()
    inference_id = response.json()["id"]
    
    while True:
        response = requests.get(
            f"{API_ENDPOINT}/inferences/{inference_id}",
            headers=headers
        )
        response.raise_for_status()
        status = response.json()["status"]
        
        if status == "COMPLETED":
            return response.json()["output"]
        elif status == "FAILED":
            raise Exception("Model inference failed")
        
        time.sleep(5)  # Wait for 5 seconds before checking again

def main():
    model_path = input("Enter the path to your model file: ")
    
    try:
        print("Submitting model...")
        submission_id = submit_model(model_path)
        print(f"Model submitted. Submission ID: {submission_id}")
        
        print("Retrieving model ID...")
        model_id = get_model_id(submission_id)
        print(f"Model ID retrieved: {model_id}")
        
        input_data = input("Enter the input data (as a JSON string): ")
        input_data = json.loads(input_data)
        
        print("Running model inference...")
        result = run_model(model_id, input_data)
        print("Model inference completed. Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
