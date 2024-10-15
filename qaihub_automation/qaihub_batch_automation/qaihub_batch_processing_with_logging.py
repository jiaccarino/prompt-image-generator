import qai_hub
import numpy as np
import requests
from PIL import Image
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qaihub_batch_processing.log'),
        logging.StreamHandler()
    ]
)

def compile_model(model, device, input_shape):
    """Submit a compile job for the model."""
    logging.info(f"Compiling model {model.model_id} for device {device.name}")
    try:
        compile_job = qai_hub.submit_compile_job(
            model=model,
            device=device,
            input_specs=dict(image=input_shape),
            options="--target_runtime tflite"
        )
        return compile_job.get_target_model()
    except Exception as e:
        logging.error(f"Error compiling model {model.model_id}: {str(e)}")
        raise

def run_inference(model, device, input_data):
    """Submit an inference job for the model."""
    logging.info(f"Running inference for model {model.model_id} on device {device.name}")
    try:
        inference_job = qai_hub.submit_inference_job(
            model=model,
            device=device,
            inputs=dict(image=[input_data]),
        )
        return inference_job.download_output_data()
    except Exception as e:
        logging.error(f"Error running inference for model {model.model_id}: {str(e)}")
        raise

def process_image(url, input_shape):
    """Process an image from a URL to the required input shape."""
    logging.info(f"Processing image from URL: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        response.raw.decode_content = True
        image = Image.open(response.raw).resize((input_shape[2], input_shape[3]))
        input_array = np.expand_dims(
            np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
        )
        return input_array
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

def fetch_models(batch_size=5):
    """Fetch models in batches from QAI Hub."""
    logging.info(f"Fetching models with batch size: {batch_size}")
    models = []
    offset = 0
    try:
        while True:
            batch = qai_hub.get_models(limit=batch_size, offset=offset)
            if not batch:
                break
            models.extend(batch)
            offset += batch_size
            if len(batch) < batch_size:
                break  # No more models to fetch
        logging.info(f"Fetched {len(models)} models in total")
        return models
    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        raise

def fetch_devices():
    """Fetch available devices from QAI Hub."""
    logging.info("Fetching available devices")
    try:
        devices = qai_hub.get_devices()
        logging.info(f"Fetched {len(devices)} devices")
        return devices
    except Exception as e:
        logging.error(f"Error fetching devices: {str(e)}")
        raise

def is_device_compatible(model_device, available_device):
    """Check if the model's device is compatible with an available device."""
    if model_device.name == available_device.name:
        return True
    if 'family' in model_device.name.lower() and model_device.name.split()[:-1] == available_device.name.split():
        return True
    return False

def main():
    logging.info("Starting QAI Hub model processing script")
    
    # Fetch models in batches
    batch_size = 5  # Number of models to fetch at a time
    models = fetch_models(batch_size)

    # Fetch available devices
    devices = fetch_devices()
    logging.info(f"Available devices: {[device.name for device in devices]}")

    # Set up input shape
    input_shape = (1, 3, 224, 224)  # Assuming models expect this shape
    sample_image_url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
    input_data = process_image(sample_image_url, input_shape)

    compatible_models = []  # List to hold models with compatible devices

    # Loop through models and check compatibility
    for model in models:
        try:
            logging.info(f"Checking model {model.model_id} for compatible devices")
            logging.debug(f"Model structure: {vars(model)}")

            # Check if the model has a compatible device
            compatible_device = None
            if hasattr(model, 'producer') and hasattr(model.producer, 'device'):
                model_device = model.producer.device
                for device in devices:
                    if is_device_compatible(model_device, device):
                        compatible_device = device
                        break
            
            if compatible_device is None:
                logging.info(f"No compatible device found for model {model.model_id}. Skipping.")
                continue
            
            logging.info(f"Compatible device found: {compatible_device.name} for model {model.model_id}")

            # Add to the list of compatible models
            compatible_models.append((model, compatible_device))

        except Exception as e:
            logging.error(f"Error checking model {model.model_id}: {str(e)}")

    # Summary of compatible models
    logging.info("\nSummary of models with compatible devices:")
    for model, device in compatible_models:
        logging.info(f"Model ID: {model.model_id}, Model Name: {model.name}, Compatible Device: {device.name}")

    # Compile and run inference on compatible models
    for model, device in compatible_models:
        try:
            compiled_model = compile_model(model, device, input_shape)
            logging.info(f"Model {model.model_id} compiled successfully")

            output = run_inference(compiled_model, device, input_data)
            
            # Process and display results
            output_name = list(output.keys())[0]
            out = output[output_name]
            logging.info(f"Output shape: {out.shape}")
            if out.ndim == 3 and out.shape[0] == 1:
                out = out[0]  # Remove the batch dimension if it exists
            elif out.ndim != 1:
                raise ValueError(f"Unexpected output shape: {out.shape}. Expected a 1D array or 2D array with shape (1, n).")
            
            # Ensure we're working with numpy array
            out = np.array(out)
            probabilities = np.exp(out) / np.sum(np.exp(out), axis=0)

            # Log top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            logging.info(f"\nTop 5 predictions for model {model.model_id}:")
            for i, idx in enumerate(top5_indices, 1):
                logging.info(f"{i}. Class {idx}: {probabilities[idx]:.2%}")

        except Exception as e:
            logging.error(f"Error compiling or running inference for model {model.model_id}: {str(e)}")

    logging.info("QAI Hub model processing script completed")

if __name__ == "__main__":
    main()