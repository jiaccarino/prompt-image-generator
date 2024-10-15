# First, ensure you have installed the qai_hub package:
# pip install qai_hub

# If the above doesn't work, you may need to use:
# pip install git+https://github.com/qai-hub/qai-hub-sdk.git

import qai_hub
import numpy as np
import torch
from torchvision.models import mobilenet_v2
from PIL import Image
import requests

def compile_model(model, device, input_shape):
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=dict(image=input_shape),
        options="--target_runtime tflite"
    )
    return compile_job.get_target_model()

def profile_model(model, device):
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
    )
    return profile_job

def run_inference(model, device, input_data):
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=dict(image=[input_data]),
    )
    return inference_job.download_output_data()

def process_image(url, input_shape):
    response = requests.get(url, stream=True)
    response.raw.decode_content = True
    image = Image.open(response.raw).resize((input_shape[2], input_shape[3]))
    input_array = np.expand_dims(
        np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
    )
    return input_array

def main():
    # Set up the model and device
    torch_model = mobilenet_v2(pretrained=True)
    torch_model.eval()
    input_shape = (1, 3, 224, 224)
    device = qai_hub.Device("Samsung Galaxy S24 (Family)")

    # Trace the model
    example_input = torch.rand(input_shape)
    traced_torch_model = torch.jit.trace(torch_model, example_input)

    # Compile the model
    print("Compiling model...")
    target_model = compile_model(traced_torch_model, device, input_shape)
    print("Model compiled successfully.")

    # Profile the model
    print("Profiling model...")
    profile_job = profile_model(target_model, device)
    print("Profile job submitted. Job ID:", profile_job.job_id)

    # Wait for the profile job to complete
    profile_job.wait()
    print("Profiling completed.")

    print("Running inference...")
    sample_image_url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
    input_data = process_image(sample_image_url, input_shape)
    output = run_inference(target_model, device, input_data)
    print("Inference completed.")

    # Process and display results
    output_name = list(output.keys())[0]
    out = output[output_name]
    print(f"Output shape: {out.shape}")
    
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]  # Remove the batch dimension if it exists
    elif out.ndim != 1:
        raise ValueError(f"Unexpected output shape: {out.shape}. Expected a 1D array or 2D array with shape (1, n).")

    print(f"Processed output shape: {out.shape}")

    # Ensure we're working with numpy array
    out = np.array(out)

    probabilities = np.exp(out) / np.sum(np.exp(out), axis=0)

    # Load ImageNet class labels
    class_labels_url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(class_labels_url)
    categories = [s.strip() for s in response.text.splitlines()]

    # Print top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    print("\nTop 5 predictions:")
    for i, idx in enumerate(top5_indices, 1):
        print(f"{i}. {categories[idx]}: {probabilities[idx]:.2%}")

    # Download the compiled model
    target_model.download("mobilenet_v2.tflite")
    print("\nCompiled model downloaded as 'mobilenet_v2.tflite'")

if __name__ == "__main__":
    main()
