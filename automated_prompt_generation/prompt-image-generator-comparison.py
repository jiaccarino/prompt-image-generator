import os
import requests
import json
from dotenv import load_dotenv
from base64 import b64decode

# Load environment variables from the .env file
load_dotenv()

# Define a dictionary to store API configurations
api_configurations = {
    "dalle": {
        "url": "https://api.openai.com/v1/images/generations",
        "key": os.getenv("OPENAI_API_KEY")
    },
    "stable_diffusion": {
        "url": "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        "key": os.getenv("STABILITY_API_KEY")
    },
    "midjourney": {
        "url": "https://api.midjourney.com/api/app/imagine-v1",  # This is a hypothetical URL
        "key": os.getenv("MIDJOURNEY_API_KEY")
    },
    "imagen": {
        "url": "https://imagen-api.googleapis.com/v1/images:generateFromPrompt",  # This is a hypothetical URL
        "key": os.getenv("IMAGEN_API_KEY")
    },
    "dreamstudio": {
        "url": "https://api.dreamstudio.ai/api/v1/generation/text-to-image",
        "key": os.getenv("DREAMSTUDIO_API_KEY")
    }
}

# Choose the desired API configuration
selected_api = "dalle"  # Change this to switch APIs
api_url = api_configurations[selected_api]["url"]
api_key = api_configurations[selected_api]["key"]

# Directory to save generated images
output_dir = "./generated_images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to generate an image based on a prompt using the selected API
def generate_image(prompt, output_format="png", seed=None):
    if selected_api == "dalle":
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json"
            }
        )
        if response.status_code == 200:
            image_data = b64decode(response.json()['data'][0]['b64_json'])
            filename = f"dalle_{prompt[:10].replace(' ', '_')}.png"
        else:
            raise Exception(f"Error: {response.status_code} - {response.json()}")

    elif selected_api == "stable_diffusion":
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            }
        )
        if response.status_code == 200:
            image_data = b64decode(response.json()['artifacts'][0]['base64'])
            filename = f"stable_diffusion_{prompt[:10].replace(' ', '_')}.png"
        else:
            raise Exception(f"Error: {response.status_code} - {response.json()}")

    elif selected_api == "midjourney":
        # Note: Midjourney primarily operates through Discord and does not have a public API
        # This is a hypothetical implementation based on how such an API might work
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
            }
        )
        if response.status_code == 200:
            # Assuming Midjourney would return a URL to the generated image
            image_url = response.json()['image_url']
            image_response = requests.get(image_url)
            image_data = image_response.content
            filename = f"midjourney_{prompt[:10].replace(' ', '_')}.png"
        else:
            raise Exception(f"Error: {response.status_code} - {response.json()}")

    elif selected_api == "imagen":
        # Note: Imagen is not publicly available. This is a hypothetical implementation.
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "prompt": prompt,
                "sampleCount": 1,
                "sampleParams": {
                    "candidates": 1,
                    "numInferenceSteps": 50,
                    "size": "1024x1024"
                }
            }
        )
        if response.status_code == 200:
            # Assuming Imagen would return image data directly
            image_data = b64decode(response.json()['images'][0])
            filename = f"imagen_{prompt[:10].replace(' ', '_')}.png"
        else:
            raise Exception(f"Error: {response.status_code} - {response.json()}")

    elif selected_api == "dreamstudio":
        # DreamStudio uses the Stability AI API
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            }
        )
        if response.status_code == 200:
            image_data = b64decode(response.json()['artifacts'][0]['base64'])
            filename = f"dreamstudio_{prompt[:10].replace(' ', '_')}.png"
        else:
            raise Exception(f"Error: {response.status_code} - {response.json()}")

    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as file:
        file.write(image_data)
    print(f"Image for prompt '{prompt[:50]}...' saved as '{file_path}'")

# Function to read prompts from a text file
def read_prompts_from_file(filename):
    try:
        with open(filename, 'r') as file:
            prompts = [line.strip() for line in file.readlines() if line.strip()]
        return prompts
    except FileNotFoundError:
        raise Exception(f"Error: {filename} not found.")

# Main function to run the image generation for each prompt
def main():
    # Load prompts from a .txt file
    prompts = read_prompts_from_file("prompts.txt")
    
    # Ensure API key is loaded
    if not api_key:
        raise Exception(f"API key for {selected_api} not found. Ensure the .env file is set up correctly.")
    
    # Generate images for each prompt
    for prompt in prompts:
        try:
            generate_image(prompt)
        except Exception as e:
            print(f"Failed to generate image for prompt '{prompt}': {e}")

# Run the main function
if __name__ == "__main__":
    main()
