import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Stability AI API key loaded from the .env file
api_key = os.getenv("STABILITY_API_KEY")  # No hardcoding of the key

# Stability AI API endpoint for image generation
api_url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

# Define the prompt and other parameters
prompt = "A futuristic cityscape at sunset with flying cars and neon lights"
output_format = "webp"  # Can be png, jpeg, or webp
aspect_ratio = "16:9"  # Optional, e.g., 16:9, 1:1, etc.
seed = 123456  # Optional, specify to get consistent results
negative_prompt = "blurry, distorted"  # Optional negative prompt

# Make sure the API key was loaded correctly
if not api_key:
    raise Exception("API key not found. Ensure the .env file is set up correctly.")

# Make the POST request
response = requests.post(
    api_url,
    headers={
        "authorization": f"Bearer {api_key}",
        "accept": "image/*"  # Adjust this to "application/json" if you want a base64 image
    },
    files={"none": ''},  # Empty multipart field
    data={
        "prompt": prompt,
        "output_format": output_format,
        "aspect_ratio": aspect_ratio,  # Optional parameter
        "seed": seed,  # Optional parameter
        "negative_prompt": negative_prompt,  # Optional parameter
    },
)

# Handle the response
if response.status_code == 200:
    # Save the image
    with open(f"./generated_image.{output_format}", 'wb') as file:
        file.write(response.content)
    print(f"Image saved as 'generated_image.{output_format}'")
else:
    # Print error if something goes wrong
    raise Exception(str(response.json()))
