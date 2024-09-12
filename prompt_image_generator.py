import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Stability AI API key loaded from the .env file
api_key = os.getenv("STABILITY_API_KEY")  # No hardcoding of the key

# Stability AI API endpoint for image generation
api_url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

# Function to generate an image based on a prompt using the API
def generate_image(prompt, output_format="webp", aspect_ratio="16:9", seed=0, negative_prompt=None):
    response = requests.post(
        api_url,
        headers={
            "authorization": f"Bearer {api_key}",
            "accept": "image/*"  # Adjust to "application/json" for base64 encoded output
        },
        files={"none": ''},  # Empty multipart field
        data={
            "prompt": prompt,
            "output_format": output_format,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "negative_prompt": negative_prompt,
        },
    )

    # Handle the response
    if response.status_code == 200:
        # Save the image
        with open(f"./generated_image_{prompt[:10]}.{output_format}", 'wb') as file:
            file.write(response.content)
        print(f"Image for prompt '{prompt[:50]}...' saved as 'generated_image_{prompt[:10]}.{output_format}'")
    else:
        raise Exception(f"Error: {response.status_code} - {response.json()}")

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
        raise Exception("API key not found. Ensure the .env file is set up correctly.")

    # Generate images for each prompt
    for prompt in prompts:
        try:
            generate_image(prompt)
        except Exception as e:
            print(f"Failed to generate image for prompt '{prompt}': {e}")

# Run the main function
if __name__ == "__main__":
    main()
