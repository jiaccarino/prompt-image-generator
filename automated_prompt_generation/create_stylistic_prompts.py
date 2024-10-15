import random

# Yolo World stylistic descriptors
yolo_styles = [
    "in neon colors",
    "with a retro vibe",
    "in minimalist style",
    "with bold typography",
    "using pastel colors",
    "in graffiti art style",
    "with abstract geometric shapes",
    "in pop art style",
    "with a street art aesthetic",
    "using vibrant gradients",
    "in flat design style",
    "with a glitch art effect",
    "using duotone color scheme",
    "in vaporwave aesthetic",
    "with hand-drawn elements",
    "in cyberpunk style",
    "using isometric design",
    "with a low poly effect",
    "in pixel art style",
    "with a holographic look"
]

def add_yolo_style(prompt):
    return f"{prompt.strip()} {random.choice(yolo_styles)}"

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            modified_prompt = add_yolo_style(line)
            outfile.write(modified_prompt + '\n')

# Example usage
input_file = 'prompts.txt'
output_file = 'style_prompts.txt'
process_file(input_file, output_file)
