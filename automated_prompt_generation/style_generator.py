import random

def add_random_style(input_file, output_file):
    styles = ["ink wash", "water color", "oil painting"]

    # Open the input file for reading and output file for writing
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Add a random style from the list to each prompt
            random_style = random.choice(styles)
            outfile.write(f"{line.strip()} with {random_style}\n")

# File paths
input_file = "updated_style_prompts.txt"
output_file = "new_updated_style_prompts.txt"

# Run the function
add_random_style(input_file, output_file)

print(f"Prompts have been updated and saved to {output_file}.")
