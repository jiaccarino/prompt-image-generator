import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import json

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load YOLO-OpenWorld dataset classes
def load_dataset_classes(file_path):
    with open(file_path, 'r') as f:
        return set(json.load(f))

coco_classes = load_dataset_classes('coco_classes.json')
lvis_classes = load_dataset_classes('lvis_classes.json')
objects365_classes = load_dataset_classes('objects365_classes.json')

all_classes = coco_classes.union(lvis_classes).union(objects365_classes)

lemmatizer = WordNetLemmatizer()

def preprocess_prompt(prompt):
    # Tokenize and lemmatize words in the prompt
    tokens = word_tokenize(prompt.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]
    return set(lemmatized)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def prompt_aligns_with_datasets(prompt_words):
    for word in prompt_words:
        if word in all_classes or any(synonym in all_classes for synonym in get_synonyms(word)):
            return True
    return False

def evaluate_prompts(input_file, output_file, removed_file):
    unique_prompts = set()
    removed_prompts = []
    total_prompts = 0

    with open(input_file, 'r') as f:
        prompts = f.readlines()

    for prompt in prompts:
        total_prompts += 1
        prompt = prompt.strip()
        prompt_words = preprocess_prompt(prompt)

        if prompt not in unique_prompts and prompt_aligns_with_datasets(prompt_words):
            unique_prompts.add(prompt)
        else:
            removed_prompts.append(prompt)

    # Write valid prompts to output file
    with open(output_file, 'w') as f:
        for prompt in unique_prompts:
            f.write(f"{prompt}\n")

    # Write removed prompts to a separate file
    with open(removed_file, 'w') as f:
        for prompt in removed_prompts:
            f.write(f"{prompt}\n")

    removed_count = total_prompts - len(unique_prompts)
    print(f"Removed {removed_count} prompts out of {total_prompts} total prompts.")
    print(f"Remaining prompts: {len(unique_prompts)}")

if __name__ == "__main__":
    input_file = "input_prompts.txt"
    output_file = "valid_prompts.txt"
    removed_file = "removed_prompts.txt"
    evaluate_prompts(input_file, output_file, removed_file)
