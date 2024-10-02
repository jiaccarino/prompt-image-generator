import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import json
import random

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

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

def generate_new_prompt():
    # Select a random class from the datasets
    base_class = random.choice(list(all_classes))
    
    # Get potential actions (verbs) related to the class
    actions = []
    for synset in wordnet.synsets(base_class):
        for lemma in synset.lemmas():
            related_verbs = lemma.derivationally_related_forms()
            actions.extend([v.name() for v in related_verbs if v.synset().pos() == 'v'])
    
    # If no actions found, use some generic actions
    if not actions:
        actions = ['observe', 'see', 'find', 'discover', 'encounter']
    
    # Generate a simple prompt
    action = random.choice(actions)
    adjectives = ['interesting', 'unique', 'colorful', 'unusual', 'striking']
    adjective = random.choice(adjectives)
    
    return f"{action} a {adjective} {base_class}"

def evaluate_and_replace_prompts(input_file, output_file):
    unique_prompts = set()
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
            # Generate a new prompt to replace the removed one
            while True:
                new_prompt = generate_new_prompt()
                new_prompt_words = preprocess_prompt(new_prompt)
                if new_prompt not in unique_prompts and prompt_aligns_with_datasets(new_prompt_words):
                    unique_prompts.add(new_prompt)
                    break

    # Write all prompts (original valid ones and new replacements) to output file
    with open(output_file, 'w') as f:
        for prompt in unique_prompts:
            f.write(f"{prompt}\n")

    replaced_count = total_prompts - len(unique_prompts)
    print(f"Replaced {replaced_count} prompts out of {total_prompts} total prompts.")
    print(f"Final number of prompts: {len(unique_prompts)}")

if __name__ == "__main__":
    input_file = "input_prompts.txt"
    output_file = "final_prompts.txt"
    evaluate_and_replace_prompts(input_file, output_file)
