# Install required libraries if not already installed
!pip install transformers

##################### Verify GPU Availability #####################
import tensorflow as tf
tf.test.gpu_device_name()  # Should return a GPU device name if available

# Using GPU with Transformers
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)  # Should output 'Using device: cuda'

##################### Load the Model and Tokenizer on GPU #####################
# Set device explicitly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use a pipeline as a high-level helper
from transformers import pipeline

ner_pipeline = pipeline("text-generation", model="Universal-NER/UniNER-7B-all", device=0)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Universal-NER/UniNER-7B-all")
model = AutoModelForCausalLM.from_pretrained("Universal-NER/UniNER-7B-all")

# Move the model to GPU
model.to(device)

##################### Clone a GitHub Repository #####################
# Clone the repository
!git clone "https://github.com/JiangYunfan1999/universal-ner_IEOR.git"

# Navigate to the repository folder
%cd universal-ner_IEOR/target_model/instruction_data/test

import os
import json

json_files = [f for f in os.listdir() if f.endswith('.json')]

# Load the JSON file
json_file = json_files[0]
with open(json_file, "r") as f:
    data = json.load(f)

#Example output
#print(data)

##################### Process Each JSON File and Extract Named Entities #####################
# Function to process a single JSON file and return named entities
def process_json_file(json_file):
    with open(json_file, 'r') as f:
        # Assuming the JSON file contains a 'text' key with the data to process
        data = json.load(f)
        text = data.get('text', '')

    # Extract named entities from the text
    entities = ner_pipeline(text)

    return entities

# Process all JSON files and store the results
results = {}

for json_file in json_files:
    data_folder = ''
    full_path = os.path.join(data_folder, json_file)

    # Extract entities for each file
    entities = process_json_file(full_path)

    # Store the results in a dictionary with the file name as the key
    results[json_file] = entities

##################### Store the Output ##################### 
import pandas as pd

# Create a DataFrame to store results
results_list = []

for json_file, entities in results.items():
    for entity in entities:
        results_list.append({
            'file_name': json_file,
            'entity_text': entity['word'],
            'entity_type': entity['entity_group'],
            'start_pos': entity['start'],
            'end_pos': entity['end']
        })

# Convert the list to a DataFrame
df = pd.DataFrame(results_list)

# Store the DataFrame to a CSV file
output_csv_path = "/path/to/output/entities_output.csv"
df.to_csv(output_csv_path, index=False)
