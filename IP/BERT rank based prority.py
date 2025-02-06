import zipfile
import os
import pandas as pd
import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import os

# Disable WANDB for this example
os.environ["WANDB_DISABLED"] = "true"

zip_file_path = '/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv.zip'
extract_dir = '/content/extracted_files'

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

extracted_files = os.listdir(extract_dir)
print("Extracted files:", extracted_files)

csv_file_path = os.path.join(extract_dir, extracted_files[0])

df = pd.read_csv(csv_file_path)


print("Columns in the CSV file:")
print(df.columns)

print("\nSample content:")
print(df.head())



df = pd.read_csv(csv_file_path)


ingredients_list = df['Cleaned_Ingredients'].apply(ast.literal_eval).explode()

def clean_ingredient(ingredient):
    ingredient = ingredient.lower().strip()
    ingredient = re.sub(r'\b\d+\s*\/\s*\d+\b', '', ingredient)  # Remove fractions like 1/2
    ingredient = re.sub(r'\b\d+\b', '', ingredient)  # Remove standalone numbers
    ingredient = re.sub(r'\b(?:grams?|ounces?|cups?|teaspoons?|tablespoons?|pounds?|kg|g|ml|l)\b', '', ingredient)
    ingredient = re.sub(r'[^\w\s]', '', ingredient)  # Remove special characters
    ingredient = ingredient.strip()
    return ingredient

cleaned_ingredients = ingredients_list.apply(clean_ingredient)

cleaned_ingredients = cleaned_ingredients[cleaned_ingredients != '']

ingredient_counts = Counter(cleaned_ingredients)

unique_ingredients = len(ingredient_counts)
print(f"Number of unique ingredients: {unique_ingredients}")

top_ingredients = pd.Series(ingredient_counts).sort_values(ascending=False)

print("Top 5,000 unique ingredients:")
print(top_ingredients.head(5000))


top_ingredients_5000 = top_ingredients.head(5000)


print("Top 5,000 unique ingredients:")
print(top_ingredients_5000)


with open('top_ingredients.txt', 'w') as file:
    for ingredient in top_ingredients_5000.index:
        file.write(f"{ingredient}\n")

print("File saved with ingredient names.")

import gensim
from gensim.models import Word2Vec

with open('top_ingredients.txt', 'r') as file:
    ingredients = file.readlines()

ingredients = [ingredient.strip() for ingredient in ingredients]

sentences = [[ingredient] for ingredient in ingredients]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def generate_substitutions(ingredient, topn=3):
    similar_words = model.wv.most_similar(ingredient, topn=topn)
    return [word for word, _ in similar_words]

with open('dataset_io.txt', 'w') as file:
    for ingredient in ingredients:
        substitutions = generate_substitutions(ingredient)
        print(f"Substitutes for {ingredient}: {', '.join(substitutions)}")
        file.write(f"{ingredient}: {', '.join(substitutions)}\n")

print("Substitutions saved to 'dataset_io.txt'.")

with open('dataset_io.txt', 'r') as file:
    data = file.readlines()

generated_sentences = []

for line in data:
    ingredient, substitutes = line.split(':')
    ingredient = ingredient.strip()
    substitutes = substitutes.strip().split(', ')
    for substitute in substitutes:
        sentence = f"Possible substitution of {ingredient} is {substitute}."
        generated_sentences.append(sentence)

with open('generated_substitutions.txt', 'w') as file:
    for sentence in generated_sentences:
        file.write(f"{sentence}\n")

print("Generated sentences saved to 'generated_substitutions.txt'.")

file_path = '/content/dataset_io.txt'

with open(file_path, 'r') as file:
    data = file.readlines()

ingredient_substitutes = {}

for line in data:
    line = line.strip()
    if ':' in line:
        ingredient, substitutes_str = line.split(":", 1)
        substitutes = [sub.strip() for sub in substitutes_str.split(',')]
        ingredient_substitutes[ingredient.strip()] = substitutes

print(ingredient_substitutes)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def create_masked_sentences(data):
    sentences = []
    for ingredient, alternatives in data.items():
        for alt in alternatives:
            sentence = f"Instead of {ingredient}, you can use [MASK]."
            sentences.append((sentence, ingredient))
    return sentences

data = ingredient_substitutes
masked_sentences = create_masked_sentences(data)

def tokenize_data(sentences):
    input_texts = []
    labels = []
    for sentence, original_word in sentences:
        input_tokens = tokenizer.encode(sentence, add_special_tokens=True)
        input_texts.append(input_tokens)
        label_tokens = tokenizer.encode(original_word, add_special_tokens=True)
        labels.append(label_tokens)
    return input_texts, labels

input_data, output_data = tokenize_data(masked_sentences)

class MaskedLanguageDataset(Dataset):
    def __init__(self, input_data, output_data, max_len=20):
        self.input_data = input_data
        self.output_data = output_data
        self.max_len = max_len

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_text = self.input_data[idx]
        output_text = self.output_data[idx]

        input_tokens = tokenizer.encode_plus(
            input_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )

        output_tokens = tokenizer.encode_plus(
            output_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )

        input_text_tensor = input_tokens['input_ids'].squeeze(0)
        output_text_tensor = output_tokens['input_ids'].squeeze(0)

        # Return as a dictionary
        return {
            'input_ids': input_text_tensor,
            'labels': output_text_tensor
        }

# Collate function for DataLoader
def collate_fn(batch):
    input_tensors, output_tensors = zip(*batch)
    input_tensors = torch.stack(input_tensors)
    output_tensors = torch.stack(output_tensors)

    return {
        'input_ids': input_tensors,
        'labels': output_tensors
    }

from torch.utils.data import DataLoader, random_split

dataset = MaskedLanguageDataset(input_data, output_data)

train_size = int(0.6 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Model setup
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset,
    eval_dataset= val_dataset,
)

trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Use the trained model for inference
def predict_ingredient(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
        predicted_token_id = predictions[0, mask_token_index].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
    return predicted_token

# Example prediction
sentence = "Instead of milk, you can use [MASK]."
predicted_ingredient = predict_ingredient(sentence)
print(f"Predicted ingredient: {predicted_ingredient}")
