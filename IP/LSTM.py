import zipfile
import os

zip_path = "/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv.zip"
extract_to = "/content/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction complete!")
import pandas as pd

csv_path = "/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

print(df.columns)
import pandas as pd
csv_path = "/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

filtered_df = df[df['Ingredients'].notna()]

print(filtered_df)
print("This is the filtered Ingredients ")
print()
print()
import pandas as pd
import re
import ast

csv_path = "/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

filtered_df = df[df['Ingredients'].notna()].copy()

def clean_ingredients(ingredient_str):
    try:
        ingredients = ast.literal_eval(ingredient_str)
        cleaned = []
        for item in ingredients:
            item = re.sub(r'[\d½⅓¼¾⅔⅛⅜⅝⅞]+', '', item)
            item = re.sub(r'[-–—]+', ' ', item)
            item = re.sub(r'\s+', ' ', item).strip()
            cleaned.append(item)
        return cleaned
    except:
        return []

filtered_df['Ingredients'] = filtered_df['Ingredients'].apply(clean_ingredients)

print(filtered_df['Ingredients'])

import pandas as pd
import re
import ast

csv_path = "/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

filtered_df = df[df['Ingredients'].notna()].copy()

def clean_ingredients(ingredient_str):
    try:
        ingredients = ast.literal_eval(ingredient_str)
        cleaned = []
        for item in ingredients:
            item = item.lower()
            item = re.sub(r'[\d½⅓¼¾⅔⅛⅜⅝⅞]+', '', item)
            item = re.sub(r'[/\\\-–—]', ' ', item)
            item = re.sub(r'[^\w\s]', '', item)
            item = re.sub(r'\s+', ' ', item).strip()
            cleaned.append(item)
        return cleaned
    except:
        return []

filtered_df['Ingredients'] = filtered_df['Ingredients'].apply(clean_ingredients)

filtered_df.to_csv('/content/cleaned_ingredients.csv', index=False)
print("Cleaned ingredients saved to /content/cleaned_ingredients.csv")

import pandas as pd
import re
import ast

csv_path = "/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
df = pd.read_csv(csv_path)

filtered_df = df[df['Ingredients'].notna()].copy()

def clean_ingredients(ingredient_str):
    try:
        ingredients = ast.literal_eval(ingredient_str)
        cleaned = []
        for item in ingredients:
            item = item.lower()
            item = re.sub(r'[\d½⅓¼¾⅔⅛⅜⅝⅞]+', '', item)
            item = re.sub(r'[/\\\-–—]', ' ', item)
            item = re.sub(r'[^\w\s]', '', item)
            item = re.sub(r'\s+', ' ', item).strip()
            cleaned.append(item)
        return cleaned
    except:
        return []

filtered_df['Ingredients'] = filtered_df['Ingredients'].apply(clean_ingredients)
filtered_df.to_csv('/content/cleaned_ingredients.csv', index=False)

substitution_path = "/content/dataset_io.txt"
substitution_dict = {}

with open(substitution_path, 'r') as f:
    for line in f:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            substitution_dict[key.strip().lower()] = value.strip()

for idx, row in filtered_df.iterrows():
    for ingredient in row['Ingredients']:
        words = ingredient.split()
        for word in words:
            if word in substitution_dict:
                print(f"Word: {word} → Substitute: {substitution_dict[word]}")

import pandas as pd

try:
    ingredients_df = pd.read_excel('/content/output.xlsx', engine='openpyxl', header=None) 
    ingredients = []

    for cell in ingredients_df.values.flatten():
        if isinstance(cell, str):
            cell = cell.strip().lower()
            if 2 <= len(cell) <= 100 and ' ' in cell or cell.isalpha():
                ingredients.append(cell)
except Exception as e:
    print("Error reading Excel:", e)
    ingredients = []


substitutions = {}
with open('/content/dataset_io.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if ':' in line:
            key, values = line.strip().split(':', 1)
            key = key.strip().lower()
            value_list = [v.strip() for v in values.split(',') if v.strip()]
            if value_list:
                substitutions[key] = value_list

matched = {
    'Ingredient': [],
    'Substitute': []
}

for ing in ingredients:
    if ing in substitutions:
        for sub in substitutions[ing]:
            matched['Ingredient'].append(ing)
            matched['Substitute'].append(sub)

result_df = pd.DataFrame(matched)
result_df.to_csv('/content/datasetFinal.csv', index=False)
print(result_df)

import pandas as pd

df = pd.read_csv('/content/datasetFinal.csv')

df['Ingredient'] = df['Ingredient'].astype(str).str.strip().str.lower()
df['Substitute'] = df['Substitute'].astype(str).str.strip().str.lower()

df_cleaned = df.drop_duplicates(subset=['Ingredient', 'Substitute']).reset_index(drop=True)
df_cleaned = df_cleaned.sort_values(by='Ingredient').reset_index(drop=True)

df_cleaned.to_csv('/content/datasetFinal_cleaned.csv', index=False)

print(df_cleaned)
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd.read_csv('/content/datasetFinal_cleaned.csv')
df['Ingredient'] = df['Ingredient'].astype(str).str.lower().str.strip()
df['Substitute'] = df['Substitute'].astype(str).str.lower().str.strip()

train_ingredients, val_ingredients, train_subs, val_subs = train_test_split(
    df['Ingredient'], df['Substitute'], test_size=0.2, random_state=42)

train_subs = ['<start> ' + s + ' <end>' for s in train_subs]
val_subs = ['<start> ' + s + ' <end>' for s in val_subs]

input_tokenizer = Tokenizer(filters='')
output_tokenizer = Tokenizer(filters='')

input_tokenizer.fit_on_texts(train_ingredients)
output_tokenizer.fit_on_texts(train_subs)

input_seqs = input_tokenizer.texts_to_sequences(train_ingredients)
output_seqs = output_tokenizer.texts_to_sequences(train_subs)

max_input_len = max(len(seq) for seq in input_seqs)
max_output_len = max(len(seq) for seq in output_seqs)

input_seqs = pad_sequences(input_seqs, maxlen=max_input_len, padding='post')
output_seqs = pad_sequences(output_seqs, maxlen=max_output_len, padding='post')

decoder_input_data = output_seqs[:, :-1]
decoder_target_data = output_seqs[:, 1:]

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

embedding_dim = 128
lstm_units = 128

encoder_inputs = tf.keras.Input(shape=(max_input_len,))
enc_emb = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
_, state_h, state_c = tf.keras.layers.LSTM(lstm_units, return_state=True)(enc_emb)

decoder_inputs = tf.keras.Input(shape=(max_output_len - 1,))
dec_emb = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
epochs = 2000

for epoch in range(1, epochs + 1):
    print(f"\n🔁 Epoch {epoch}/{epochs}")
    hist = model.fit([input_seqs, decoder_input_data],
                     np.expand_dims(decoder_target_data, -1),
                     batch_size=32, epochs=1, validation_split=0.2, verbose=0)

    history['loss'].append(hist.history['loss'][0])
    history['accuracy'].append(hist.history['accuracy'][0])
    history['val_loss'].append(hist.history['val_loss'][0])
    history['val_accuracy'].append(hist.history['val_accuracy'][0])


    print(f"Train Loss: {hist.history['loss'][0]:.4f} | Val Loss: {hist.history['val_loss'][0]:.4f}")
    print(f"Train Acc : {hist.history['accuracy'][0]:.4f} | Val Acc : {hist.history['val_accuracy'][0]:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
