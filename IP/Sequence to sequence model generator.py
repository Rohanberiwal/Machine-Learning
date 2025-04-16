import zipfile
import os
import os
import os
os.environ["WANDB_MODE"] = "disabled" 
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
            item = re.sub(r'[\d½⅓¼¾⅔⅛⅜⅝⅞]+', '', item)
            item = re.sub(r'[-–—]+', ' ', item)
            item = re.sub(r'\s+', ' ', item).strip()
            cleaned.append(item)
        return cleaned
    except:
        return []

filtered_df['Ingredients'] = filtered_df['Ingredients'].apply(clean_ingredients)
import pandas as pd

csv_path = '/content/unique_ingredients_from_set.csv'
df = pd.read_csv(csv_path)

print(df)
print(f"\nTotal Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Column Names: {list(df.columns)}")
import pandas as pd

csv_path = '/content/unique_ingredients_from_set.csv'
df = pd.read_csv(csv_path)

substitutions = {}
with open('/content/dataset_ionew.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if ':' in line:
            key, values = line.strip().split(':', 1)
            key = key.strip().lower()
            value_list = [v.strip().lower() for v in values.split(',') if v.strip()]
            if value_list:
                substitutions[key] = value_list

def get_substitutes(ingredient):
    ing = str(ingredient).strip().lower()
    candidates = [ing]
    if ing.endswith('es'):
        candidates.append(ing[:-2])
    elif ing.endswith('s'):
        candidates.append(ing[:-1])
    for candidate in candidates:
        if candidate in substitutions:
            return ', '.join(substitutions[candidate])
    return ''

df['Substitute'] = df['Ingredients'].apply(get_substitutes)
found_count = df['Substitute'].apply(lambda x: x != '').sum()

output_path = '/content/ingredients_with_substitutes.csv'
df.to_csv(output_path, index=False)

print(df)
print(f"\nTotal Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Column Names: {list(df.columns)}")
print(f"\nSubstitutions Found For: {found_count} ingredients")
import pandas as pd

df = pd.read_csv('/content/ingredients_with_substitutes.csv')
df = df[df['Substitute'].notna() & df['Substitute'].str.strip().ne('')]
df.to_csv('/content/filtered_ingredients_with_substitutes.csv', index=False)

print(df)
print(f"\nFiltered Rows (with substitutions): {len(df)}")
print(f"Column Names: {list(df.columns)}")

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/filtered_ingredients_with_substitutes.csv')
train_df, val_df = train_test_split(df[['Ingredients', 'Substitute']], test_size=0.2, random_state=42)

X_train = train_df['Ingredients']
y_train = train_df['Substitute']
X_val = val_df['Ingredients']
y_val = val_df['Substitute']


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
input_sequences = tokenizer.texts_to_sequences(X_train)
target_sequences = tokenizer.texts_to_sequences(y_train)

max_input_len = max([len(seq) for seq in input_sequences])

max_target_len = 6  
X_train_pad = pad_sequences(input_sequences, padding='post', maxlen=max_input_len)
y_train_pad = pad_sequences(target_sequences, padding='post', maxlen=max_target_len)

X_val_sequences = tokenizer.texts_to_sequences(X_val)
y_val_sequences = tokenizer.texts_to_sequences(y_val)

X_val_pad = pad_sequences(X_val_sequences, padding='post', maxlen=max_input_len)
y_val_pad = pad_sequences(y_val_sequences, padding='post', maxlen=max_target_len)

model_seq2seq = Sequential()
model_seq2seq.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_input_len))
model_seq2seq.add(GRU(128, return_sequences=True))
model_seq2seq.add(Dropout(0.5))
model_seq2seq.add(TimeDistributed(Dense(len(tokenizer.word_index) + 1, activation='softmax')))

model_seq2seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


y_train_onehot = np.zeros((y_train_pad.shape[0], y_train_pad.shape[1], len(tokenizer.word_index) + 1), dtype='float32')
y_val_onehot = np.zeros((y_val_pad.shape[0], y_val_pad.shape[1], len(tokenizer.word_index) + 1), dtype='float32')

for i, seq in enumerate(y_train_pad):
    for t, word_idx in enumerate(seq):
        if word_idx > 0:
            y_train_onehot[i, t, word_idx] = 1.0

for i, seq in enumerate(y_val_pad):
    for t, word_idx in enumerate(seq):
        if word_idx > 0:
            y_val_onehot[i, t, word_idx] = 1.0

history_seq2seq = model_seq2seq.fit(X_train_pad, y_train_onehot, validation_data=(X_val_pad, y_val_onehot), epochs=1000, batch_size=32)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_seq2seq.history['accuracy'], label='Train Accuracy')
plt.plot(history_seq2seq.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Seq2Seq - Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history_seq2seq.history['loss'], label='Train Loss')
plt.plot(history_seq2seq.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Seq2Seq - Loss')

plt.tight_layout()
plt.show()


model_seq2seq.save('/content/seq2seq_gru_model.h5')


def predict_substitute(ingredient):
    sequence = tokenizer.texts_to_sequences([ingredient])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=max_input_len)
    pred = model_seq2seq.predict(padded_sequence)
    pred_sequence = pred.argmax(axis=-1)[0]
    predicted_substitute = ' '.join([tokenizer.index_word.get(idx, '') for idx in pred_sequence if idx > 0])
    return predicted_substitute


ingredient_example = "almond milk"
predicted_substitute = predict_substitute(ingredient_example)
print(f"Substitute for '{ingredient_example}': {predicted_substitute}")
