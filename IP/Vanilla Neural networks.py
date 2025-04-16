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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

df = pd.read_csv('/content/filtered_ingredients_with_substitutes.csv')
train_df, val_df = train_test_split(df[['Ingredients', 'Substitute']], test_size=0.2, random_state=42)

X_train = train_df['Ingredients']
y_train = train_df['Substitute']
X_val = val_df['Ingredients']
y_val = val_df['Substitute']

all_labels = pd.concat([y_train, y_val])

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

y_train_encoded = to_categorical(y_train_encoded, num_classes=num_classes)
y_val_encoded = to_categorical(y_val_encoded, num_classes=num_classes)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_val_tokenized = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_tokenized, padding='post', maxlen=20)
X_val_pad = pad_sequences(X_val_tokenized, padding='post', maxlen=20)

model_nn = Sequential()
model_nn.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=20))
model_nn.add(GlobalAveragePooling1D())
model_nn.add(Dense(64, activation='relu'))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(num_classes, activation='softmax'))

model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_nn = model_nn.fit(X_train_pad, y_train_encoded, validation_data=(X_val_pad, y_val_encoded), epochs=10, batch_size=32)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_nn.history['accuracy'], label='Train Accuracy')
plt.plot(history_nn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Vanilla NN - Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history_nn.history['loss'], label='Train Loss')
plt.plot(history_nn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Vanilla NN - Loss')

plt.tight_layout()
plt.show()

model_nn.save('/content/vanilla_nn_model.h5')
