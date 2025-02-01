import zipfile
import os
import pandas as pd
import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import re


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

"""
plt.figure(figsize=(12, 8))
top_ingredients.head(20).plot(kind='bar', color='skyblue')
plt.title('Top 20 Most Frequent Ingredients')
plt.xlabel('Ingredients')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
top_ingredients.head(5000).plot(kind='bar', color='skyblue')
plt.title('Top 5,000 Most Frequent Ingredients')
plt.xlabel('Ingredients')
plt.ylabel('Frequency')
plt.xticks(rotation=90, ha='right') 
plt.tight_layout()
plt.show()

import requests
import pandas as pd

api_key = "6d688b58c9af41b988f2f6b86c7c1a82"

def get_substitutions(ingredient, api_key):
    url = f"https://api.spoonacular.com/food/ingredients/substitutes?ingredientName={ingredient}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "substitutes" in data:
            return data["substitutes"]
    return []

top_100_ingredients = top_ingredients.head(100).index.tolist()

substitution_data = []
for ingredient in top_100_ingredients:
    substitutions = get_substitutions(ingredient, api_key)
    substitution_data.append({
        "Ingredient": ingredient,
        "Substitutions": ", ".join(substitutions) if substitutions else "No substitutions found"
    })

substitution_df = pd.DataFrame(substitution_data)

substitution_df.to_csv("ingredient_substitutions.csv", index=False)

print(substitution_df)


"""

from gensim.models import Word2Vec
tokenized_ingredients = [ingredient.split() for ingredient in cleaned_ingredients]

model = Word2Vec(sentences=tokenized_ingredients, vector_size=100, window=5, min_count=1, workers=4)

model.save("ingredient_word2vec.model")

def find_substitutes(ingredient, model, topn=5):
    try:
        substitutes = model.wv.most_similar(ingredient, topn=topn)
        return [sub[0] for sub in substitutes]
    except KeyError:
        return [] 

substitution_dict = {}
for ingredient in top_ingredients.index[:5000]:  
    substitutes = find_substitutes(ingredient, model)
    if substitutes:
        substitution_dict[ingredient] = substitutes


for ingredient, substitutes in substitution_dict.items():
    print(f"{ingredient}: {', '.join(substitutes)}")

with open("subsWordVec.txt", "w") as file:
    for ingredient, substitutes in substitution_dict.items():
        file.write(f"{ingredient}: {', '.join(substitutes)}\n")

print("Substitutions saved to 'subs.txt'.")

with open("subsWordVec.txt", "r") as file:
    num_lines = len(file.readlines())

print(f"Number of lines in 'subs.txt': {num_lines}")
print("Runnign the golbve model now ")

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = "glove.6B.100d.txt"  
word2vec_output_file = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

def find_substitutes(ingredient, model, topn=5):
    try:
        substitutes = model.most_similar(ingredient, topn=topn)
        return [sub[0] for sub in substitutes]
    except KeyError:
        return []


substitution_dict = {}
for ingredient in top_ingredients.index[:5000]:  
    substitutes = find_substitutes(ingredient, glove_model)
    if substitutes:
        substitution_dict[ingredient] = substitutes

with open("subus.txt", "w") as file:
    for ingredient, substitutes in substitution_dict.items():
        file.write(f"{ingredient}: {', '.join(substitutes)}\n")

print("Substitutions saved to 'subus.txt'.")

with open("subus.txt", "r") as file:
    num_lines = len(file.readlines())

print(f"Number of lines in 'subus.txt': {num_lines}")

with open("glove_embedding_substitution.txt", "w") as glove_file:
    for ingredient, substitutes in substitution_dict.items():
        
        if ingredient in glove_model:
            glove_file.write(f"Ingredient: {ingredient}\n")
            glove_file.write(f"Embedding: {glove_model[ingredient]}\n")
        
        
        for sub in substitutes:
            if sub in glove_model:
                glove_file.write(f"Substitute: {sub}\n")
                glove_file.write(f"Embedding: {glove_model[sub]}\n")
        
        glove_file.write("\n") 

print("GloVe embeddings for substitutions saved to 'glove_embedding_substitution.txt'.")



print("This is the next approach for the word vec and the stuff ")
import pandas as pd

csv_file_path = "/content/extracted_files/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"  
df = pd.read_csv(csv_file_path)

print(df.head())

cleaned_ingredients = df["Cleaned_Ingredients"].apply(eval)  
print(cleaned_ingredients[:5])
from gensim.models import Word2Vec

# Train Word2Vec model
model = Word2Vec(
    sentences=cleaned_ingredients,  # List of ingredient lists
    vector_size=100,  # Size of the embedding vector
    window=5,  # Number of words to consider as context
    min_count=1,  # Ignore words with frequency less than this
    workers=4  # Number of CPU cores to use
)

# Save the trained model
model.save("ingredient2vec.model")

# Function to find substitutes
def find_substitutes(ingredient, model, topn=5):
    try:
        substitutes = model.wv.most_similar(ingredient, topn=topn)
        return [sub[0] for sub in substitutes]
    except KeyError:
        return []

# Example usage
ingredient = "tomato"
substitutes = find_substitutes(ingredient, model)
print(f"Substitutes for {ingredient}: {substitutes}")
