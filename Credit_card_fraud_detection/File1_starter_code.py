import pandas as pd
from imblearn.over_sampling import SMOTE
from google.colab import files
import zipfile
import os
from sklearn.preprocessing import LabelEncoder

zip_train = '/content/fraudTrain.csv.zip'
zip_test = '/content/fraudTest.csv.zip'
extracted_folder = '/content/extracted_files/'

os.makedirs(extracted_folder, exist_ok=True)

with zipfile.ZipFile(zip_train, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

with zipfile.ZipFile(zip_test, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

train_df = pd.read_csv(os.path.join(extracted_folder, 'fraudTrain.csv'))
test_df = pd.read_csv(os.path.join(extracted_folder, 'fraudTest.csv'))

train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['trans_date_trans_time'] = train_df['trans_date_trans_time'].astype(int) / 10**9

categorical_cols = train_df.select_dtypes(include=['object']).columns

def one_hot_encode_one_by_one(df, categorical_cols):
    encoded_df = df.copy()

    for col in categorical_cols:
        encoded_col = pd.get_dummies(df[col], prefix=col, drop_first=True)
        encoded_df = pd.concat([encoded_df, encoded_col], axis=1).drop(col, axis=1)

    return encoded_df

train_df_encoded = one_hot_encode_one_by_one(train_df, categorical_cols)

print("Data after One-Hot Encoding:")
print(train_df_encoded.head())

train_df_encoded.to_csv('/content/fraudTrain_one_hot_encoded.csv', index=False)

print("This code ends")
