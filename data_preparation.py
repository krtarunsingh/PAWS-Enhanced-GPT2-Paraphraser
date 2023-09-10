
import os
import requests
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Download the PAWS dataset
def download_dataset(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# Step 2: Extract the downloaded files
def extract_files(zip_path, extract_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Step 3: Preprocess the data to create training and validation sets
def preprocess_data(data_path, train_path, valid_path):
    # Load the dataset
    data = pd.read_csv(data_path, sep='\t')
    
    # Create a dataset with pairs of sentences
    paraphrase_pairs = data[['sentence1', 'sentence2']].dropna()
    
    # Split the data into training and validation sets
    train_data, valid_data = train_test_split(paraphrase_pairs, test_size=0.2, random_state=42)
    
    # Save the training and validation data to files
    train_data.to_csv(train_path, sep='\t', index=False)
    valid_data.to_csv(valid_path, sep='\t', index=False)

def main():
    # Define the URLs and paths
    dataset_url = 'https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz'
    download_path = 'paws_wiki_labeled_final.tar.gz'
    extract_path = 'paws_dataset'
    data_file_path = os.path.join(extract_path, 'final', 'wiki_00')
    train_file_path = 'train_data.tsv'
    valid_file_path = 'valid_data.tsv'
    
    # Download, extract and preprocess the dataset
    download_dataset(dataset_url, download_path)
    extract_files(download_path, extract_path)
    preprocess_data(data_file_path, train_file_path, valid_file_path)

if __name__ == "__main__":
    main()
