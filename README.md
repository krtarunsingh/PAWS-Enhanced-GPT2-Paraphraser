
# GPT-2 Paraphraser with PAWS Dataset

This repository contains scripts and instructions to fine-tune the GPT-2 model for paraphrasing tasks using the PAWS dataset and to use the fine-tuned model to paraphrase texts.

## Step-by-Step Instructions

### Step 1: Setup the Environment

Before running the scripts, ensure that you have set up a Python environment with the necessary libraries. Install the required packages by running the following commands:

```bash
pip install transformers
pip install torch
pip install pandas
pip install scikit-learn
```

### Step 2: Data Preparation

Run the `data_preparation.py` script to download and prepare the PAWS dataset for training and validation. This script performs the following tasks:

1. Downloads the PAWS dataset from the official URL.
2. Extracts the downloaded files.
3. Preprocesses the data to create training and validation datasets with pairs of sentences (original and paraphrase).

Run the script with the following command:

```bash
python scripts/data_preparation.py
```

### Step 3: Fine-Tuning the GPT-2 Model

After preparing the data, run the `fine_tuning.py` script to fine-tune the GPT-2 model on the prepared dataset. This script performs the following tasks:

1. Loads the pre-trained GPT-2 model and tokenizer.
2. Prepares the training and validation datasets.
3. Sets up the training arguments and fine-tunes the model.

Run the script with the following command:

```bash
python scripts/fine_tuning.py
```

### Step 4: Using the Fine-Tuned Model for Paraphrasing

Once the model is fine-tuned, use the `paraphrase.py` script to paraphrase texts. This script performs the following tasks:

1. Loads the fine-tuned model and tokenizer.
2. Reads input texts from a file and paraphrases them.
3. Saves the paraphrased texts to another file.

You can modify the `sample_input.txt` file in the `sample_data` directory with the texts you want to paraphrase. The paraphrased texts will be saved to `sample_output.txt` in the same directory.

Run the script with the following command:

```bash
python scripts/paraphrase.py
```

## Scripts

- `data_preparation.py`: Script for downloading and preparing the PAWS dataset.
- `fine_tuning.py`: Script for fine-tuning the GPT-2 model on the prepared dataset.
- `paraphrase.py`: Script for using the fine-tuned model to paraphrase texts.

## Sample Data

- `sample_input.txt`: Sample input file with texts to be paraphrased.
- `sample_output.txt`: Sample output file with paraphrased texts.

Thank you for using the GPT-2 Paraphraser with PAWS dataset. Enjoy paraphrasing!
