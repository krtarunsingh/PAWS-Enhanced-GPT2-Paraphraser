
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model and tokenizer
model_name = "./gpt2_fine_tuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate paraphrases
def paraphrase(text, max_length=50, temperature=1.0):
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)

    # Decode the output to text
    paraphrased_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return paraphrased_text

# Usage example
input_file_path = "sample_data/sample_input.txt"
output_file_path = "sample_data/sample_output.txt"

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        paraphrased_text = paraphrase(line.strip())
        outfile.write(paraphrased_text + '\n')
