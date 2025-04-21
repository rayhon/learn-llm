# Learning Machine Learning By Examples

## 0. Setup

1. Make sure you have Python 3.8+ installed
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

This repository contains examples of different machine learning concepts.

## 1. Neural Network with Iris Dataset

The `1_nn_iris.py` demonstrates a basic neural network implementation using PyTorch to classify iris flowers. This implementation follows the tutorial series from [Python Neural Networks with PyTorch](https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1).

### Running the Iris Example:
```bash
python 1_nn_iris.py
```

The script will:
1. Load and prepare the iris dataset
2. Train a neural network model
3. Display training progress and loss plot
4. Show detailed evaluation metrics
5. Demonstrate model saving and loading
6. Verify loaded model performance

## 2. Tokenization Example

This directory also contains a simple example of text tokenization using the Hugging Face Transformers library.

### Running the Tokenization Example:
```bash
python test_tokenizer.py
```


The script will:
1. Load the GPT-2 tokenizer
2. Tokenize a sample text
3. Show the resulting tokens and their IDs
4. Demonstrate decoding tokens back to text

### What is Tokenization?

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, subwords, or characters, depending on the tokenizer used. In this example, we're using the GPT-2 tokenizer which uses a byte-pair encoding (BPE) approach to create subword tokens. 

### Model Types

The transformers library has two types of model classes:  AutoModelForCausalLM  and AutoModelForMaskedLM. 
* Causal language models represent the decoder-only models that are used for text generation. They are described as causal, because to predict the next token, the model can only attend to the preceding left tokens. 
* Masked language models represent the encoder-only models that are used for rich text representation. They are described as masked, because they are trained to predict a masked or hidden token in a sequence.