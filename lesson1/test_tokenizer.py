# Warning control

from transformers import AutoTokenizer

# A list of colors in RGB for representing the tokens
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer_name: str):
    """ Show the tokens each separated by a different color """

    # Load the tokenizer and tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length
    print(f"\n\nVocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )

def main():
    # Initialize the tokenizer
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # Example text to tokenize
    sentence = "Hello world!"
    
    # apply the tokenizer to the sentence and extract the token ids
    token_ids = tokenizer(sentence).input_ids
    print(token_ids)
    
    for id in token_ids:
        print(tokenizer.decode(id))

    text = """
        English and CAPITALIZATION
        🎵 鸟
        show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
        12.0*50=600
        """
    show_tokens(text, "bert-base-cased")
    show_tokens(text, "Xenova/gpt-4")

if __name__ == "__main__":
    main()
