# import the required classes
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("../models/microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "../models/microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # False means to not include the prompt text in the returned text
    max_new_tokens=50, 
    do_sample=False, # no randomness in the generated text
)

prompt = "The capital of France is"
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Get the output of the model before the lm_head
model_output = model.model(input_ids)
# Get the shape the output the model before the lm_head
model_output[0].shape
# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])
lm_head_output.shape

token_id = lm_head_output[0,-1].argmax(-1)
tokenizer.decode(token_id)
