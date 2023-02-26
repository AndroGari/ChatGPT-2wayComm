import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained GPT-2 model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Start the conversation
print("Bot: Hi, I'm a chatbot. What's your name?")
user_input = input("You: ")

while True:
    # Tokenize the user input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate a response from the model
    output = model.generate(input_ids=input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response and print it
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Bot:", bot_response)

    # Ask the user for their next input
    user_input = input("You: ")