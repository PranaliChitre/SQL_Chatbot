from transformers import T5Tokenizer, T5ForConditionalGeneration

# Replace 't5-small' with the model you prefer (e.g., t5-base, t5-large)
model_name = "t5-small"

# Load pretrained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Save the tokenizer and model to a local folder
save_dir = "model/t5_nl2sql"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Model and tokenizer have been saved to: {save_dir}")
