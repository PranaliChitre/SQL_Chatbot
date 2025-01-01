from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path="sergears/sql-translator"):
    """Load the SQL Translator model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model


def predict_sql(question, tokenizer, model, max_len=128):
    """Predict SQL query from the input question"""
    # Tokenize the question with the necessary input format for the model
    inputs = tokenizer(
        question,
        return_tensors="pt",
        max_length=max_len,
        padding="max_length",
        truncation=True,
    )
    
    # Generate the SQL query using the model
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
    )

    # Decode the generated tokens into a human-readable SQL query
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the final SQL query
    return decoded_output


if __name__ == "__main__":
    # Load the model and tokenizer
    tokenizer, model = load_model("sergears/sql-translator")

    # Take the user's question as input
    question = input("Enter your question: ")

    # Predict the SQL query based on the input question
    sql_query = predict_sql(question, tokenizer, model)

    # Display the generated SQL query
    print("Generated SQL Query:", sql_query)
