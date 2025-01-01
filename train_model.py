import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the dataset
class NLSQLDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Adjusting for column names 'text_query' and 'sql_command'
        question = self.data.iloc[index]["text_query"]  # Changed column name to 'text_query'
        sql_query = self.data.iloc[index]["sql_command"]  # Changed column name to 'sql_command'
        
        # Tokenize the input question (English)
        inputs = self.tokenizer(
            "translate English to SQL: " + question,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize the output SQL query
        targets = self.tokenizer(
            sql_query, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }

# Training function
def train_nl_to_sql(train_file, model_save_path="model/t5_nl2sql", epochs=5, batch_size=8, lr=5e-5):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = NLSQLDataset(train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Change to the correct CSV file path, now using 'spider_text_sql.csv' instead of 'train.csv'
    train_nl_to_sql("spider_text_sql.csv")
