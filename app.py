import gradio as gr
from mysql.connector import connect, Error
from predict_sql import load_model, predict_sql

# Load the trained model
tokenizer, model = load_model()

# Function to handle user input
def gradio_nl_to_sql(host, user, password, database, question):
    try:
        # Connect to MySQL
        connection = connect(
            host=host,
            user=user,
            password=password,
            database=database,
        )
        cursor = connection.cursor()

        # Generate SQL Query
        sql_query = predict_sql(question, tokenizer, model)

        # Execute SQL Query
        cursor.execute(sql_query)
        results = cursor.fetchall()

        # Clean up and close the connection
        cursor.close()
        connection.close()

        return f"SQL Query: {sql_query}\n\nResults: {results}"
    except Error as e:
        return f"Error: {e}"

# Gradio interface
interface = gr.Interface(
    fn=gradio_nl_to_sql,
    inputs=[
        gr.Textbox(label="Host", placeholder="localhost"),
        gr.Textbox(label="User"),
        gr.Textbox(label="Password", type="password"),
        gr.Textbox(label="Database"),
        gr.Textbox(label="Ask your question"),
    ],
    outputs="text",
    title="NL-to-SQL Query Execution",
    description="Ask questions in plain English and get results directly from the database.",
)

if __name__ == "__main__":
    interface.launch()
