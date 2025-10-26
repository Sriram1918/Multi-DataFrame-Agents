🧠 Multi DataFrame Agent using LangChain + OpenAI

This project demonstrates how to use LangChain and OpenAI’s LLMs to interact intelligently with multiple Pandas DataFrames.
Instead of writing manual queries, you can ask questions in natural language, and the agent will automatically analyze, combine, and summarize data from multiple sources.

🚀 Features

🧩 Query multiple DataFrames simultaneously.

💬 Ask natural language questions about your datasets.

⚙️ Uses LangChain’s create_pandas_dataframe_agent for reasoning and analysis.

🤖 Powered by OpenAI GPT models (gpt-4o-mini or similar).

📊 Automatically executes safe Pandas code to answer questions.

🧠 Extendable with memory, callbacks, or custom tools.

📂 Project Structure
📦 Multi_dataframe_Agents
 ┣ 📜 Multi_dataframe_Agents.ipynb   ← Main notebook
 ┣ 📊 sales.csv                      ← Example dataset (Sales)
 ┣ 📊 customers.csv                  ← Example dataset (Customers)
 ┗ 📘 README.md                      ← Documentation (this file)

🧰 Requirements

Install dependencies before running the notebook:

pip install langchain langchain-openai pandas python-dotenv


You’ll also need an OpenAI API key:

export OPENAI_API_KEY="your_api_key_here"


Or create a .env file with:

OPENAI_API_KEY=your_api_key_here

🧩 How It Works

Import Dependencies

from langchain.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd


Load Multiple DataFrames

df_sales = pd.read_csv("sales.csv")
df_customers = pd.read_csv("customers.csv")


Initialize the LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


Create the Agent

agent = create_pandas_dataframe_agent(
    llm,
    [df_sales, df_customers],
    verbose=True
)


Ask Questions in Natural Language

agent.run("Which customer had the highest sales in 2024?")
agent.run("Show top 3 customers by total revenue.")

🧠 How It Works (Under the Hood)

LangChain gives the LLM access to tools like Pandas DataFrames.

The LLM:

Reads metadata about your DataFrames (columns, sample rows).

Writes Python code to answer your query.

Executes it safely and returns the result.

This is an example of a Tool-Using Agent in LangChain.

🪄 Key Concepts to Remember
Concept	Description
LLM (Large Language Model)	Understands and generates human-like text.
LangChain	Framework that lets LLMs use external tools, APIs, and data.
Pandas DataFrame Agent	Allows the LLM to query structured data.
Tool-Using Agent	The model uses Pandas as a tool to execute code.
RAG (Retrieval-Augmented Generation)	Not used here, but similar concept for text-based data.
Memory	Can store past conversation context (optional).
📈 Example Questions You Can Ask

“Which customer spent the most money in 2024?”

“Find total sales by region and sort descending.”

“Show customers who purchased both Product A and Product B.”

“Compare 2023 vs 2024 total sales growth.”

🔧 Extensions

You can extend this notebook by:

Adding more DataFrames (finance, marketing, etc.)

Integrating Conversation Memory for chat-style use:

from langchain.memory import ConversationBufferMemory


Tracking and debugging with LangSmith callbacks.

Deploying it as a Streamlit app for interactive querying.
