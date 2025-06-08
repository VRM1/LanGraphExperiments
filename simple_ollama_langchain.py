from langchain_ollama import ChatOllama
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np

# Load the datasets
data_path = 'Datasets/UCI_Credit_Card.csv'
feature_description_path = 'Datasets/credit_card_feature_descriptions.csv'

# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Load the data
@tool
def load_data():
    """
    Load the credit card dataset and return basic information about it.
    """
    try:
        df = pd.read_csv(data_path)
        return f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

@tool
def get_feature_descriptions():
    """
    Return descriptions of all features in the dataset.
    """
    try:
        feature_df = pd.read_csv(feature_description_path)
        descriptions = []
        for _, row in feature_df.iterrows():
            descriptions.append(f"{row['Feature']}: {row['Description']}")
        return "\n".join(descriptions)
    except Exception as e:
        return f"Error loading feature descriptions: {str(e)}"

@tool
def get_descriptive_statistics():
    """
    Generate descriptive statistics for the dataset.
    """
    try:
        df = pd.read_csv(data_path)
        stats = df.describe().to_string()
        return f"Descriptive Statistics:\n{stats}"
    except Exception as e:
        return f"Error generating descriptive statistics: {str(e)}"

@tool
def get_feature_statistics(feature_name: str):
    """
    Get detailed statistics for a specific feature.
    """
    try:
        df = pd.read_csv(data_path)
        if feature_name not in df.columns:
            return f"Feature '{feature_name}' not found in the dataset. Available features are: {', '.join(df.columns.tolist())}"
        
        feature_data = df[feature_name]
        
        # Calculate statistics based on data type
        if pd.api.types.is_numeric_dtype(feature_data):
            stats = {
                "mean": feature_data.mean(),
                "median": feature_data.median(),
                "std": feature_data.std(),
                "min": feature_data.min(),
                "max": feature_data.max(),
                "25%": feature_data.quantile(0.25),
                "75%": feature_data.quantile(0.75),
                "missing values": feature_data.isna().sum()
            }
            return f"Statistics for {feature_name}:\n" + "\n".join([f"{k}: {v}" for k, v in stats.items()])
        else:
            value_counts = feature_data.value_counts().to_dict()
            missing = feature_data.isna().sum()
            stats = f"Value counts for {feature_name}:\n" + "\n".join([f"{k}: {v}" for k, v in value_counts.items()])
            stats += f"\nMissing values: {missing}"
            return stats
    except Exception as e:
        return f"Error analyzing feature: {str(e)}"

@tool
def correlation_analysis():
    """
    Calculate and return correlations between numeric features.
    """
    try:
        df = pd.read_csv(data_path)
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().round(2)
        return f"Correlation Matrix:\n{corr_matrix.to_string()}"
    except Exception as e:
        return f"Error calculating correlations: {str(e)}"

# Define custom tools list
tools = [
    load_data,
    get_feature_descriptions,
    get_descriptive_statistics,
    get_feature_statistics,
    correlation_analysis
]

# Set up memory for the agent
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Define the prompt template with system message and memory
system_message = """You are a helpful data analysis assistant. Your task is to analyze the UCI Credit Card dataset.

The dataset contains information about credit card users, including their demographic information, credit data, history of payment, and bill statements.

When asked, you should use the appropriate tools to:
1. Load and provide information about the dataset
2. Explain the features in the dataset
3. Generate descriptive statistics
4. Analyze specific features
5. Calculate correlations between features

Provide clear, accurate responses based on the data. When you don't know something or if the data is not available, say so.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

# Example usage function
def run_agent(query):
    """Run the agent with a user query and return the response."""
    result = agent_executor.invoke({"input": query})
    return result["output"]

# Interactive loop for testing
if __name__ == "__main__":
    print("Data Analysis Agent initialized. You can start asking questions about the UCI Credit Card dataset.")
    print("Type 'exit' to quit.")
    
    # Initial data loading
    print(run_agent("Load the dataset and tell me what it contains."))
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        response = run_agent(user_input)
        print("\nAgent:", response)