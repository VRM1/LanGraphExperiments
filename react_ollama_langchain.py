from langchain_ollama import ChatOllama
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# Tool functions with proper argument handling
def load_data(dummy_input=""):
    """
    Load the credit card dataset and return basic information about it.
    The dummy_input parameter is ignored but required for LangChain compatibility.
    """
    try:
        df = pd.read_csv(data_path)
        return f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

def get_feature_descriptions(dummy_input=""):
    """
    Return descriptions of all features in the dataset.
    The dummy_input parameter is ignored but required for LangChain compatibility.
    """
    try:
        feature_df = pd.read_csv(feature_description_path)
        descriptions = []
        for _, row in feature_df.iterrows():
            descriptions.append(f"{row['Feature']}: {row['Description']}")
        return "\n".join(descriptions)
    except Exception as e:
        return f"Error loading feature descriptions: {str(e)}"

def get_descriptive_statistics(dummy_input=""):
    """
    Generate descriptive statistics for the dataset.
    The dummy_input parameter is ignored but required for LangChain compatibility.
    """
    try:
        df = pd.read_csv(data_path)
        stats = df.describe().to_string()
        return f"Descriptive Statistics:\n{stats}"
    except Exception as e:
        return f"Error generating descriptive statistics: {str(e)}"

def get_feature_statistics(feature_name):
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

def correlation_analysis(dummy_input=""):
    """
    Calculate and return correlations between numeric features.
    The dummy_input parameter is ignored but required for LangChain compatibility.
    """
    try:
        df = pd.read_csv(data_path)
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().round(2)
        return f"Correlation Matrix:\n{corr_matrix.to_string()}"
    except Exception as e:
        return f"Error calculating correlations: {str(e)}"

# Define tools in LangChain format
tools = [
    Tool(
        name="LoadData",
        func=load_data,
        description="Load the credit card dataset and get basic information about it. Use this when you need to know what data is available."
    ),
    Tool(
        name="GetFeatureDescriptions",
        func=get_feature_descriptions,
        description="Get descriptions of all features in the dataset. Use this when you need to understand what each column means."
    ),
    Tool(
        name="GetDescriptiveStatistics",
        func=get_descriptive_statistics,
        description="Generate descriptive statistics (mean, std, min, max, etc.) for all numeric columns in the dataset. Use this for an overall statistical summary."
    ),
    Tool(
        name="GetFeatureStatistics",
        func=get_feature_statistics,
        description="Get detailed statistics for a specific feature. Input should be a column name from the dataset."
    ),
    Tool(
        name="CorrelationAnalysis",
        func=correlation_analysis,
        description="Calculate and return correlations between numeric features. Use this to understand relationships between variables."
    )
]

# Set up memory
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Define the ReAct prompt template correctly
react_template = """You are a data analysis assistant that analyzes the UCI Credit Card dataset.

The dataset contains information about credit card users, including their demographic information, credit data, history of payment, and bill statements.

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!
Previous conversation history:
{chat_history}
New input: {input}
{agent_scratchpad}
"""

# Use from_template with just the needed variables
prompt = ChatPromptTemplate.from_template(react_template)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    max_iterations=5,  # Limit iterations to prevent excessive tool use
)

# Example usage function
def run_agent(query):
    """Run the agent with a user query and return the response."""
    result = agent_executor.invoke({"input": query})
    return result["output"]

# Interactive loop for testing
if __name__ == "__main__":
    print("ReAct Data Analysis Agent initialized. You can start asking questions about the UCI Credit Card dataset.")
    print("Type 'exit' to quit.")
    
    # Initial data loading
    print(run_agent("Load the dataset and tell me what it contains."))
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        response = run_agent(user_input)
        print("\nAgent:", response)