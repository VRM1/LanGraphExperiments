from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import numpy as np

# Load the datasets (same as your original code)
data_path = 'Datasets/UCI_Credit_Card.csv'
feature_description_path = 'Datasets/credit_card_feature_descriptions.csv'

# Initialize the LLM (same as your original code)
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Define the same tools as your original code
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

# Define tools list
tools = [
    load_data,
    get_feature_descriptions,
    get_descriptive_statistics,
    get_feature_statistics,
    correlation_analysis
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# System message (same as your original)
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

# Define the agent node
def agent_node(state: State):
    """
    The main agent node that processes user input and decides whether to use tools.
    """
    # Add system message if this is the first interaction
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_message)] + messages
    
    # Get response from LLM
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: State):
    """
    Execute tools when the agent decides to use them.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Execute all tool calls in the last message
    tool_messages = []
    for tool_call in last_message.tool_calls:
        # Find the tool by name
        tool_to_call = next((t for t in tools if t.name == tool_call["name"]), None)
        if tool_to_call:
            try:
                # Execute the tool
                result = tool_to_call.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call["id"]
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_call['name']} not found",
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {"messages": tool_messages}

def should_continue(state: State):
    """
    Determine whether to continue with tool execution or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, execute tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation turn
    return END

# Create the graph
def create_graph():
    # Initialize memory
    memory = MemorySaver()
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Compile the graph with memory
    graph = workflow.compile(checkpointer=memory)
    
    return graph

# Interactive function (similar to your run_agent function)
def run_langgraph_agent(graph, query, thread_id="default_thread"):
    """Run the LangGraph agent with a user query and return the response."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Stream the graph execution
    result = None
    for event in graph.stream(
        {"messages": [HumanMessage(content=query)]}, 
        config=config,
        stream_mode="updates"
    ):
        for node_name, output in event.items():
            if node_name == "agent" and output["messages"]:
                last_message = output["messages"][-1]
                if isinstance(last_message, AIMessage) and not hasattr(last_message, 'tool_calls'):
                    result = last_message.content
                elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    result = last_message.content
    
    return result

# Main execution
if __name__ == "__main__":
    print("LangGraph Data Analysis Agent initialized. You can start asking questions about the UCI Credit Card dataset.")
    print("Type 'exit' to quit.")
    
    # Create the graph
    graph = create_graph()
    
    # Initial data loading
    print(run_langgraph_agent(graph, "Load the dataset and tell me what it contains.", "session_1"))
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        response = run_langgraph_agent(graph, user_input, "session_1")
        if response:
            print("\nAgent:", response)