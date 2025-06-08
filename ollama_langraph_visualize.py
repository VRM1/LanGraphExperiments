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
import json
from datetime import datetime
import time

# Load the datasets
data_path = 'Datasets/UCI_Credit_Card.csv'
feature_description_path = 'Datasets/credit_card_feature_descriptions.csv'

# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Define tools with performance monitoring
@tool
def load_data():
    """Load the credit card dataset and return basic information about it."""
    start_time = time.time()
    try:
        df = pd.read_csv(data_path)
        result = f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}"
        execution_time = time.time() - start_time
        print(f"⏱️  load_data executed in {execution_time:.3f} seconds")
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ load_data failed in {execution_time:.3f} seconds: {e}")
        return f"Error loading dataset: {str(e)}"

@tool
def get_feature_descriptions():
    """Return descriptions of all features in the dataset."""
    start_time = time.time()
    try:
        feature_df = pd.read_csv(feature_description_path)
        descriptions = []
        for _, row in feature_df.iterrows():
            descriptions.append(f"{row['Feature']}: {row['Description']}")
        result = "\n".join(descriptions)
        execution_time = time.time() - start_time
        print(f"⏱️  get_feature_descriptions executed in {execution_time:.3f} seconds")
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ get_feature_descriptions failed in {execution_time:.3f} seconds: {e}")
        return f"Error loading feature descriptions: {str(e)}"

@tool
def get_descriptive_statistics():
    """Generate descriptive statistics for the dataset."""
    start_time = time.time()
    try:
        df = pd.read_csv(data_path)
        stats = df.describe().to_string()
        result = f"Descriptive Statistics:\n{stats}"
        execution_time = time.time() - start_time
        print(f"⏱️  get_descriptive_statistics executed in {execution_time:.3f} seconds")
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ get_descriptive_statistics failed in {execution_time:.3f} seconds: {e}")
        return f"Error generating descriptive statistics: {str(e)}"

@tool
def get_feature_statistics(feature_name: str):
    """Get detailed statistics for a specific feature."""
    start_time = time.time()
    try:
        df = pd.read_csv(data_path)
        if feature_name not in df.columns:
            execution_time = time.time() - start_time
            print(f"⚠️  get_feature_statistics - feature not found in {execution_time:.3f} seconds")
            return f"Feature '{feature_name}' not found in the dataset. Available features are: {', '.join(df.columns.tolist())}"
        
        feature_data = df[feature_name]
        
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
            result = f"Statistics for {feature_name}:\n" + "\n".join([f"{k}: {v}" for k, v in stats.items()])
        else:
            value_counts = feature_data.value_counts().to_dict()
            missing = feature_data.isna().sum()
            result = f"Value counts for {feature_name}:\n" + "\n".join([f"{k}: {v}" for k, v in value_counts.items()])
            result += f"\nMissing values: {missing}"
        
        execution_time = time.time() - start_time
        print(f"⏱️  get_feature_statistics executed in {execution_time:.3f} seconds")
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ get_feature_statistics failed in {execution_time:.3f} seconds: {e}")
        return f"Error analyzing feature: {str(e)}"

@tool
def correlation_analysis():
    """Calculate and return correlations between numeric features."""
    start_time = time.time()
    try:
        df = pd.read_csv(data_path)
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().round(2)
        result = f"Correlation Matrix:\n{corr_matrix.to_string()}"
        execution_time = time.time() - start_time
        print(f"⏱️  correlation_analysis executed in {execution_time:.3f} seconds")
        return result
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ correlation_analysis failed in {execution_time:.3f} seconds: {e}")
        return f"Error calculating correlations: {str(e)}"

# Tools list
tools = [load_data, get_feature_descriptions, get_descriptive_statistics, get_feature_statistics, correlation_analysis]
llm_with_tools = llm.bind_tools(tools)

# Enhanced State with comprehensive debugging
class DataAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    # Debugging fields
    current_step: str
    step_count: int
    tools_called: list
    execution_trace: list
    performance_metrics: dict
    user_intent: str
    error_count: int

# System message
system_message = """You are a helpful data analysis assistant for the UCI Credit Card dataset.

Available tools:
- load_data: Get basic dataset information
- get_feature_descriptions: Get feature descriptions
- get_descriptive_statistics: Get statistical summary
- get_feature_statistics: Get statistics for specific features
- correlation_analysis: Get correlation matrix

Use the appropriate tools to help users analyze the data. Be thorough and helpful."""

def agent_node(state: DataAnalysisState):
    """Enhanced agent node with comprehensive debugging."""
    step_start = time.time()
    step_num = state.get('step_count', 0) + 1
    
    print(f"\n{'='*60}")
    print(f"🤖 AGENT NODE - Step {step_num}")
    print(f"{'='*60}")
    print(f"📍 Current step: {state.get('current_step', 'Starting')}")
    print(f"🕐 Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    # Analyze user intent from latest message
    messages = state["messages"]
    last_human_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    user_intent = "unknown"
    if last_human_msg:
        content = last_human_msg.content.lower()
        if "load" in content or "dataset" in content:
            user_intent = "data_loading"
        elif "feature" in content and "description" in content:
            user_intent = "feature_info"
        elif "statistic" in content or "summary" in content:
            user_intent = "statistics"
        elif "correlation" in content:
            user_intent = "correlation"
        else:
            user_intent = "general_query"
        
        print(f"💭 User intent detected: {user_intent}")
        print(f"💬 Processing: '{last_human_msg.content[:80]}...'")
    
    # Add system message if needed
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_message)] + messages
    
    # Call LLM
    print("🧠 Calling LLM...")
    llm_start = time.time()
    response = llm_with_tools.invoke(messages)
    llm_time = time.time() - llm_start
    print(f"🧠 LLM responded in {llm_time:.3f} seconds")
    
    # Analyze response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        print(f"🔧 Tools planned: {', '.join(tool_names)}")
        
        # Show tool arguments for debugging
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"   Tool {i}: {tc['name']}")
            if tc.get('args'):
                print(f"      Args: {tc['args']}")
    else:
        print("💭 Direct response (no tools needed)")
    
    step_time = time.time() - step_start
    
    # Update state with debugging info
    updated_state = {
        "messages": [response],
        "current_step": "agent_completed",
        "step_count": step_num,
        "tools_called": state.get("tools_called", []),
        "user_intent": user_intent,
        "error_count": state.get("error_count", 0),
        "execution_trace": state.get("execution_trace", []) + [{
            "step": step_num,
            "node": "agent",
            "timestamp": datetime.now().isoformat(),
            "duration": step_time,
            "llm_duration": llm_time,
            "user_intent": user_intent,
            "tool_calls": tool_names if hasattr(response, 'tool_calls') and response.tool_calls else [],
            "message_type": type(response).__name__
        }],
        "performance_metrics": {
            **state.get("performance_metrics", {}),
            f"agent_step_{step_num}": step_time,
            f"llm_step_{step_num}": llm_time
        }
    }
    
    print(f"⏱️  Agent step completed in {step_time:.3f} seconds")
    return updated_state

def tool_node(state: DataAnalysisState):
    """Enhanced tool execution with detailed monitoring."""
    step_start = time.time()
    step_num = state.get('step_count', 0) + 1
    
    print(f"\n{'='*60}")
    print(f"🔧 TOOL NODE - Step {step_num}")
    print(f"{'='*60}")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_messages = []
    tools_executed = []
    tool_metrics = {}
    error_count = state.get("error_count", 0)
    
    for i, tool_call in enumerate(last_message.tool_calls, 1):
        tool_name = tool_call["name"]
        print(f"\n🛠️  Executing Tool {i}/{len(last_message.tool_calls)}: {tool_name}")
        print(f"📋 Arguments: {json.dumps(tool_call['args'], indent=2)}")
        
        # Find and execute tool
        tool_to_call = next((t for t in tools if t.name == tool_name), None)
        if tool_to_call:
            tool_start = time.time()
            try:
                result = tool_to_call.invoke(tool_call["args"])
                tool_time = time.time() - tool_start
                
                print(f"✅ Success in {tool_time:.3f}s")
                print(f"📊 Result length: {len(str(result))} characters")
                print(f"📝 Preview: {str(result)[:100]}...")
                
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                tools_executed.append(tool_name)
                tool_metrics[tool_name] = tool_time
                
            except Exception as e:
                tool_time = time.time() - tool_start
                error_count += 1
                
                print(f"❌ Failed in {tool_time:.3f}s: {str(e)}")
                tool_messages.append(
                    ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"])
                )
                tool_metrics[f"{tool_name}_error"] = tool_time
        else:
            print(f"❌ Tool '{tool_name}' not found")
            error_count += 1
            tool_messages.append(
                ToolMessage(content=f"Tool {tool_name} not found", tool_call_id=tool_call["id"])
            )
    
    step_time = time.time() - step_start
    
    # Update state
    updated_state = {
        "messages": tool_messages,
        "current_step": "tools_completed",
        "step_count": step_num,
        "tools_called": state.get("tools_called", []) + tools_executed,
        "user_intent": state.get("user_intent", "unknown"),
        "error_count": error_count,
        "execution_trace": state.get("execution_trace", []) + [{
            "step": step_num,
            "node": "tools",
            "timestamp": datetime.now().isoformat(),
            "duration": step_time,
            "tools_executed": tools_executed,
            "tools_count": len(tools_executed),
            "errors": error_count - state.get("error_count", 0)
        }],
        "performance_metrics": {
            **state.get("performance_metrics", {}),
            f"tools_step_{step_num}": step_time,
            **{f"tool_{k}_step_{step_num}": v for k, v in tool_metrics.items()}
        }
    }
    
    print(f"\n🏁 Tool execution completed in {step_time:.3f} seconds")
    print(f"✅ Tools executed: {len(tools_executed)}")
    print(f"❌ Errors: {error_count - state.get('error_count', 0)}")
    
    return updated_state

def should_continue(state: DataAnalysisState):
    """Enhanced routing with debugging."""
    print(f"\n🔀 ROUTING DECISION")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    print(f"📨 Last message type: {type(last_message).__name__}")
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        decision = "tools"
        print(f"➡️  Decision: {decision} (found {len(last_message.tool_calls)} tool calls)")
    else:
        decision = END
        print(f"➡️  Decision: {decision} (no tool calls)")
    
    return decision

def create_data_analysis_graph():
    """Create the data analysis graph with debugging."""
    print("🏗️  Building Data Analysis Graph...")
    
    memory = MemorySaver()
    workflow = StateGraph(DataAnalysisState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Compile
    graph = workflow.compile(checkpointer=memory)
    
    print("✅ Graph compiled successfully!")
    return graph

def print_execution_summary(final_state):
    """Print comprehensive execution summary."""
    print(f"\n{'='*60}")
    print("📈 EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    # Basic metrics
    total_steps = final_state.get('step_count', 0)
    tools_used = set(final_state.get('tools_called', []))
    total_tools = len(final_state.get('tools_called', []))
    errors = final_state.get('error_count', 0)
    user_intent = final_state.get('user_intent', 'unknown')
    
    print(f"🎯 User Intent: {user_intent}")
    print(f"🔢 Total Steps: {total_steps}")
    print(f"🔧 Unique Tools Used: {', '.join(tools_used) if tools_used else 'None'}")
    print(f"🔧 Total Tool Calls: {total_tools}")
    print(f"❌ Errors: {errors}")
    
    # Performance metrics
    performance = final_state.get('performance_metrics', {})
    if performance:
        print(f"\n⏱️  PERFORMANCE BREAKDOWN:")
        total_time = sum(v for k, v in performance.items() if 'step_' in k and 'agent_' in k) + \
                    sum(v for k, v in performance.items() if 'step_' in k and 'tools_' in k)
        print(f"   Total Time: {total_time:.3f} seconds")
        
        agent_time = sum(v for k, v in performance.items() if 'agent_step_' in k)
        tools_time = sum(v for k, v in performance.items() if 'tools_step_' in k)
        llm_time = sum(v for k, v in performance.items() if 'llm_step_' in k)
        
        print(f"   Agent Time: {agent_time:.3f}s ({agent_time/total_time*100:.1f}%)")
        print(f"   Tools Time: {tools_time:.3f}s ({tools_time/total_time*100:.1f}%)")
        print(f"   LLM Time: {llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
    
    # Execution trace
    print(f"\n📋 DETAILED EXECUTION TRACE:")
    for trace in final_state.get('execution_trace', []):
        timestamp = trace.get('timestamp', '').split('T')[1][:12] if 'T' in trace.get('timestamp', '') else ''
        step = trace.get('step', '?')
        node = trace.get('node', 'unknown')
        duration = trace.get('duration', 0)
        
        if node == 'agent':
            intent = trace.get('user_intent', '')
            tools = trace.get('tool_calls', [])
            if tools:
                print(f"   Step {step} [{timestamp}] 🤖 Agent ({duration:.3f}s) → Intent: {intent}, Planning: {', '.join(tools)}")
            else:
                print(f"   Step {step} [{timestamp}] 🤖 Agent ({duration:.3f}s) → Intent: {intent}, Direct response")
        elif node == 'tools':
            tools = trace.get('tools_executed', [])
            errors = trace.get('errors', 0)
            error_str = f", {errors} errors" if errors > 0 else ""
            print(f"   Step {step} [{timestamp}] 🔧 Tools ({duration:.3f}s) → Executed: {', '.join(tools)}{error_str}")

def run_data_agent_with_debugging(graph, query, thread_id="debug_session"):
    """Run the data analysis agent with full debugging."""
    print(f"\n🚀 STARTING DATA ANALYSIS")
    print(f"❓ Query: '{query}'")
    print(f"🧵 Thread: {thread_id}")
    print(f"🕐 Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_step": "starting",
        "step_count": 0,
        "tools_called": [],
        "execution_trace": [],
        "performance_metrics": {},
        "user_intent": "unknown",
        "error_count": 0
    }
    
    final_state = None
    final_response = None
    
    # Stream execution
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        final_state = list(event.values())[0]  # Get the state from the event
        
        # Extract final response
        if "agent" in event and final_state.get("messages"):
            last_msg = final_state["messages"][-1]
            if isinstance(last_msg, AIMessage) and not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                final_response = last_msg.content
    
    # Print summary
    if final_state:
        print_execution_summary(final_state)
    
    print(f"\n{'='*60}")
    if final_response:
        print(f"💬 FINAL RESPONSE:")
        print(f"{final_response}")
    print(f"{'='*60}")
    
    return final_response

if __name__ == "__main__":
    # Create and visualize graph
    graph = create_data_analysis_graph()
    
    print("\n📊 Graph Structure:")
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except:
        print("   START → agent → [decision]")
        print("            ↓         ↓")
        print("          END     tools → agent")
    
    # Demo queries
    demo_queries = [
        "Load the dataset and tell me about it",
        "What are the feature descriptions?",
        "Show me descriptive statistics for the LIMIT_BAL feature"
    ]
    
    print(f"\n🎮 RUNNING DEMO QUERIES")
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'🔹'*20} DEMO {i} {'🔹'*20}")
        run_data_agent_with_debugging(graph, query, f"demo_{i}")
        input("\nPress Enter to continue to next demo...")
    
    # Interactive mode
    print(f"\n🎮 INTERACTIVE MODE")
    print("Ask questions about the UCI Credit Card dataset (or 'exit' to quit)")
    
    while True:
        user_query = input(f"\n❓ Your question: ")
        if user_query.lower() == 'exit':
            break
        
        run_data_agent_with_debugging(graph, user_query, "interactive")