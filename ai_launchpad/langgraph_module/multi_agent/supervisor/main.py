from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.types import RunnableConfig
from ai_launchpad.langgraph_module.multi_agent.supervisor.supervisor import graph as supervisor_graph, SupervisorState
from langchain_core.messages import HumanMessage, AIMessageChunk

load_dotenv()

async def stream_graph_responses(
        input: SupervisorState,
        graph: StateGraph,
        **kwargs
        ):
    """Asynchronously stream the result of the graph run with subgraph support.

    Args:
        input: The input to the graph.
        graph: The compiled graph.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The final LLM or tool call response
    """
    # ANSI color codes
    COLORS = {
        'researcher': '\033[36m',  # Cyan for researcher
        'copywriter': '\033[35m',  # Magenta for copywriter
        'supervisor': '\033[32m',  # Green for supervisor
        'tool': '\033[33m',        # Yellow for tools
        'reset': '\033[0m',        # Reset color
        'bold': '\033[1m',         # Bold text
        'dim': '\033[2m'           # Dim text
    }

    # Track current AI message source to detect transitions
    current_ai_source = None
    last_was_tool = False

    async for chunk in graph.astream(
        input=input,
        stream_mode="messages",
        subgraphs=True,
        **kwargs
        ):
        # When subgraphs=True, the structure is (namespace, (message_chunk, metadata))
        namespace, (message_chunk, _) = chunk

        if isinstance(message_chunk, AIMessageChunk):
            # Determine the source of this AI message directly from namespace
            if namespace:
                # This is from a subgraph - detect agent from namespace
                namespace_str = str(namespace)
                if "call_researcher" in namespace_str:
                    ai_source = "researcher"
                    color = COLORS['researcher']
                    agent_name = "🔬 Researcher"
                elif "call_copywriter" in namespace_str:
                    ai_source = "copywriter"
                    color = COLORS['copywriter']
                    agent_name = "✍️ Copywriter"
                else:
                    # Fallback for unknown subgraphs
                    ai_source = "researcher"
                    color = COLORS['researcher']
                    agent_name = "🔬 Researcher"
            else:
                # This is from the main graph (supervisor)
                ai_source = "supervisor"
                color = COLORS['supervisor']
                agent_name = "🎯 Supervisor"

            # Check if we're transitioning between different AI sources
            if current_ai_source != ai_source and current_ai_source is not None:
                # Add visual separation when switching between agents
                yield f"\n\n{COLORS['dim']}{'─' * 50}{COLORS['reset']}\n"
                yield f"{COLORS['bold']}{color}{agent_name}{COLORS['reset']}\n\n"
                last_was_tool = False
            elif current_ai_source is None:
                # First AI message - show which agent is starting
                yield f"{COLORS['bold']}{color}{agent_name}{COLORS['reset']}\n\n"
                last_was_tool = False
            elif last_was_tool:
                # Coming back from tool calls to AI content
                yield f"\n{color}"
                last_was_tool = False

            current_ai_source = ai_source

            if message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    yield f"{COLORS['reset']}\n\n{COLORS['tool']}🔧 Tool call completed{COLORS['reset']}\n\n"
                    last_was_tool = True

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]
                tool_name = tool_chunk.get("name", "")
                args = tool_chunk.get("args", "")

                if tool_name:
                    yield f"{COLORS['reset']}\n\n{COLORS['tool']}🔧 TOOL CALL: {tool_name}{COLORS['reset']}\n{COLORS['dim']}"
                    last_was_tool = True
                if args:
                    yield args
            else:
                # Stream the actual content with color
                if message_chunk.content:
                    if not last_was_tool:
                        yield f"{color}{message_chunk.content}"
                    else:
                        yield f"{COLORS['reset']}{color}{message_chunk.content}"
        else:
            # Handle other message types
            # yield f"[TOOL MESSAGE] {type(message_chunk)}: {message_chunk}\n"
            pass

    # Reset color at the end
    yield f"{COLORS['reset']}"


async def main():
    """Main function to run the supervisor with subgraphs."""
    try:
        config = RunnableConfig(configurable={
            "thread_id": "1",
            "recursion_limit": 50,
        })

        print("\n🎯 Running Supervisor with Subgraphs")
        print("=" * 50)

        while True:
            user_input = input("\n\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\n\nExit command received. Exiting...\n\n")
                break

            print(f"\n ----- 🥷 Human ----- \n\n{user_input}\n")

            graph_input = SupervisorState(
                messages=[HumanMessage(content=user_input)]
            )

            print(f" ---- 🤖 AI Agents ---- \n")
            async for response in stream_graph_responses(graph_input, supervisor_graph, config=config):
                print(response, end="", flush=True)

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
    
# Example prompts
# write a linkedin post on the top AI tools that small businesses and entrepreneurs need to be using to scale their businesses. include real-world examples and case studies where businesses are using these tools to scale their business with real numbers. include a call to action at the end for readers to follow me for more actionable playbooks on how to generate real value for their business.