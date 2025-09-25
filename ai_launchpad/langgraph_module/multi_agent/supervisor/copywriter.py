import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated, List
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch, TavilyExtract
from datetime import datetime
from langgraph.prebuilt import InjectedState
import os

load_dotenv()


class CopyWriterState(BaseModel):
    messages: Annotated[list, add_messages] = []
    research_reports: Annotated[list, operator.add] = []


@tool
async def review_research_reports(
    state: Annotated[CopyWriterState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Use this tool to review available research reports to inform your writing."""
    return [report.model_dump_json() for report in state.research_reports]

@tool
async def generate_linkedin_post(
    title: str,
    content: str,
    state: Annotated[CopyWriterState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Use this tool to generate a LinkedIn post.
    
    Args:
        content: The content of the post in markdown format.
    """
    filename=f"ai_files/{title}.md"
    with open(filename, "w") as f:
        f.write(content)

    return f"The LinkedIn post has been generated and saved to {filename}"

@tool
async def generate_blog_post(
    title: str,
    content: str,
    state: Annotated[CopyWriterState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Use this tool to generate a blog post.
    
    Args:
        content: The content of the post in markdown format.
    """
    filename=f"ai_files/{title}.md"
    with open(filename, "w") as f:
        f.write(content)

    return f"The blog post has been generated and saved to {filename}"

llm = ChatOpenAI(
    name="CopyWriter",
    model="gpt-5-mini-2025-08-07",
    reasoning_effort="minimal",
)

tools=[
    review_research_reports,
    generate_linkedin_post, 
    generate_blog_post
    ]
llm_with_tools = llm.bind_tools(tools)

# load example content
linkedin_example = open("example_content/linkedin.md", "r").read()
blog_example = open("example_content/blog.md", "r").read()

async def copywriter(state: CopyWriterState):
    system_prompt = SystemMessage(content=f"""You are a copywriter. Your job is to write highly engaging content based on the topic provided by the user. For some topics, you may be provided additional context in the form of research reports. Always check to see if there are research reports available and use them to inform your writing. ALWAYS use the tools to generate the content.

    <tools>
    review_research_reports: Use this tool to review the research reports to inform your writing. If there are no research reports available but you think they would be helpful, you should request the research you need to write the content.
    generate_linkedin_post: Use this tool to generate a LinkedIn post.
    generate_blog_post: Use this tool to generate a blog post.
    </tools>
                                  
    <example_linkedin_post>
        {linkedin_example}
    </example_linkedin_post>

    <example_blog_post>
        {blog_example}
    </example_blog_post>
                                  
    The current date and time is {datetime.now()}.
    """)
    response = llm_with_tools.invoke([system_prompt] + state.messages)
    return {"messages": [response]}

async def copywriter_router(state: CopyWriterState) -> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(CopyWriterState)

builder.add_node(copywriter)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("copywriter")

builder.add_conditional_edges(
    "copywriter",
    copywriter_router,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_edge("tools", "copywriter")

# don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
graph = builder.compile()

# graph = builder.compile(checkpointer=MemorySaver())


# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())
    