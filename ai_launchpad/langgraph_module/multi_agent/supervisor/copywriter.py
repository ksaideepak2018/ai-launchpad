import operator
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode, InjectedState
from pydantic import BaseModel


load_dotenv()


# Resolve all paths relative to this file instead of the terminal location.
CURRENT_DIR = Path(__file__).resolve().parent

PROMPTS_DIR = CURRENT_DIR / "prompts"
EXAMPLE_CONTENT_DIR = CURRENT_DIR / "example_content"

COPYWRITER_PROMPT_PATH = PROMPTS_DIR / "copywriter.md"
LINKEDIN_EXAMPLE_PATH = EXAMPLE_CONTENT_DIR / "linkedin.md"
BLOG_EXAMPLE_PATH = EXAMPLE_CONTENT_DIR / "blog.md"

# Save generated files inside the repository's ai_files folder.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = PROJECT_ROOT / "ai_files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with COPYWRITER_PROMPT_PATH.open("r", encoding="utf-8") as prompt_file:
    copywriter_prompt = prompt_file.read()

with LINKEDIN_EXAMPLE_PATH.open("r", encoding="utf-8") as example_file:
    linkedin_example = example_file.read()

with BLOG_EXAMPLE_PATH.open("r", encoding="utf-8") as example_file:
    blog_example = example_file.read()


def create_safe_filename(title: str) -> str:
    """
    Convert a post title into a valid Windows-safe filename.
    """

    safe_title = re.sub(r'[<>:"/\\|?*]', "", title)
    safe_title = re.sub(r"\s+", "_", safe_title.strip())
    safe_title = safe_title[:100]

    return safe_title or "generated_content"


class CopyWriterState(BaseModel):
    """
    State used by the copywriter agent.

    The research_reports attribute is shared with the supervisor state.
    This allows the supervisor to send research reports to the copywriter.
    """

    messages: Annotated[list, add_messages] = []
    research_reports: Annotated[list, operator.add] = []


@tool
async def review_research_reports(
    state: Annotated[CopyWriterState, InjectedState],
):
    """
    Review the research reports available in the current graph state.

    Returns:
        A list of serialized research reports.
    """

    reports = []

    for report in state.research_reports:
        if hasattr(report, "model_dump_json"):
            reports.append(report.model_dump_json())
        elif isinstance(report, dict):
            reports.append(report)
        else:
            reports.append(str(report))

    return reports


@tool
async def generate_linkedin_post(
    title: str,
    content: str,
):
    """
    Generate and save a LinkedIn post.

    Args:
        title: The title of the LinkedIn post.
        content: The LinkedIn post content in Markdown format.

    Returns:
        A message containing the saved file location.
    """

    safe_title = create_safe_filename(title)
    file_path = OUTPUT_DIR / f"{safe_title}.md"

    with file_path.open("w", encoding="utf-8") as output_file:
        output_file.write(content)

    return f"The LinkedIn post has been generated and saved to {file_path}"


@tool
async def generate_blog_post(
    title: str,
    content: str,
):
    """
    Generate and save a blog post.

    Args:
        title: The title of the blog post.
        content: The blog post content in Markdown format.

    Returns:
        A message containing the saved file location.
    """

    safe_title = create_safe_filename(title)
    file_path = OUTPUT_DIR / f"{safe_title}.md"

    with file_path.open("w", encoding="utf-8") as output_file:
        output_file.write(content)

    return f"The blog post has been generated and saved to {file_path}"


llm = ChatOpenAI(
    name="CopyWriter",
    model="gpt-5-mini-2025-08-07",
    reasoning_effort="minimal",
)


tools = [
    review_research_reports,
    generate_linkedin_post,
    generate_blog_post,
]


llm_with_tools = llm.bind_tools(tools)


async def copywriter(
    state: CopyWriterState,
):
    """Run the copywriter agent."""

    system_prompt = SystemMessage(
        content=copywriter_prompt.format(
            current_datetime=datetime.now(),
            linkedin_example=linkedin_example,
            blog_example=blog_example,
        )
    )

    response = await llm_with_tools.ainvoke(
        [system_prompt] + state.messages
    )

    return {
        "messages": [response]
    }


async def copywriter_router(
    state: CopyWriterState,
) -> str:
    """
    Route to the tools node if the copywriter requests a tool.
    Otherwise, end the copywriter graph.
    """

    if state.messages[-1].tool_calls:
        return "tools"

    return END


builder = StateGraph(CopyWriterState)

builder.add_node(
    "copywriter",
    copywriter,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

builder.set_entry_point("copywriter")

builder.add_conditional_edges(
    "copywriter",
    copywriter_router,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge(
    "tools",
    "copywriter",
)


# Do not use a separate checkpointer when this graph runs as a subgraph.
# The parent supervisor graph provides the checkpointer.
graph = builder.compile()


# Optional standalone checkpointer:
# from langgraph.checkpoint.memory import MemorySaver
# graph = builder.compile(checkpointer=MemorySaver())


# Optional graph visualization:
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())