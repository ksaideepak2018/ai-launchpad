import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel


load_dotenv()


# Resolve files relative to researcher.py instead of the terminal's current folder.
CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = CURRENT_DIR / "prompts" / "researcher.md"

with PROMPT_PATH.open("r", encoding="utf-8") as prompt_file:
    researcher_prompt = prompt_file.read()


def normalize_tavily_response(response: Any) -> list[dict]:
    """
    Normalize Tavily responses across different package versions.

    Tavily may return:
    - A dictionary containing a "results" list
    - A JSON string containing a "results" list
    - A plain string
    - A list of result dictionaries
    """

    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]

    if isinstance(response, dict):
        results = response.get("results", [])

        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]

        return []

    if isinstance(response, str):
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            return [
                {
                    "url": "unknown",
                    "raw_content": response,
                }
            ]

        if isinstance(parsed_response, list):
            return [
                item for item in parsed_response
                if isinstance(item, dict)
            ]

        if isinstance(parsed_response, dict):
            results = parsed_response.get("results", [])

            if isinstance(results, list):
                return [
                    item for item in results
                    if isinstance(item, dict)
                ]

    return []


@tool
async def search_web(
    query: str,
    num_results: int = 3,
):
    """
    Search the web and return page titles, URLs, and short content previews.

    Args:
        query: The search query.
        num_results: Number of results to return. Maximum is 3.

    Returns:
        A dictionary containing the processed search results.
    """

    web_search = TavilySearch(
        max_results=min(num_results, 3),
        topic="general",
    )

    response = web_search.invoke(
        input={"query": query}
    )

    search_results = normalize_tavily_response(response)

    processed_results = {
        "query": query,
        "results": [],
    }

    for result in search_results:
        processed_results["results"].append(
            {
                "title": result.get("title", "Untitled"),
                "url": result.get("url", ""),
                "content_preview": result.get(
                    "content",
                    result.get("raw_content", ""),
                ),
            }
        )

    return processed_results


@tool
async def extract_content_from_webpage(
    urls: List[str],
):
    """
    Extract content from one or more webpages.

    Args:
        urls: A list of webpage URLs.

    Returns:
        A list of dictionaries containing extracted webpage content.
    """

    web_extract = TavilyExtract()

    response = web_extract.invoke(
        input={"urls": urls}
    )

    results = normalize_tavily_response(response)

    return results


class ResearchReport(BaseModel):
    topic: str
    report: str


@tool
async def generate_research_report(
    topic: str,
    report: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Generate and save a structured research report.

    Args:
        topic: The topic being researched.
        report: The completed research report.
        tool_call_id: The ID of the current tool call.

    Returns:
        A Command that updates the graph state.
    """

    research_report = ResearchReport.model_validate(
        {
            "topic": topic,
            "report": report,
        }
    )

    return Command(
        update={
            "research_reports": [research_report],
            "messages": [
                ToolMessage(
                    name="generate_research_report",
                    content=research_report.model_dump_json(),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


class ResearcherState(BaseModel):
    """
    State used by the researcher agent.

    research_reports is shared with the supervisor state, allowing
    the supervisor to access researcher output and send it to the
    copywriter.
    """

    messages: Annotated[list, add_messages] = []
    research_reports: Annotated[list, operator.add] = []


tools = [
    search_web,
    extract_content_from_webpage,
    generate_research_report,
]


llm = ChatOpenAI(
    name="Researcher",
    model="gpt-5-mini-2025-08-07",
    reasoning_effort="minimal",
)

llm_with_tools = llm.bind_tools(tools)


async def researcher(
    state: ResearcherState,
):
    """Run the researcher agent."""

    response = await llm_with_tools.ainvoke(
        [
            SystemMessage(
                content=researcher_prompt.format(
                    current_datetime=datetime.now()
                )
            )
        ]
        + state.messages
    )

    return {
        "messages": [response]
    }


async def researcher_router(
    state: ResearcherState,
) -> str:
    """
    Route to the tools node when the researcher requests a tool.
    Otherwise, finish the researcher graph.
    """

    if state.messages[-1].tool_calls:
        return "tools"

    return END


builder = StateGraph(ResearcherState)

builder.add_node(
    "researcher",
    researcher,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

builder.set_entry_point("researcher")

builder.add_edge(
    "tools",
    "researcher",
)

builder.add_conditional_edges(
    "researcher",
    researcher_router,
    {
        "tools": "tools",
        END: END,
    },
)


# Do not use a separate checkpointer when this graph is used as a subgraph.
# The parent supervisor graph provides the checkpointer.
graph = builder.compile()


# Optional standalone checkpointer:
# from langgraph.checkpoint.memory import MemorySaver
# graph = builder.compile(checkpointer=MemorySaver())


# Optional graph visualization:
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())