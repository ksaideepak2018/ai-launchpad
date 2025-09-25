import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated, List
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch, TavilyExtract
from datetime import datetime
from langgraph.types import Command

load_dotenv()


@tool
async def search_web(
    query: str,
    num_results: int = 3
    ):
    """Search the web and get back a list of search results including the page title, url, and a short summary of each webpage.

    Args:
        query: The search query.
        num_results: The number of results to return, max is 3.

    Returns:
        A dictionary of the search results.
    """
    web_search = TavilySearch(max_results=min(num_results, 3), topic="general")
    search_results = web_search.invoke(input={"query": query})
    
    processed_results = {
        "query": query,
        "results": []
    }

    for result in search_results["results"]:
        processed_results["results"].append({
            "title": result["title"],
            "url": result["url"],
            "content_preview": result["content"]
        })

    return processed_results


@tool
async def extract_content_from_webpage(urls: List[str]):
    """Extract the content from a webpage.

    Args:
        url: The url of the webpage to extract content from.
    """
    web_extract = TavilyExtract()
    results = web_extract.invoke(input={"urls": urls})["results"]
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
    """Generate a research report on a specific topic.

    Args:
        topic: The topic to research.
        report: The research report.
    """
    research_report = ResearchReport.model_validate({
        "topic": topic,
        "report": report
        })

    return Command(update={
        "research_reports": [research_report],
        "messages": [ToolMessage(
            name="generate_research_report",
            content=research_report.model_dump_json(),
            tool_call_id=tool_call_id,
            )],
        })


class ResearcherState(BaseModel):
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

async def researcher(state: ResearcherState):
    system_prompt = SystemMessage(content=f"""You are a research assistant. Your job is to help the user answer questions by performing research. Do not rely on your own knowledge, always use the tools to answer the user's questions. ALWAYS use the tools to generate the final research report.

    <tools>
    search_web: Search the web. Returned results include the page title, url, and a content snippet of each webpage.
    extract_content_from_webpage: Extract the complete contents from a webpage given the url.
    generate_research_report: Generate a research report on a specific topic.
    </tools>
                                  
    You should use the search_web and extract_content_from_webpage tools to gather information. You can call these tools multiple times to gather all the information you need and then use the generate_research_report tool to generate the final research report.
                                  
    <report_format>
    The output of the final report should be in markdown format and always include a list of citations at the end of the report with the format: [Source Name] (URL).
    </report_format>
                                  
    <generate_research_report_example>
    {{
        "topic": "Top 5 companies in the world by market value",
        "report": "## Executive Summary
            Here are the top 5 companies in the world by market value (market capitalization):
                                        
            1. Nvidia — $4.3 trillion
            2. Microsoft — $3.8 trillion
            3. Apple — $3.5 trillion
            4. Alphabet (Google) — $3 trillion
            5. Amazon — $2.5 trillion
                                  
            ## Additional Sections...
            
            ## Citations
            [1] [Motley Fool — "The Largest Companies by Market Cap" (updated Sep 3 / data listed Sep 16, 2025)](https://www.fool.com/research/largest-companies-by-market-cap/)"
    }}
    </generate_research_report_example>
                                  
    The current date and time is {datetime.now()}.
    """)
    response = llm_with_tools.invoke([system_prompt] + state.messages)
    return {"messages": [response]}

async def researcher_router(state: ResearcherState) -> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(ResearcherState)

builder.add_node(researcher)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("researcher")
builder.add_edge("tools", "researcher")
builder.add_conditional_edges(
    "researcher",
    researcher_router,
    {
        "tools": "tools",
        END: END,
    }
)

# don't use a checkpointer if using as a subgraph, the parent graph's checkpointer will be used
graph = builder.compile()

# graph = builder.compile(checkpointer=MemorySaver())

# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())
