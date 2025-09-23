import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from ai_launchpad.langgraph_module.multi_agent.supervisor.researcher import graph as research_agent
from ai_launchpad.langgraph_module.multi_agent.supervisor.copywriter import graph as copywriter_agent
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

load_dotenv()

class ResearchReport(BaseModel):
    topic: str
    report: str

class SupervisorState(BaseModel):
    messages: Annotated[list, add_messages] = []
    research_reports: Annotated[list, operator.add] = []
    task_description: str | None = None


@tool
async def handoff_to_subagent(
    agent_name: Literal["researcher", "copywriter"],
    task_description: str,
    state: Annotated[SupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    ):
    """Use this tool to assign a task to a sub agent.
    
    Args:
        agent_name: The name of the agent to handoff the task to. Valid agent names are researcher and copywriter.
        task_description: The description of the task to be completed.
    """
    update = {
        "task_description": task_description,
        "messages": [ToolMessage(
            name=f"handoff_to_{agent_name}",
            content=f"Successfully handed off task to {agent_name}.",
            tool_call_id=tool_call_id,
        )],
        }

    if agent_name == "researcher":
        update["research_reports"]= state.research_reports if state.research_reports else []

    return Command(
        goto=f"call_{agent_name}",
        update=update
        )


async def call_researcher(state: SupervisorState):
    research_response = await research_agent.ainvoke(
        input={"messages": [HumanMessage(content=state.task_description)]}
    )

    ai_response = AIMessage(name="researcher", content=research_response["messages"][-1].content)

    research_report = ResearchReport.model_validate({
        "topic": state.task_description,
        "report": research_response["messages"][-1].content
        })

    return Command(update={
        "research_reports": [research_report],
        "messages": [ai_response],
        })

async def call_copywriter(state: SupervisorState):
    copywriter_response = await copywriter_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=state.task_description)],
            "research_reports": state.research_reports,
            })
    
    ai_message = AIMessage(name="copywriter", content=copywriter_response["messages"][-1].content)

    return Command(update={"messages": [ai_message]})

llm = ChatOpenAI(
    name="Supervisor",
    model="gpt-5-mini-2025-08-07",
    reasoning_effort="minimal",
)

tools=[handoff_to_subagent]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


async def supervisor(state: SupervisorState):
    system_prompt = SystemMessage(content=f"""You are a supervisor managing a team of agents. You can call on the agents to perform tasks for you. Do not rely on your own knowledge, always use the tools to answer the user's questions. Do not offer to do anything for the user that are not explicitly capable of doing given the tools you have access to.

    Always start by thinking through the user's request and creating a short plan for how you will complete the request, including the agents you will call on and the order in which you will call them. Then use the appropriate tools as outlined in your plan.

    IMPORTANT: Call only ONE agent at a time. Wait for their response before calling the next agent.

    Do not repeat the output of the researcher or copywriter. Instead, summarize the outputs, provide additional context if necessary, and let the user know that the task has been completed.

    <agents>
    researcher: The researcher agent can perform deep dives on specific topics and generate comprehensive reports.
    copywriter: The copywriter agent can generate high quality content such as linkedin posts and blog articles based on the research reports.
    </agents>

    <tools>
    handoff_to_subagent: Use this tool to assign a task to either the researcher or copywriter agent. Specify the agent_name ("researcher" or "copywriter") and task_description.
    </tools>

    The current date and time is {datetime.now()}.
    """)
    response = llm_with_tools.invoke([system_prompt] + state.messages)
    return {"messages": [response]}

async def supervisor_router(state: SupervisorState) -> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return END

builder = StateGraph(SupervisorState)

builder.add_node(supervisor)
builder.add_node("tools", ToolNode(tools))
builder.add_node(call_researcher)
builder.add_node(call_copywriter)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "tools": "tools",
        END: END,
    }
)
# builder.add_edge("tools", "supervisor")
builder.add_edge("call_researcher", "supervisor")
builder.add_edge("call_copywriter", "supervisor")

graph = builder.compile(checkpointer=MemorySaver())


# Visualize the graph
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())
