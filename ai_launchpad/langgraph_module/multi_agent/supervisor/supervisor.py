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
from langgraph.types import Command, RunnableConfig

load_dotenv()

# load system prompts
supervisor_prompt = open("prompts/supervisor.md", "r").read()
supervisor_prompt.format(current_datetime=datetime.now())

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

    return Command(
        goto=f"call_{agent_name}",
        update=update
        )


async def call_researcher(state: SupervisorState, config: RunnableConfig):
    research_response = await research_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=state.task_description)],
            },
        config=config,
    )

    ai_message = AIMessage(name="researcher", content=research_response["messages"][-1].content)

    return Command(update={
        "research_reports": research_response["research_reports"],
        "messages": [ai_message],
        })

async def call_copywriter(state: SupervisorState, config: RunnableConfig):
    copywriter_response = await copywriter_agent.ainvoke(
        input={
            "messages": [HumanMessage(content=state.task_description)],
            "research_reports": state.research_reports,
            },
        config=config,
        )
    
    ai_message = AIMessage(name="copywriter", content=copywriter_response["messages"][-1].content)

    return Command(update={"messages": [ai_message]})

llm = ChatOpenAI(
    name="Supervisor",
    model="gpt-5-mini-2025-08-07",
    reasoning_effort="low",
)

tools=[handoff_to_subagent]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


async def supervisor(state: SupervisorState):
    system_prompt = SystemMessage(content=f"""You are a supervisor managing a team of agents. You can call on the agents to perform tasks for you. Do not rely on your own knowledge, always use the tools to answer the user's questions. Do not offer to do anything for the user that are not explicitly capable of doing, given the tools you have access to.

    WORKFLOW INSTRUCTIONS:
    1. ANALYZE the user's request and identify if it requires multiple research angles or subtopics
    2. BREAK DOWN complex topics into 2-4 atomic research tasks (each focusing on one specific aspect)
    3. COMMUNICATE your plan to the user and then proceed
    4. CALL the researcher multiple times - once for each atomic research task
    5. WAIT for all research to complete before calling the copywriter
    6. CALL the copywriter once with clear instructions to synthesize all research reports

    RESEARCH TASK GUIDELINES:
    - Each research task should be atomic (focused on ONE specific angle/subtopic)
    - For broad topics, always break into multiple research calls (e.g., current state + trends + challenges + future predictions)
    - For content requests about industries/technologies, research: market data + key players + challenges + opportunities
    - For "how-to" content, research: current methods + best practices + tools + case studies
    - Each research task should specify target sources and expected deliverables

    IMPORTANT: Call the researcher multiple times for comprehensive coverage. One broad research call is insufficient for quality content creation.

    Do not repeat the output of the researcher or copywriter. Instead, summarize the outputs, provide additional context if necessary, and let the user know that the task has been completed.

    <tools>
    handoff_to_subagent: Use this tool to assign a task to either the researcher or copywriter agent. Specify the agent_name ("researcher" or "copywriter") and task_description.
    </tools>

    <agents>
    researcher: Performs focused research on specific subtopics. CALL MULTIPLE TIMES for comprehensive coverage:
        - Each call should focus on ONE specific research angle
        - All research reports are automatically saved for the copywriter to access
        - Typical pattern: 2-4 research calls per content request
        - Examples: "current market data", "key challenges", "future trends", "best practices"

    copywriter: Creates content using ALL available research reports:
        - Call ONCE after all research is complete
        - Has access to all previously generated research reports
        - Can synthesize multiple research angles into cohesive content
    </agents>
                                  
    <example>
    User Request: "Write a blog post about the future of remote work, including how AI tools are changing productivity, the challenges companies face, and predictions for the next 5 years."

    Supervisor Plan:
    1. Break down into atomic research tasks:
       - Research current remote work statistics and trends (2023-2024)
       - Research AI productivity tools and their impact on remote teams
       - Research challenges companies face with remote work management
       - Research expert predictions and forecasts for remote work (2025-2030)

    2. Call researcher multiple times for comprehensive coverage:
       - Call 1: handoff_to_subagent(agent_name="researcher", task_description="Research current remote work statistics, adoption rates, and key trends from 2023-2024. Include data on productivity metrics, employee satisfaction, and company policies. Focus on authoritative sources like Gallup, McKinsey, and Bureau of Labor Statistics.")

       - Call 2: handoff_to_subagent(agent_name="researcher", task_description="Research AI productivity tools specifically designed for remote teams. Include tools for collaboration, project management, communication, and automation. Analyze their impact on team efficiency and provide specific examples and case studies.")

       - Call 3: handoff_to_subagent(agent_name="researcher", task_description="Research the main challenges companies face with remote work management. Include issues like team coordination, company culture, performance monitoring, cybersecurity, and employee isolation. Provide solutions and best practices.")

       - Call 4: handoff_to_subagent(agent_name="researcher", task_description="Research expert predictions and forecasts for the future of remote work from 2025-2030. Include insights from industry leaders, technology trends, generational shifts, and potential policy changes. Focus on credible future-looking analysis.")

    3. After all research is complete, call copywriter:
       - Call 5: handoff_to_subagent(agent_name="copywriter", task_description="Write a comprehensive 1500-2000 word blog post about the future of remote work using all the research reports. Structure it with: engaging introduction, current state analysis, AI tools impact, challenges and solutions, future predictions, and actionable conclusion. Use a professional but accessible tone.")

    This approach ensures each research task is atomic, focused, and builds comprehensive knowledge before content creation.
    </example>

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
