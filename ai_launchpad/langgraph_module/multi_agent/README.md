# Multi-Agent Systems with LangGraph

This module demonstrates how to design and implement multi-agent AI applications using LangGraph.

The examples focus on dividing complex, open-ended work across multiple specialized agents. Each agent has its own responsibilities, prompts, tools, and context, while LangGraph manages communication, routing, state, and execution.

## Overview

Before building a multi-agent system, it is important to determine whether the problem actually requires one.

The simplest architecture that solves the business problem is usually the best choice.

A useful decision order is:

```text
Deterministic workflow
        |
        v
Single AI agent
        |
        v
Multi-agent system
```

Start with a deterministic workflow when the steps are predictable. Move to a single agent when the task is open-ended. Use multiple agents only when one agent becomes difficult to manage, loses focus, or cannot perform the complete task reliably.

## Choosing the Right Architecture

### 1. Deterministic AI Workflow

Use a workflow when the sequence of steps is known in advance.

Example:

```text
Receive a document
→ Extract text
→ Summarize the content
→ Save the result
```

Benefits:

- Easy to understand
- Easy to test
- Predictable execution
- Lower API cost
- Easier monitoring
- Better production reliability

A workflow should be the default choice when the process can be clearly defined in code.

### 2. Single AI Agent

Use a single agent when the problem is open-ended and the required steps cannot be fully determined beforehand.

Example:

```text
User request
→ Agent analyzes the request
→ Agent selects tools
→ Agent performs actions
→ Agent returns the result
```

A single agent may decide whether it needs to:

- Search the web
- Retrieve documents
- Call an API
- Ask for clarification
- Generate content
- Perform calculations

A single agent is often sufficient for moderately complex tasks.

### 3. Multi-Agent System

Use a multi-agent system when:

- The problem is open-ended
- A single agent is no longer performing reliably
- The task contains clearly different responsibilities
- Specialized tools or prompts are needed
- Context becomes too large or distracting
- Different tasks can run independently
- Multiple stages require separate reasoning

Example:

```text
User
  |
  v
Supervisor
  |
  +---- Researcher
  |
  +---- Analyst
  |
  +---- Copywriter
  |
  v
Final response
```

## Why Use Multiple Agents?

### Specialized Responsibilities

Each agent can focus on one task.

For example:

- A researcher gathers information
- An analyst evaluates evidence
- A copywriter creates polished content
- A supervisor coordinates the workflow

This helps prevent one agent from being overloaded with unrelated instructions.

### Better Context Management

Each agent can receive only the information relevant to its task.

For example, a researcher may receive only:

```text
Research customer-support automation for small businesses.
```

It does not need the supervisor's complete conversation history.

Focused context can improve:

- Accuracy
- Tool selection
- Prompt adherence
- Response consistency
- Debugging

### Independent Tools and Models

Different agents can use different:

- Language models
- System prompts
- Tools
- Temperature settings
- Reasoning settings
- Memory
- Retrieval sources

A researcher may use web-search tools, while a copywriter may use writing examples and formatting tools.

### Modularity

Each agent can be developed and tested separately.

A well-designed researcher agent may later be reused for:

- Competitor analysis
- Market research
- Content creation
- Product research
- Customer prospecting

### Parallel Execution

Independent tasks may be executed in parallel.

For example:

```text
Supervisor
  |
  +---- Research market trends
  |
  +---- Research competitor activity
  |
  +---- Research customer feedback
```

Parallel execution can reduce total processing time, although it may increase API cost and implementation complexity.

## Challenges of Multi-Agent Systems

Multi-agent systems should not be used only because they are technically interesting.

They introduce several challenges:

- Higher API usage
- Increased token consumption
- Longer execution time
- More complex state management
- More failure points
- Harder debugging
- More difficult evaluation
- More complicated observability
- Risk of repeated handoffs
- Risk of agents producing conflicting results
- Higher maintenance cost

A multi-agent architecture should provide a clear benefit over a workflow or single agent.

## Multi-Agent Patterns

LangGraph supports several multi-agent design patterns.

## 1. Supervisor Pattern

A central supervisor coordinates specialized sub-agents.

```text
User
  |
  v
Supervisor
  |
  +---- Researcher
  |
  +---- Copywriter
  |
  v
Final response
```

The supervisor:

- Communicates with the user
- Understands the request
- Creates task descriptions
- Selects the appropriate agent
- Coordinates handoffs
- Maintains shared state
- Returns the final response

Sub-agents usually do not communicate directly with the user.

This pattern is useful when responsibilities are clearly separated.

The implementation in this repository is available in:

```text
ai_launchpad/langgraph_module/multi_agent/supervisor/
```

## 2. Network Pattern

Agents communicate with each other as peers.

```text
Agent A <----> Agent B
   ^             |
   |             v
Agent D <----> Agent C
```

There may be no single central coordinator.

This pattern is useful when agents need to collaborate dynamically, but it can be harder to control and debug.

## 3. Hierarchical Pattern

Agents are organized in multiple management levels.

```text
Root Supervisor
      |
      +---- Research Supervisor
      |          |
      |          +---- Web Researcher
      |          +---- Document Researcher
      |
      +---- Writing Supervisor
                 |
                 +---- Copywriter
                 +---- Editor
```

This pattern is useful for large and complex systems with multiple teams of agents.

## 4. Custom Pattern

Agents are connected using application-specific routing rules.

Example:

```text
Classifier
   |
   +---- Fraud Agent
   |
   +---- Support Agent
   |
   +---- Escalation Agent
```

LangGraph allows developers to define custom nodes, edges, conditions, commands, and state transitions.

## LangGraph Concepts Used

### State

State stores information shared during graph execution.

Example:

```python
class AgentState(BaseModel):
    messages: list = []
    task_description: str | None = None
    research_reports: list = []
```

State may contain:

- Conversation messages
- Task descriptions
- Research reports
- Tool results
- User preferences
- Intermediate outputs
- Workflow metadata

### Nodes

A node represents a unit of work.

Examples:

- Supervisor node
- Researcher node
- Copywriter node
- Tool node
- Validation node

### Edges

Edges define where execution moves next.

Example:

```text
Supervisor → Researcher
Researcher → Supervisor
Supervisor → Copywriter
Copywriter → Supervisor
```

### Conditional Routing

Conditional edges determine the next node based on the current state.

Example:

```python
if latest_message.tool_calls:
    return "tools"

return END
```

### Tools

Tools allow agents to perform external actions.

Examples:

- Web search
- Webpage extraction
- Document retrieval
- Database queries
- File generation
- API calls

### Subgraphs

A subgraph is a LangGraph workflow used inside another graph.

Example:

```text
Supervisor Graph
    |
    +---- Researcher Subgraph
    |
    +---- Copywriter Subgraph
```

Subgraphs provide:

- Context isolation
- Better modularity
- Independent testing
- Cleaner observability
- Reusable agent workflows

### Command-Based Routing

LangGraph's `Command` primitive can update state and select the next node.

Example:

```python
return Command(
    goto="call_researcher",
    update={
        "task_description": task_description
    },
)
```

This is useful for agent handoffs.

## Module Structure

```text
multi_agent/
├── supervisor/
│   ├── example_content/
│   │   ├── blog.md
│   │   └── linkedin.md
│   ├── prompts/
│   │   ├── copywriter.md
│   │   ├── researcher.md
│   │   └── supervisor.md
│   ├── __init__.py
│   ├── copywriter.py
│   ├── main.py
│   ├── researcher.py
│   ├── supervisor.py
│   └── README.md
└── README.md
```

## Supervisor Example

The current implementation demonstrates a content-creation workflow.

The agents are:

### Supervisor

The supervisor:

- Understands the user request
- Creates focused tasks
- Delegates work
- Coordinates the sub-agents
- Maintains shared research state
- Returns the final response

### Researcher

The researcher:

- Searches the web using Tavily
- Extracts webpage content
- Produces structured research reports
- Returns findings to the supervisor

### Copywriter

The copywriter:

- Reviews research reports
- Generates LinkedIn posts
- Generates blog posts
- Saves generated content as Markdown files

## Example Workflow

```text
User Request
    |
    v
Supervisor
    |
    v
Researcher
    |
    +---- Tavily Search
    +---- Webpage Extraction
    +---- Research Report
    |
    v
Supervisor
    |
    v
Copywriter
    |
    +---- Review Research
    +---- Generate Content
    +---- Save Markdown
    |
    v
Supervisor Final Response
```

## Example Request

```text
Research one practical AI use for a small business and write a LinkedIn post under 100 words.
```

## Technology Stack

- Python 3.13
- LangGraph
- LangChain
- OpenAI API
- Tavily Search API
- Pydantic
- Rich
- python-dotenv
- uv
- Git
- GitHub

## Installation

The repository uses `uv` for dependency management.

From the repository root:

```powershell
uv sync
```

Verify the Python environment:

```powershell
uv run python --version
```

Verify important packages:

```powershell
uv run python -c "import langgraph; import langchain_openai; print('Dependencies installed successfully')"
```

## Environment Variables

Create a `.env` file from the provided example:

```powershell
Copy-Item .env.example .env
```

Add:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ai-launchpad
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Do not commit `.env`.

Verify that Git ignores it:

```powershell
git check-ignore .env
```

Expected result:

```text
.env
```

## Running the Supervisor Example

From the repository root:

```powershell
uv run python -m ai_launchpad.langgraph_module.multi_agent.supervisor.main
```

Enter a request when the terminal displays:

```text
User:
```

To stop the application, enter:

```text
exit
```

or:

```text
quit
```

## Reliability Improvements

The supervisor example includes several improvements to the original implementation.

### File Path Resolution

Prompt and example files are loaded relative to their Python source files instead of the terminal location.

This allows the application to run from the repository root.

### Tavily Response Handling

The researcher supports Tavily responses returned as:

- Dictionaries
- Lists
- JSON strings
- Plain strings

This prevents response-format errors from crashing the entire workflow.

### Asynchronous Agent Calls

The agents use asynchronous model invocation:

```python
await llm_with_tools.ainvoke(...)
```

This is consistent with the asynchronous LangGraph execution model.

### Safe Output Handling

The copywriter:

- Creates the output directory automatically
- Uses UTF-8 encoding
- Removes invalid Windows filename characters
- Safely handles generated content files

### Safer Agent Responses

The supervisor handles:

- Missing tasks
- Empty responses
- Missing reports
- Unexpected sub-agent outputs

### Optional Visualization

Graph visualization code remains available but does not run automatically during normal application execution.

## When Not to Use Multi-Agent Systems

Avoid a multi-agent design when:

- A deterministic workflow solves the problem
- One agent can complete the task reliably
- The task is simple
- API cost is a major constraint
- Low latency is required
- Debugging resources are limited
- The business value is unclear

For example, this process does not require multiple agents:

```text
Upload CSV
→ Validate schema
→ Remove duplicates
→ Save cleaned file
```

A normal Python workflow is simpler and more reliable.

## Production Considerations

Before deploying a multi-agent system, consider:

### Cost

Track:

- Tokens per agent
- Tool-call count
- Model-call count
- Cost per request
- Cost by workflow stage

### Latency

Measure:

- Total execution time
- Time per agent
- Tool latency
- Model latency
- Queue time
- Retry time

### Reliability

Add:

- Retry logic
- Timeouts
- Fallback models
- Input validation
- Output validation
- Circuit breakers
- Maximum handoff limits
- Recursion limits

### Observability

Track:

- Agent decisions
- Tool inputs
- Tool outputs
- State changes
- Errors
- Token usage
- Latency
- Cost
- Final answer quality

LangSmith can be used for tracing and debugging LangGraph workflows.

### Evaluation

Evaluate each component separately.

#### Supervisor Evaluation

Measure:

- Correct agent selection
- Task decomposition quality
- Handoff accuracy
- Completion rate
- Unnecessary handoffs

#### Researcher Evaluation

Measure:

- Search relevance
- Source quality
- Citation accuracy
- Factual coverage
- Retrieval success rate

#### Copywriter Evaluation

Measure:

- Instruction adherence
- Tone consistency
- Word-count compliance
- Factual grounding
- Writing quality

#### End-to-End Evaluation

Measure:

- Task completion
- Accuracy
- Faithfulness
- Latency
- Cost
- User satisfaction

## Common Failure Modes

### Repeated Agent Handoffs

The supervisor may repeatedly call agents without completing the request.

Possible fixes:

- Add a recursion limit
- Limit the number of handoffs
- Strengthen the supervisor prompt
- Store completed-task indicators in state

### Context Overflow

The graph may accumulate too many messages or research reports.

Possible fixes:

- Pass only focused task descriptions
- Summarize old messages
- Store structured outputs
- Remove unnecessary tool messages

### Tool Failure

External search or extraction tools may fail.

Possible fixes:

- Add retries
- Add exception handling
- Validate tool responses
- Provide fallback tools
- Continue with partial results when appropriate

### Conflicting Agent Outputs

Different agents may produce inconsistent conclusions.

Possible fixes:

- Add an evaluator agent
- Add deterministic validation
- Require citations
- Use structured schemas
- Add human review

### High API Cost

Multiple agents may generate many model calls.

Possible fixes:

- Use smaller models for simple tasks
- Reduce unnecessary research
- Limit tool calls
- Cache repeated results
- Use deterministic workflows where possible

## Future Improvements

Potential additions include:

- Streamlit interface
- FastAPI backend
- React frontend
- LangSmith tracing
- Automated evaluation
- Retry logic
- Timeout handling
- Cost tracking
- Token tracking
- Parallel research tasks
- Human approval steps
- Source-quality scoring
- Citation validation
- Structured logging
- Unit tests
- Integration tests
- Docker support
- CI/CD workflows
- Configurable model selection
- Additional LLM providers
- Persistent conversation memory
- Database-backed state
- Content publishing integrations

## Learning Outcomes

This module demonstrates:

- Multi-agent architecture selection
- LangGraph state management
- Node and edge construction
- Conditional routing
- Tool calling
- Agent handoffs
- Command-based routing
- Subgraph orchestration
- Shared state
- Context isolation
- Asynchronous execution
- Web-search integration
- Structured outputs
- Error handling
- Git-based project development

## Detailed Supervisor Documentation

For complete setup instructions, architecture details, troubleshooting, and implementation notes, see:

```text
ai_launchpad/langgraph_module/multi_agent/supervisor/README.md
```

## Attribution

This module is based on the AI Launchpad repository created by Kenneth Liao.

Original repository:

```text
https://github.com/kenneth-liao/ai-launchpad
```

Supervisor example video:

```text
https://www.youtube.com/watch?v=rwqGQEzXF-o
```

The current fork includes additional compatibility fixes, path handling, Tavily response normalization, asynchronous execution improvements, safer file generation, and expanded documentation.

## License

Refer to the license in the repository root for the terms that apply to this project.