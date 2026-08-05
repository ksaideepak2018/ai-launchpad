# Multi-Agent Content Creation System with LangGraph

This project implements a multi-agent content creation system using the LangGraph supervisor pattern. A central supervisor agent coordinates specialized researcher and copywriter agents to research a topic and generate content such as LinkedIn posts and blog articles.

This implementation is based on the AI Launchpad project created by Kenneth Liao and includes additional reliability, compatibility, and Windows support improvements.

## Project Overview

The application accepts a content request from the user and automatically coordinates the work required to complete it.

For example, a user can enter:

```text
Research one practical AI use for a small business and write a LinkedIn post under 100 words.
```

The supervisor agent analyzes the request, delegates research to the researcher agent, passes the research findings to the copywriter agent, and returns the completed result.

## Architecture

```text
User
  |
  v
Supervisor Agent
  |
  +----> Researcher Agent
  |         |
  |         +---- Tavily web search
  |         +---- Webpage extraction
  |         +---- Structured research reports
  |
  +----> Copywriter Agent
            |
            +---- Reviews research reports
            +---- Generates LinkedIn posts
            +---- Generates blog posts
            +---- Saves content as Markdown files
  |
  v
Final Supervisor Response
```

## How It Works

1. The user submits a request through the terminal.
2. The supervisor analyzes the request.
3. The supervisor creates a focused task description.
4. The supervisor delegates research tasks to the researcher agent.
5. The researcher searches the web using Tavily.
6. The researcher extracts information from selected webpages.
7. The researcher creates structured research reports.
8. The supervisor passes the research reports to the copywriter.
9. The copywriter generates the requested content.
10. The supervisor returns the final result to the user.

## Why Use the Supervisor Pattern?

The supervisor pattern is useful when a task requires multiple specialized capabilities.

Instead of asking one agent to perform every responsibility, the application separates the work into focused sub-agents.

The supervisor acts like a project manager. It decides:

- Which agent should handle the next task
- What instructions should be given to each agent
- Whether additional research is required
- When the final result is ready

The researcher and copywriter operate as separate LangGraph subgraphs. This provides better context separation, modularity, debugging, testing, and observability.

## Agents

### Supervisor Agent

The supervisor manages the complete workflow and communicates with the user.

Responsibilities:

- Understand the user request
- Break the request into focused tasks
- Route work to the appropriate sub-agent
- Maintain shared graph state
- Store research reports
- Coordinate handoffs between agents
- Return the final response

### Researcher Agent

The researcher gathers and organizes information needed for the writing task.

Responsibilities:

- Search the web using Tavily
- Extract content from selected webpages
- Review search results
- Generate structured research reports
- Return research findings to the supervisor

### Copywriter Agent

The copywriter transforms research findings into polished written content.

Responsibilities:

- Review research reports
- Follow the requested tone and format
- Generate LinkedIn posts
- Generate blog posts
- Save generated content as Markdown files

## Multi-Agent Workflow

```text
User Request
    |
    v
Supervisor
    |
    v
Handoff Tool
    |
    +------ Researcher Subgraph
    |           |
    |           +------ Search Web
    |           +------ Extract Web Content
    |           +------ Generate Research Report
    |
    v
Supervisor
    |
    v
Handoff Tool
    |
    +------ Copywriter Subgraph
                |
                +------ Review Research Reports
                +------ Generate LinkedIn Post
                +------ Generate Blog Post
    |
    v
Supervisor Final Response
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
- Git and GitHub

## Project Structure

```text
supervisor/
├── example_content/
│   ├── blog.md
│   └── linkedin.md
├── prompts/
│   ├── copywriter.md
│   ├── researcher.md
│   └── supervisor.md
├── __init__.py
├── copywriter.py
├── main.py
├── researcher.py
├── supervisor.py
└── README.md
```

### File Responsibilities

#### `main.py`

Provides the terminal interface and starts the supervisor graph.

It:

- Accepts user input
- Creates the initial graph state
- Streams agent and tool responses
- Displays formatted terminal output
- Supports `exit` and `quit` commands

#### `supervisor.py`

Defines the main supervisor graph.

It:

- Interprets user requests
- Creates task descriptions
- Routes tasks to sub-agents
- Stores research reports
- Coordinates the researcher and copywriter
- Returns the final response

#### `researcher.py`

Defines the researcher subgraph.

It:

- Searches the web with Tavily
- Extracts webpage content
- Normalizes Tavily responses
- Produces structured research reports
- Returns reports to the supervisor

#### `copywriter.py`

Defines the copywriter subgraph.

It:

- Reviews available research reports
- Uses writing examples from the repository
- Generates LinkedIn posts
- Generates blog posts
- Saves generated files safely

#### `prompts/`

Contains the system prompts used by the supervisor, researcher, and copywriter agents.

#### `example_content/`

Contains sample LinkedIn and blog content that helps guide the copywriter's tone and formatting.

## Improvements Added

The original implementation was updated with several reliability and compatibility improvements.

### Reliable File Paths

Prompt and example files are now loaded relative to the Python source files instead of the terminal's current working directory.

This allows the application to run from the repository root using:

```powershell
uv run python -m ai_launchpad.langgraph_module.multi_agent.supervisor.main
```

### Tavily Response Normalization

Different Tavily package versions may return results as:

- A Python dictionary
- A Python list
- A JSON-formatted string
- A plain string

The researcher now normalizes these formats before processing them. This prevents errors such as:

```text
TypeError: string indices must be integers, not 'str'
```

### Asynchronous Model Calls

The agents now use:

```python
await llm_with_tools.ainvoke(...)
```

instead of synchronous model calls inside asynchronous functions.

This makes the implementation more consistent with the asynchronous LangGraph workflow.

### Safer Sub-Agent Handling

The supervisor safely handles:

- Missing task descriptions
- Missing agent responses
- Empty research results
- Unexpected sub-agent outputs

### Automatic Output Directory Creation

The copywriter automatically creates the output directory when it does not exist.

### Windows-Safe Filenames

Generated titles are converted into safe filenames by removing characters that Windows does not allow.

Examples of removed characters include:

```text
< > : " / \ | ? *
```

### UTF-8 File Handling

Prompt, example, and generated Markdown files use UTF-8 encoding.

This improves support for emojis and international characters.

### Graph Visualization Disabled During Runtime

Graph visualization code is kept as an optional feature and no longer runs automatically when the application starts.

## Prerequisites

Before running the project, install:

- Python 3.13 or later
- Git
- VS Code or another code editor
- uv
- An OpenAI API key
- A Tavily API key

The OpenAI API account must have an available API balance.

A ChatGPT Plus subscription does not include OpenAI API usage.

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/ksaideepak2018/ai-launchpad.git
cd ai-launchpad
```

### 2. Install Dependencies

```powershell
uv sync
```

This command creates a project virtual environment and installs the dependencies declared in `pyproject.toml`.

### 3. Verify Python

```powershell
uv run python --version
```

Expected output:

```text
Python 3.13.x
```

### 4. Verify Important Packages

```powershell
uv run python -c "import langgraph; import langchain_openai; print('Dependencies installed successfully')"
```

Expected output:

```text
Dependencies installed successfully
```

## Environment Variables

Create a local `.env` file from the provided template.

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Add your credentials:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ai-launchpad
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

LangSmith is optional and can remain disabled during initial testing.

## API Key Security

Never commit the `.env` file.

Confirm that Git ignores it:

```powershell
git check-ignore .env
```

Expected output:

```text
.env
```

Also verify that `.env` does not appear in:

```powershell
git status
```

Never include API keys in:

- Git commits
- GitHub repositories
- Screenshots
- Documentation
- Chat messages
- Source-code files

## Running the Application

Run the application from the repository root:

```powershell
uv run python -m ai_launchpad.langgraph_module.multi_agent.supervisor.main
```

The application displays an interactive terminal prompt:

```text
User:
```

Enter a request such as:

```text
Research one practical AI use for a small business and write a LinkedIn post under 100 words.
```

The application will display:

- Supervisor decisions
- Sub-agent handoffs
- Tool calls
- Research output
- Copywriter output
- Final supervisor response

## Stopping the Application

At the `User:` prompt, enter:

```text
exit
```

or:

```text
quit
```

If the application is still processing and must be interrupted, press:

```text
Ctrl + C
```

## Example Request

```text
Research one practical AI use for a small business and write a LinkedIn post under 100 words.
```

## Example Output

```text
AI chatbots can handle routine customer questions instantly, reducing response times and allowing support teams to focus on complex issues.

Start with your three most common customer questions, connect the chatbot to a verified knowledge base, and create a clear handoff process for requests that require a human.

The goal is not to replace customer support. It is to make support faster, more consistent, and available when customers need it.
```

## Generated Files

When the copywriter uses a content-generation tool, the output is saved as a Markdown file in:

```text
ai_files/
```

The application automatically creates this directory if it does not already exist.

Generated filenames are sanitized so they work correctly on Windows.

## Common Errors

### OpenAI Credit Balance Error

```text
429 credit_balance_exhausted
```

This means:

- The API key was recognized
- Authentication succeeded
- The OpenAI API account has no remaining credits

Add API credits through the OpenAI Platform billing settings and run the application again.

### Invalid API Key

```text
401 invalid_api_key
```

Check that:

- The correct key was added to `.env`
- There are no quotation marks around the key
- There are no extra spaces
- The `.env` file was saved
- The key has not been deleted or revoked

### Prompt File Not Found

An older version of the application could produce:

```text
FileNotFoundError: prompts/researcher.md
```

The updated implementation resolves prompt and example paths relative to each Python file, allowing the project to run from the repository root.

### Tavily Extraction Error

An older version could produce:

```text
TypeError: string indices must be integers, not 'str'
```

The updated researcher handles Tavily responses returned as dictionaries, lists, JSON strings, or plain strings.

### Long Execution Time

Multi-agent systems make several model and tool calls.

A single request may involve:

```text
Supervisor model call
→ Researcher model call
→ Tavily searches
→ Webpage extraction
→ Research report generation
→ Additional supervisor decision
→ Copywriter model call
→ Final supervisor response
```

Use a small, focused prompt when testing to reduce execution time and API cost.

## Testing

A basic manual test prompt is:

```text
Research one practical AI use for a small business and write a LinkedIn post under 100 words.
```

A successful run should show:

```text
Supervisor
→ Researcher
→ Research tools
→ Research report
→ Supervisor
→ Copywriter
→ Supervisor final response
```

## Git Workflow

Check modified files:

```powershell
git status
```

Review the change summary:

```powershell
git diff --stat
```

Check formatting:

```powershell
git diff --check
```

Stage changes:

```powershell
git add ai_launchpad/langgraph_module/multi_agent/supervisor/
```

Create a commit:

```powershell
git commit -m "Improve multi-agent supervisor implementation"
```

Push changes:

```powershell
git push origin main
```

## Limitations

This project is intended as a learning and demonstration application.

Current limitations include:

- API usage costs
- Reliance on external APIs
- Limited retry handling
- No graphical user interface
- No automated evaluation framework
- No automated source-quality validation
- No human approval step before saving content
- Research quality depends on retrieved sources
- Multi-agent workflows may be slower than simpler workflows

## Future Improvements

Potential enhancements include:

- Streamlit user interface
- React frontend
- FastAPI backend
- LangSmith tracing
- Retry logic with exponential backoff
- API timeout handling
- Token and cost tracking
- Parallel research tasks
- Human approval before publishing
- Source-quality scoring
- Citation validation
- Content moderation
- Structured logging
- Unit tests
- Integration tests
- Docker support
- CI/CD workflow
- Configurable model selection
- Support for additional LLM providers
- Database-backed conversation history
- User authentication
- Export to PDF or DOCX
- Social-media publishing integrations

## Key Learning Outcomes

This project demonstrates:

- LangGraph state management
- Supervisor-based multi-agent architecture
- Subgraph orchestration
- Agent handoffs using `Command`
- Tool calling
- Shared state between agents
- Web search integration
- Webpage content extraction
- Structured research reports
- Asynchronous LLM invocation
- Environment-variable management
- Secure API-key handling
- Error debugging
- Git version control
- GitHub repository management

## Attribution

This project is based on the AI Launchpad repository created by Kenneth Liao.

Original repository:

```text
https://github.com/kenneth-liao/ai-launchpad
```

Original video:

```text
https://www.youtube.com/watch?v=rwqGQEzXF-o
```

This fork includes additional debugging fixes, compatibility improvements, safer file handling, and documentation updates.

## License

Refer to the root repository license for the terms that apply to this project.