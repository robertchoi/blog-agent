# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based multi-agent blog writing pipeline that automates blog post creation through a series of specialized AI agents. The application uses Korean as the primary language and creates technical blog posts.

## Architecture

The pipeline consists of 7 agents connected in a StateGraph:
1. **Researcher** → Searches and compiles research notes using Tavily API
2. **Outliner** → Creates structured H1-H3 outline from research
3. **Writer** → Generates markdown draft from outline and research
4. **ImagePrompter** → Creates DALL-E/SDXL image prompts
5. **SEO Agent** → Generates SEO metadata (title, description, keywords, FAQs)
6. **Editor** → Evaluates quality and determines if revision needed
7. **Reviser** → (conditional) Applies editor feedback if revision required

State flows through `BlogState` TypedDict containing topic, audience, style, language, word_count and intermediate outputs.

## Dependencies & Environment

**Required Python version**: >=3.12

**Key dependencies**:
- `langgraph>=0.6.5` - Multi-agent orchestration framework
- `langchain>=0.3.27` + `langchain-openai>=0.3.30` - LLM integration
- `tavily-python>=0.7.11` - Web search API (optional)
- `pydantic>=2.11.7` - Data validation
- `python-dotenv>=1.1.1` - Environment variable management

**Environment variables** (create `.env` file):
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...  # Optional for web search
```

## Common Commands

**Install dependencies**:
```bash
uv sync
```

**Run the application**:
```bash
uv run main.py
```

**Development setup**:
```bash
uv add <package>  # Add new dependency
```

## Key Implementation Details

- Uses `MemorySaver` checkpointer for in-memory state persistence (switch to `SQLiteCheckpointer` for production)
- **Important**: The checkpointer requires `config` parameter with `configurable.thread_id` passed to `astream()` (handled in `run_once()`)
- Implements conditional flow: Editor determines if revision needed via `needs_revision` boolean
- Includes basic safety filtering with banned words check
- Fallback dummy search tool when Tavily API unavailable
- All agents are async functions that receive and return `BlogState`
- Uses streaming execution with `astream()` for real-time progress updates

## Default Configuration

The application runs with these defaults (modify `DEFAULT_INPUT` in main.py):
- Topic: LangGraph multi-agent blog automation
- Audience: AI/Backend Engineers  
- Style: Professional but friendly tutorial
- Language: Korean
- Target length: 1200 words