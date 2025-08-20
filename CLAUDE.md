# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based multi-agent blog generator that creates Naver-optimized Korean blog posts from web content. The application uses LangGraph to orchestrate 4 specialized AI agents that work sequentially to analyze content, optimize for SEO, write blog posts, and generate images.

## Architecture

The pipeline consists of 4 agents connected in a sequential StateGraph:
1. **Researcher** (`researcher_node`) → Scrapes content from provided URL using BeautifulSoup
2. **SEO Specialist** (`seo_specialist_node`) → Analyzes content and generates Naver SEO tags using Tavily search
3. **Writer** (`writer_node`) → Creates Korean blog post in markdown format with emojis and SEO optimization
4. **Art Director** (`art_director_node`) → Generates DALL-E 3 images based on blog content

State flows through `AgentState` TypedDict containing URL, scraped content, SEO analysis, draft post, titles, and image data.

## Dependencies & Environment

**Required Python version**: >=3.12

**Key dependencies**:
- `streamlit` - Web UI framework
- `langgraph>=0.6.5` - Multi-agent orchestration framework
- `langchain>=0.3.27` + `langchain-openai>=0.3.30` - LLM integration  
- `tavily-python>=0.7.11` - Web search API for SEO research
- `beautifulsoup4` + `requests` - Web scraping
- `python-dotenv>=1.1.1` - Environment variable management
- `pydantic>=2.11.7` - Data validation
- `openai` - DALL-E 3 image generation

**Environment variables** (create `.env` file):
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=...  # Optional for LangSmith tracing
```

## Common Commands

**Install dependencies**:
```bash
uv sync
```

**Run the Streamlit application**:
```bash
uv run streamlit run main.py
```

**Alternative run command**:
```bash
uv run main.py
```

**Add new dependencies**:
```bash
uv add <package>
```

## Key Implementation Details

- **Streamlit UI**: Web interface for URL input and result display, runs on default port (8501)
- **Sequential execution**: Agents run synchronously using `app.invoke()`, not streaming
- **Web scraping**: Custom `scrape_web_content()` function extracts main content from URLs
- **Korean language focus**: All prompts and outputs are in Korean for Naver blog optimization
- **SEO optimization**: Searches for latest Naver SEO trends and generates 10 optimized tags
- **Image generation**: Uses DALL-E 3 with custom prompt generation for blog thumbnails
- **Error handling**: Includes try-catch blocks for web scraping and image generation failures
- **LangSmith integration**: Optional tracing enabled via environment variables (main.py:19-21)
- **Model configuration**: Uses GPT-4o with temperature 0.7 for all text generation

## Application Workflow

1. User inputs URL in Streamlit interface
2. Researcher scrapes and cleans web content 
3. SEO Specialist searches Naver trends and generates Korean SEO strategy + tags
4. Writer creates markdown blog post with title, sections, and emojis
5. Art Director generates DALL-E prompt and creates 1024x1024 image
6. Results displayed: image, title, tags, and full markdown post

## State Schema

The `AgentState` TypedDict includes:
- `url`: Input URL to analyze
- `scraped_content`: Cleaned text from web scraping
- `seo_analysis`: SEO strategy and keyword analysis
- `seo_tags`: List of 10 Korean SEO tags
- `draft_post`: Complete markdown blog post
- `final_title`: Extracted H1 title
- `final_subheadings`: List of H2 subheadings
- `image_prompt`: Generated DALL-E prompt
- `image_url`: Generated image URL
- `messages`: BaseMessage list for agent communication