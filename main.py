"""
LangGraph 기반 멀티에이전트 블로그 작성 파이프라인
- Agents: Researcher -> Outliner -> Writer -> ImagePrompter -> SEO -> Editor -> (if needed) Reviser
- Tools: Web search(Tavily), Markdown formatter, Simple toxicity check
- Persistence: MemorySaver (in-memory). 실사용 시 SQLiteCheckpointer 등을 사용 권장.

필요 패키지
pip install -U langgraph langchain langchain-openai tavily-python pydantic python-dotenv

환경 변수(.env 또는 쉘)
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...   # 선택 (연구용 검색)

실행 예시
python blog_multia gents.py

메모: LangGraph v0.2+ API 기준
"""
from __future__ import annotations

import os
from typing import TypedDict, Annotated, List, Optional, Literal
from dataclasses import dataclass
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
#from langgraph.types import StreamExecutor
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 선택적: Tavily 검색
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

load_dotenv()

# -----------------------------
# State 정의
# -----------------------------
class BlogState(TypedDict):
    # 입력
    topic: str
    audience: str
    style: str
    language: str
    word_count: int

    # 중간 산출물
    research_notes: str
    outline: str
    draft_markdown: str
    image_prompts: List[str]
    seo: dict

    # 에디터 판단
    needs_revision: bool
    revision_notes: str


# -----------------------------
# 공통 유틸
# -----------------------------

def get_llm(model: str = "gpt-4o", temperature: float = 0.5):
    return ChatOpenAI(model=model, temperature=temperature)


def safe_len(text: Optional[str]) -> int:
    return len(text) if text else 0


# 간단한 독성/안전성 필터(데모용)
BANNED = ["hate", "violence", "terror", "illegal"]

def simple_safety_check(text: str) -> List[str]:
    hits = [w for w in BANNED if w in (text or "").lower()]
    return hits


# -----------------------------
# Tools
# -----------------------------

def build_tools():
    tools: List[Tool] = []

    # 검색 툴 (선택)
    if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearchResults(k=5)
        tools.append(tavily)
    else:
        # 더미 검색 툴
        def dummy_search(q: str):
            return [
                {"title": "Stub Source 1", "url": "https://example.com/1", "content": f"No live search. Query: {q}"},
                {"title": "Stub Source 2", "url": "https://example.com/2", "content": "Provide TAVILY_API_KEY to enable real search."},
            ]
        tools.append(Tool(name="search", func=dummy_search, description="Fallback search tool (returns stubs)."))

    return tools

TOOLS = build_tools()

# 헬퍼: 검색 호출
async def run_search(query: str) -> str:
    for t in TOOLS:
        if getattr(t, "name", None) in ("tavily_search_results", "search"):
            try:
                res = t.run(query)  # langchain Tool 인터페이스
            except TypeError:
                res = t.func(query)
            # 문자열/리스트 모두 처리
            if isinstance(res, str):
                return res
            if isinstance(res, list):
                lines = []
                for i, item in enumerate(res, 1):
                    if isinstance(item, dict):
                        lines.append(f"[{i}] {item.get('title')} - {item.get('url')}\n{item.get('content')}")
                    else:
                        lines.append(f"[{i}] {item}")
                return "\n\n".join(lines)
            return str(res)
    return "(no search tool configured)"


# -----------------------------
# Nodes (에이전트)
# -----------------------------
async def researcher(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.2)
    q = f"Topic: {state['topic']} | Audience: {state['audience']} | Style: {state['style']} | Language: {state['language']}"
    search_block = await run_search(f"In-depth background and latest stats for: {state['topic']}")

    # LLM로 노트 구조화
    sys = (
        "You are a meticulous research assistant. Extract facts, stats (with years), key sources (with URLs), and controversies."
        " Output in bullet points. Keep only verifiable information."
    )
    user = f"Search Results (raw):\n{search_block}\n\nPlease consolidate concise research notes for the blog."
    notes = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content

    return {"research_notes": notes}


async def outliner(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.3)
    sys = "You are a senior content strategist. Create a detailed H1-H3 outline for a blog post."
    user = f"Topic: {state['topic']}\nAudience: {state['audience']}\nStyle: {state['style']}\nTarget length: {state['word_count']} words\nResearch Notes:\n{state.get('research_notes','')}\n\nConstraints:\n- Use clear Korean headings if language is Korean.\n- Include an intro hook and a conclusion with CTA.\n- Add a short TL;DR."
    outline = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content
    return {"outline": outline}


async def writer(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.5)
    sys = (
        "You are a tech blogger. Write in markdown. Keep paragraphs short, include code blocks or tables when relevant."
        " Use fact-checked details from research notes." 
    )
    user = f"Outline:\n{state.get('outline','')}\n\nResearch Notes:\n{state.get('research_notes','')}\n\nWrite the full draft in {state['language']} (~{state['word_count']} words)."
    draft = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content

    # 간단 안전 검사
    hits = simple_safety_check(draft)
    if hits:
        draft += f"\n\n> [auto-note] flagged_terms: {', '.join(hits)} (review needed)"

    return {"draft_markdown": draft}


async def image_prompter(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.6)
    sys = "You are a creative image prompt engineer."
    user = (
        f"Based on the blog topic '{state['topic']}', audience '{state['audience']}', and outline below,\n"
        f"generate 4 DALLE/SDXL-friendly prompts (vivid but concise).\n"
        f"Outline:\n{state.get('outline','')}"
    )
    prompts_text = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content
    # 간단히 줄 단위로 분해
    prompts = [p.strip("- • ") for p in prompts_text.split("\n") if p.strip()][:4]
    return {"image_prompts": prompts}


async def seo_agent(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.3)
    sys = "You are an SEO specialist. Return compact JSON only."
    user = (
        f"Create SEO pack (title<=60 chars, meta<=155 chars, slug, 6-10 keywords, 3-5 FAQs) for the draft below in {state['language']}.\n"
        f"Draft:\n{state.get('draft_markdown','')}"
    )
    resp = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content

    # 매우 관대한 JSON 파서 (실패 시 텍스트로 보관)
    import json, re
    try:
        json_str = re.search(r"\{[\s\S]*\}$", resp).group(0) if "{" in resp else resp
        data = json.loads(json_str)
    except Exception:
        data = {"raw": resp}
    return {"seo": data}


async def editor(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.2)
    sys = (
        "You are a strict editor. Evaluate clarity, accuracy, structure, tone, and SEO alignment."
        " Return JSON with fields: needs_revision(bool), notes(string)."
    )
    user = (
        f"Topic: {state['topic']}\nAudience: {state['audience']}\nStyle: {state['style']}\nDraft:\n{state.get('draft_markdown','')}\n\nSEO:\n{state.get('seo','')}"
    )
    result = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content

    import json, re
    needs = True
    notes = ""
    try:
        json_str = re.search(r"\{[\s\S]*\}$", result).group(0) if "{" in result else result
        data = json.loads(json_str)
        needs = bool(data.get("needs_revision", True))
        notes = data.get("notes", "")
    except Exception:
        notes = result

    return {"needs_revision": needs, "revision_notes": notes}


async def reviser(state: BlogState) -> BlogState:
    llm = get_llm(temperature=0.4)
    sys = "You are a senior reviser. Apply the editor's notes precisely. Output full revised markdown only."
    user = (
        f"Original Draft:\n{state.get('draft_markdown','')}\n\nEditor Notes:\n{state.get('revision_notes','')}\n\nLanguage: {state['language']}"
    )
    revised = (await llm.ainvoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])).content
    return {"draft_markdown": revised, "needs_revision": False}


# -----------------------------
# Graph 구성
# -----------------------------

def build_graph():
    g = StateGraph(BlogState)

    g.add_node("researcher", researcher)
    g.add_node("outliner", outliner)
    g.add_node("writer", writer)
    g.add_node("image_prompter", image_prompter)
    g.add_node("seo", seo_agent)
    g.add_node("editor", editor)
    g.add_node("reviser", reviser)

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "outliner")
    g.add_edge("outliner", "writer")
    g.add_edge("writer", "image_prompter")
    g.add_edge("image_prompter", "seo")
    g.add_edge("seo", "editor")

    # 조건 분기: 수정 필요 시 reviser로, 아니면 종료
    def route_after_edit(state: BlogState) -> Literal["reviser", END]:
        return "reviser" if state.get("needs_revision", True) else END

    g.add_conditional_edges("editor", route_after_edit, {"reviser": "reviser", END: END})
    # reviser 후 종료
    g.add_edge("reviser", END)

    return g


graph = build_graph()
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Visualize graph (optional)
try:
    print("Graph structure:")
    print(app.get_graph().draw_mermaid())
except Exception as e:
    print(f"Visualization not available: {e}")


# -----------------------------
# 실행 유틸
# -----------------------------
DEFAULT_INPUT = {
    "topic": "LangGraph란",
    "audience": "AI/백엔드 엔지니어",
    "style": "전문적이지만 친절한 튜토리얼",
    "language": "Korean",
    "word_count": 1200,
    "research_notes": "",
    "outline": "",
    "draft_markdown": "",
    "image_prompts": [],
    "seo": {},
    "needs_revision": True,
    "revision_notes": "",
}


async def run_once(user_input: dict):
    # 스트리밍으로 단계별 결과 출력
    print("[Start] Mulit-agent blog pipeline\n")
    # Configuration for MemorySaver checkpointer
    config = {"configurable": {"thread_id": "default"}}
    async for event in app.astream(user_input, config=config, stream_mode="updates"):
        for node, update in event.items():
            if node == "__end__":
                continue
            print(f"\n=== {node.upper()} OUTPUT ===")
            # 핵심 필드만 요약 출력
            if update.get("research_notes"):
                print("- research_notes: ", safe_len(update["research_notes"]), "chars")
            if update.get("outline"):
                print("- outline: ", safe_len(update["outline"]), "chars")
            if update.get("draft_markdown"):
                print("- draft_markdown: ", safe_len(update["draft_markdown"]), "chars")
            if update.get("image_prompts"):
                print("- image_prompts: ", update["image_prompts"])            
            if update.get("seo"):
                print("- seo keys: ", list(update["seo"].keys()))
            if update.get("revision_notes"):
                print("- revision_notes: ", update["revision_notes"][:200], "...")
            if update.get("needs_revision") is not None:
                print("- needs_revision: ", update["needs_revision"])    

    final = app.get_state(config).values
    print("\n[Done] Title:", final.get("seo", {}).get("title"))
    print("[Done] Draft chars:", safe_len(final.get("draft_markdown", "")))
    
    # Save draft_markdown to file
    if final.get("draft_markdown"):
        filename = f"blog_draft_{final.get('seo', {}).get('slug', 'untitled')}.md"
        # Clean filename
        import re
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final["draft_markdown"])
        print(f"[Done] Draft saved to: {filename}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_once(DEFAULT_INPUT))
