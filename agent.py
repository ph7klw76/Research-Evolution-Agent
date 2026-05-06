import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv
from ollama import chat
from rich import print


DB_PATH = "memory.db"
PROFILE_PATH = "research_profile.yaml"
REPORT_DIR = "reports"

# Default is intentionally Qwen3:8b because it is a strong local tool/planning model in Ollama.
# Override with: OLLAMA_MODEL=qwen3:8b
DEFAULT_MODEL = "qwen3:8b"

AUTONOMY_POLICY = {
    "search_papers": "auto",
    "score_papers": "auto",
    "generate_brief": "auto",
    "generate_gap_analysis": "auto",
    "draft_grant_skeleton": "auto",
    "draft_email": "draft_only",
    "update_memory": "auto",
    "evolve_profile": "ask",
    "send_email": "never",
    "submit_grant": "never",
    "delete_files": "never",
    "financial_decision": "never",
}

AGENT_ROLES = {
    "strategic_governor": "Decide priority, mission fit, risk, and next actions.",
    "research_scout": "Search and interpret papers as research opportunities, not summaries.",
    "grant_architect": "Convert research opportunities into fundable proposal logic.",
    "scientific_verifier": "Attack weak claims, unsupported novelty, feasibility gaps, and evidence gaps.",
    "communication_engine": "Convert outputs into teaching, LinkedIn, public, and collaboration drafts.",
    "self_evolution_engine": "Learn from feedback and update memory, rubrics, and workflow rules safely.",
}


# -----------------------------
# Core utilities
# -----------------------------

def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def today() -> str:
    return dt.date.today().isoformat()


def ensure_dirs() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)


def load_settings() -> Dict[str, str]:
    load_dotenv()
    return {
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "openalex_api_key": os.getenv("OPENALEX_API_KEY", "").strip(),
        "temperature": os.getenv("OLLAMA_TEMPERATURE", "0.15").strip(),
        "num_ctx": os.getenv("OLLAMA_NUM_CTX", "8192").strip(),
    }


def load_profile() -> Dict[str, Any]:
    if not os.path.exists(PROFILE_PATH):
        raise FileNotFoundError(
            f"Missing {PROFILE_PATH}. Create it before running the agent."
        )
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_profile(profile: Dict[str, Any]) -> None:
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clamp_score(value: Any, default: float = 5.0) -> float:
    try:
        score = float(value)
    except Exception:
        score = default
    return max(0.0, min(10.0, score))


def get_ollama_content(response: Any) -> str:
    try:
        return response.message.content
    except AttributeError:
        return response["message"]["content"]


def strip_thinking(text: str) -> str:
    """Qwen3 can emit <think>...</think>. Remove it for stable downstream parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def clean_json(text: str) -> Dict[str, Any]:
    text = strip_thinking(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def clean_yaml(text: str) -> str:
    text = strip_thinking(text).strip()
    fenced = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    json_mode: bool = False,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
) -> str:
    settings = load_settings()
    options = {
        "temperature": temperature if temperature is not None else float(settings["temperature"]),
        "num_ctx": num_ctx if num_ctx is not None else int(settings["num_ctx"]),
    }
    kwargs: Dict[str, Any] = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
    }
    if json_mode:
        kwargs["format"] = "json"
    response = chat(**kwargs)
    return strip_thinking(get_ollama_content(response))


# -----------------------------
# Database and memory
# -----------------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_key TEXT UNIQUE,
            source TEXT,
            title TEXT,
            abstract TEXT,
            url TEXT,
            published_date TEXT,
            authors TEXT,
            cited_by_count INTEGER DEFAULT 0,
            relevance REAL,
            novelty REAL,
            grant_potential REAL,
            collaboration_potential REAL,
            industry_potential REAL,
            teaching_public_value REAL,
            feasibility REAL DEFAULT 5,
            risk REAL DEFAULT 5,
            total_score REAL,
            agent_commentary TEXT,
            evidence_gaps TEXT,
            recommended_action TEXT,
            status TEXT DEFAULT 'new',
            created_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_text TEXT,
            created_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT,
            content TEXT,
            source TEXT,
            confidence REAL,
            tags TEXT,
            created_at TEXT,
            review_after TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            rationale TEXT,
            agent TEXT,
            priority INTEGER,
            effort INTEGER,
            risk INTEGER,
            status TEXT DEFAULT 'open',
            created_at TEXT,
            completed_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow TEXT,
            user_input TEXT,
            result_summary TEXT,
            lesson TEXT,
            memory_update TEXT,
            created_at TEXT
        )
        """
    )

    # Lightweight migrations for existing DBs.
    for column, definition in [
        ("feasibility", "REAL DEFAULT 5"),
        ("risk", "REAL DEFAULT 5"),
        ("evidence_gaps", "TEXT"),
    ]:
        try:
            cur.execute(f"ALTER TABLE papers ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


def save_memory(
    memory_type: str,
    content: str,
    source: str,
    confidence: float = 0.7,
    tags: Optional[List[str]] = None,
    review_after: Optional[str] = None,
) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories (memory_type, content, source, confidence, tags, created_at, review_after)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_type,
            content.strip(),
            source,
            clamp_score(confidence * 10) / 10,
            json.dumps(tags or [], ensure_ascii=False),
            now(),
            review_after,
        ),
    )
    conn.commit()
    conn.close()


def load_recent_memories(limit: int = 10, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if memory_type:
        cur.execute(
            "SELECT memory_type, content, source, confidence, tags, created_at FROM memories WHERE memory_type=? ORDER BY id DESC LIMIT ?",
            (memory_type, limit),
        )
    else:
        cur.execute(
            "SELECT memory_type, content, source, confidence, tags, created_at FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "memory_type": r[0],
            "content": r[1],
            "source": r[2],
            "confidence": r[3],
            "tags": json.loads(r[4] or "[]"),
            "created_at": r[5],
        }
        for r in rows
    ]


def format_memories(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return "No relevant memories yet."
    return "\n".join(
        f"- [{m['memory_type']}; confidence={m['confidence']}] {m['content']} (source={m['source']})"
        for m in memories
    )


def add_task(title: str, rationale: str, agent: str, priority: int, effort: int, risk: int) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tasks (title, rationale, agent, priority, effort, risk, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (title, rationale, agent, int(priority), int(effort), int(risk), now()),
    )
    conn.commit()
    conn.close()


# -----------------------------
# Paper search tools
# -----------------------------

def reconstruct_openalex_abstract(inv_index: Optional[Dict[str, List[int]]]) -> str:
    if not inv_index:
        return ""
    positioned_words: List[Tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            positioned_words.append((pos, word))
    positioned_words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positioned_words)


def search_openalex(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    settings = load_settings()
    params = {"search": query, "per-page": max_results, "sort": "publication_date:desc"}
    if settings["openalex_api_key"]:
        params["api_key"] = settings["openalex_api_key"]
    response = requests.get("https://api.openalex.org/works", params=params, timeout=30)
    response.raise_for_status()
    papers = []
    for item in response.json().get("results", []):
        primary_location = item.get("primary_location") or {}
        authors = []
        for a in (item.get("authorships") or [])[:8]:
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        papers.append(
            {
                "source": "OpenAlex",
                "title": (item.get("title") or item.get("display_name") or "").strip(),
                "abstract": reconstruct_openalex_abstract(item.get("abstract_inverted_index")).strip(),
                "url": item.get("doi") or primary_location.get("landing_page_url") or item.get("id") or "",
                "published_date": item.get("publication_date", ""),
                "authors": ", ".join(authors),
                "cited_by_count": item.get("cited_by_count", 0),
            }
        )
    time.sleep(0.2)
    return papers


def arxiv_query_from_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return "all:OLED"
    return f'all:"{topic}"' if " " in topic else f"all:{topic}"


def search_arxiv(topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
    params = {
        "search_query": arxiv_query_from_topic(topic),
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get("https://export.arxiv.org/api/query", params=params, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]
        papers.append(
            {
                "source": "arXiv",
                "title": " ".join(entry.findtext("atom:title", default="", namespaces=ns).split()),
                "abstract": " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split()),
                "url": entry.findtext("atom:id", default="", namespaces=ns),
                "published_date": entry.findtext("atom:published", default="", namespaces=ns)[:10],
                "authors": ", ".join([a for a in authors if a][:8]),
                "cited_by_count": 0,
            }
        )
    time.sleep(3)
    return papers


def make_unique_key(paper: Dict[str, Any]) -> str:
    base = paper.get("url") or paper.get("title") or ""
    return re.sub(r"\s+", " ", base.lower()).strip()


# -----------------------------
# Agentic reasoning modules
# -----------------------------

def mission_context(profile: Dict[str, Any]) -> str:
    return yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)


def strategic_governor(user_input: str, memories: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    profile = load_profile()
    system = f"""
You are the Strategic Governor for a bounded self-evolving research agent.
Your job is to choose the highest-leverage next move, not merely answer.
Be evidence-grounded, realistic, and mission-aligned.
Never authorize external irreversible actions.
"""
    prompt = f"""
Research profile:
{mission_context(profile)}

Recent memories:
{format_memories(memories or load_recent_memories(12))}

User input:
{user_input}

Return only valid JSON:
{{
  "strategic_classification": "research|grant|collaboration|teaching|communication|memory|mixed",
  "mission_fit": 0,
  "priority": 0,
  "risk": 0,
  "why_it_matters": "...",
  "recommended_workflow": "paper_to_opportunity|idea_to_grant|weekly_brief|gap_analysis|reviewer_simulation|reflection|custom",
  "next_actions": ["action 1", "action 2", "action 3"],
  "agents_to_run": ["research_scout", "grant_architect", "scientific_verifier"],
  "approval_required": false
}}
"""
    try:
        data = clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.1))
    except Exception:
        data = {
            "strategic_classification": "mixed",
            "mission_fit": 5,
            "priority": 5,
            "risk": 5,
            "why_it_matters": "Could not parse governor output; proceed cautiously.",
            "recommended_workflow": "custom",
            "next_actions": ["Review input manually", "Run a conservative analysis", "Save feedback"],
            "agents_to_run": ["research_scout", "scientific_verifier"],
            "approval_required": False,
        }
    return data


def research_scout(user_input: str) -> Dict[str, Any]:
    profile = load_profile()
    system = "You are a Research Scout. Convert information into opportunity maps. Distinguish evidence, assumptions, and speculation."
    prompt = f"""
Profile:
{mission_context(profile)}

Recent memories:
{format_memories(load_recent_memories(12))}

Input:
{user_input}

Return only valid JSON:
{{
  "research_opportunity": "...",
  "scientific_gap": "...",
  "novelty_hypothesis": "...",
  "evidence_strength": 0,
  "key_uncertainties": ["..."],
  "possible_paper_angle": "...",
  "possible_grant_angle": "...",
  "next_research_actions": ["..."],
  "search_queries": ["..."]
}}
"""
    return clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.15))


def grant_architect(user_input: str, scout_output: Dict[str, Any]) -> Dict[str, Any]:
    profile = load_profile()
    system = "You are a Grant Architect. Convert ideas into fundable, reviewable proposal logic. Do not invent citations."
    prompt = f"""
Profile:
{mission_context(profile)}

Original input:
{user_input}

Research Scout output:
{json.dumps(scout_output, indent=2, ensure_ascii=False)}

Return only valid JSON:
{{
  "possible_title": "...",
  "problem_statement": "...",
  "central_hypothesis": "...",
  "objectives": ["Objective 1", "Objective 2", "Objective 3"],
  "methodology_overview": "...",
  "work_packages": ["..."],
  "expected_outcomes": ["..."],
  "risk_mitigation": ["..."],
  "reviewer_attack_points": ["..."],
  "collaborator_needs": ["..."],
  "grant_fit_score": 0
}}
"""
    return clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.15))


def scientific_verifier(user_input: str, scout_output: Dict[str, Any], grant_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    system = "You are a Scientific Verifier. Be skeptical. Separate supported claims from attractive but weak claims."
    prompt = f"""
Original input:
{user_input}

Research Scout output:
{json.dumps(scout_output, indent=2, ensure_ascii=False)}

Grant Architect output:
{json.dumps(grant_output or {}, indent=2, ensure_ascii=False)}

Return only valid JSON:
{{
  "strongest_claim": "...",
  "weakest_claim": "...",
  "unsupported_assumptions": ["..."],
  "missing_evidence": ["..."],
  "feasibility_score": 0,
  "novelty_score": 0,
  "fundability_score": 0,
  "risk_score": 0,
  "concrete_improvements_needed": ["..."],
  "final_recommendation": "proceed|revise|park|reject"
}}
"""
    return clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.1))


def communication_engine(user_input: str, scout_output: Dict[str, Any], verifier_output: Dict[str, Any]) -> Dict[str, Any]:
    system = "You are a Teaching and Communication Engine. Make science clear without overstating evidence."
    prompt = f"""
Input:
{user_input}

Scout:
{json.dumps(scout_output, indent=2, ensure_ascii=False)}

Verifier:
{json.dumps(verifier_output, indent=2, ensure_ascii=False)}

Return only valid JSON:
{{
  "teaching_angle": "...",
  "linkedin_post_draft": "...",
  "collaboration_email_draft": "...",
  "public_explanation": "...",
  "caution_statement": "..."
}}
"""
    return clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.25))


def save_reflection(workflow: str, user_input: str, outputs: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
    system = "You are the Self-Evolution Engine. Extract durable lessons, not every detail."
    prompt = f"""
Workflow: {workflow}
User input: {user_input}
Outputs:
{json.dumps(outputs, indent=2, ensure_ascii=False)}
User feedback: {feedback or 'No explicit feedback.'}

Return only valid JSON:
{{
  "result_summary": "...",
  "lesson": "...",
  "memory_update": "...",
  "tags": ["..."],
  "confidence": 0.0,
  "should_store_memory": true
}}
"""
    try:
        reflection = clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.1))
    except Exception:
        reflection = {
            "result_summary": "Workflow completed, but reflection parsing failed.",
            "lesson": "Keep outputs parseable and conservative.",
            "memory_update": "Parsing failure occurred during reflection.",
            "tags": ["self-evolution", "parse-failure"],
            "confidence": 0.5,
            "should_store_memory": True,
        }

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reflections (workflow, user_input, result_summary, lesson, memory_update, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            workflow,
            user_input,
            reflection.get("result_summary", ""),
            reflection.get("lesson", ""),
            reflection.get("memory_update", ""),
            now(),
        ),
    )
    conn.commit()
    conn.close()

    if reflection.get("should_store_memory", True):
        save_memory(
            "workflow_lesson",
            reflection.get("memory_update", reflection.get("lesson", "")),
            source=f"reflection:{workflow}",
            confidence=float(reflection.get("confidence", 0.7) or 0.7),
            tags=reflection.get("tags", []),
        )
    return reflection


def run_agentic_workflow(user_input: str, include_communication: bool = True) -> Dict[str, Any]:
    init_db()
    governor = strategic_governor(user_input)
    scout = research_scout(user_input)
    grant = grant_architect(user_input, scout)
    verifier = scientific_verifier(user_input, scout, grant)
    communication = communication_engine(user_input, scout, verifier) if include_communication else {}

    outputs = {
        "strategic_governor": governor,
        "research_scout": scout,
        "grant_architect": grant,
        "scientific_verifier": verifier,
        "communication_engine": communication,
    }

    for action in governor.get("next_actions", [])[:5]:
        add_task(
            title=str(action)[:120],
            rationale=governor.get("why_it_matters", ""),
            agent="strategic_governor",
            priority=int(clamp_score(governor.get("priority", 5))),
            effort=5,
            risk=int(clamp_score(governor.get("risk", 5))),
        )

    outputs["reflection"] = save_reflection("agentic_workflow", user_input, outputs)
    return outputs


# -----------------------------
# Scoring and persistence
# -----------------------------

def score_paper(profile: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    system = "You are a bounded Research Evolution Agent using Qwen3. Score papers as strategic opportunities. Return strict JSON."
    prompt = f"""
Research profile:
{mission_context(profile)}

Negative filters:
{profile.get('negative_filters', [])}

Paper:
Title: {paper.get('title', '')}
Authors: {paper.get('authors', '')}
Source: {paper.get('source', '')}
Published: {paper.get('published_date', '')}
Citations: {paper.get('cited_by_count', 0)}
Abstract: {paper.get('abstract', '')}

Score 0-10. Be skeptical. Reward fundable novelty, device relevance, collaboration potential, and clear next action.
Return only valid JSON:
{{
  "relevance": 0,
  "novelty": 0,
  "grant_potential": 0,
  "collaboration_potential": 0,
  "industry_potential": 0,
  "teaching_public_value": 0,
  "feasibility": 0,
  "risk": 0,
  "commentary": "short critical explanation",
  "evidence_gaps": ["missing evidence 1"],
  "recommended_action": "read_now|save_for_later|use_for_grant|contact_author|use_for_teaching|use_for_linkedin|ignore"
}}
"""
    try:
        data = clean_json(ask_llm(system, prompt, json_mode=True, temperature=0.1, num_ctx=8192))
    except Exception:
        data = {
            "relevance": 5,
            "novelty": 5,
            "grant_potential": 5,
            "collaboration_potential": 5,
            "industry_potential": 5,
            "teaching_public_value": 5,
            "feasibility": 5,
            "risk": 5,
            "commentary": "LLM output could not be parsed. Manual review required.",
            "evidence_gaps": ["Could not parse model evidence assessment."],
            "recommended_action": "save_for_later",
        }
    for key in [
        "relevance", "novelty", "grant_potential", "collaboration_potential",
        "industry_potential", "teaching_public_value", "feasibility", "risk",
    ]:
        data[key] = clamp_score(data.get(key, 5))
    data["commentary"] = str(data.get("commentary", "")).strip()
    data["recommended_action"] = str(data.get("recommended_action", "save_for_later")).strip()
    if not isinstance(data.get("evidence_gaps"), list):
        data["evidence_gaps"] = [str(data.get("evidence_gaps", ""))]
    return data


def calculate_total(profile: Dict[str, Any], scores: Dict[str, Any]) -> float:
    w = profile.get("weights", {})
    total = (
        float(w.get("relevance", 0.25)) * scores["relevance"]
        + float(w.get("novelty", 0.20)) * scores["novelty"]
        + float(w.get("grant_potential", 0.20)) * scores["grant_potential"]
        + float(w.get("collaboration_potential", 0.15)) * scores["collaboration_potential"]
        + float(w.get("industry_potential", 0.10)) * scores["industry_potential"]
        + float(w.get("teaching_public_value", 0.10)) * scores["teaching_public_value"]
    )
    # Penalize high risk unless feasibility is also high.
    total += 0.05 * scores.get("feasibility", 5) - 0.05 * scores.get("risk", 5)
    return round(max(0.0, min(10.0, total)), 2)


def save_scored_paper(paper: Dict[str, Any], scores: Dict[str, Any], total_score: float) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO papers (
            unique_key, source, title, abstract, url, published_date, authors, cited_by_count,
            relevance, novelty, grant_potential, collaboration_potential, industry_potential,
            teaching_public_value, feasibility, risk, total_score, agent_commentary,
            evidence_gaps, recommended_action, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            make_unique_key(paper), paper.get("source", ""), paper.get("title", ""),
            paper.get("abstract", ""), paper.get("url", ""), paper.get("published_date", ""),
            paper.get("authors", ""), int(paper.get("cited_by_count", 0) or 0),
            scores["relevance"], scores["novelty"], scores["grant_potential"],
            scores["collaboration_potential"], scores["industry_potential"], scores["teaching_public_value"],
            scores.get("feasibility", 5), scores.get("risk", 5), total_score,
            scores.get("commentary", ""), json.dumps(scores.get("evidence_gaps", []), ensure_ascii=False),
            scores.get("recommended_action", "save_for_later"), now(),
        ),
    )
    conn.commit()
    conn.close()


def collect_and_score(max_per_topic: int = 3) -> None:
    init_db()
    profile = load_profile()
    all_papers: List[Dict[str, Any]] = []

    for topic in profile.get("topics", []):
        print(f"[bold blue]Searching OpenAlex:[/bold blue] {topic}")
        try:
            all_papers.extend(search_openalex(topic, max_results=max_per_topic))
        except Exception as e:
            print(f"[red]OpenAlex failed for {topic}: {e}[/red]")

        print(f"[bold blue]Searching arXiv:[/bold blue] {topic}")
        try:
            all_papers.extend(search_arxiv(topic, max_results=max_per_topic))
        except Exception as e:
            print(f"[red]arXiv failed for {topic}: {e}[/red]")

    seen, unique_papers = set(), []
    for paper in all_papers:
        key = make_unique_key(paper)
        if not paper.get("title") or key in seen:
            continue
        seen.add(key)
        unique_papers.append(paper)

    print(f"[bold green]Found {len(unique_papers)} unique papers.[/bold green]")
    for paper in unique_papers:
        print(f"[bold yellow]Scoring:[/bold yellow] {paper['title'][:100]}")
        scores = score_paper(profile, paper)
        total = calculate_total(profile, scores)
        save_scored_paper(paper, scores, total)
        print(f"  Score={total} | Action={scores['recommended_action']} | Risk={scores.get('risk', 5)}")


def get_top_papers(limit: int = 12) -> List[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT title, abstract, url, published_date, authors, source, cited_by_count,
               relevance, novelty, grant_potential, collaboration_potential,
               industry_potential, teaching_public_value, feasibility, risk, total_score,
               agent_commentary, evidence_gaps, recommended_action
        FROM papers
        ORDER BY total_score DESC, published_date DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    keys = [
        "title", "abstract", "url", "published_date", "authors", "source", "cited_by_count",
        "relevance", "novelty", "grant_potential", "collaboration_potential", "industry_potential",
        "teaching_public_value", "feasibility", "risk", "total_score", "agent_commentary",
        "evidence_gaps", "recommended_action",
    ]
    return [dict(zip(keys, row)) for row in rows]


# -----------------------------
# Reports and evolution
# -----------------------------

def generate_weekly_brief(limit: int = 12) -> str:
    ensure_dirs()
    profile = load_profile()
    papers = get_top_papers(limit=limit)
    if not papers:
        return "No papers found yet. Run: python agent.py run"
    paper_text = "\n\n".join(json.dumps(p, indent=2, ensure_ascii=False) for p in papers)
    system = "You are a strategic research operating system. Produce actionable weekly intelligence."
    prompt = f"""
Profile:
{mission_context(profile)}

Recent memories:
{format_memories(load_recent_memories(20))}

Top papers:
{paper_text}

Create a Weekly Research Evolution Brief with:
1. Executive Summary
2. Top 5 opportunities
3. Best research gap
4. Grant angle
5. Collaboration angle
6. Teaching/LinkedIn angle
7. Risks and missing evidence
8. Top 5 high-leverage tasks for this week
"""
    brief = ask_llm(system, prompt, temperature=0.25, num_ctx=12000)
    filename = os.path.join(REPORT_DIR, f"weekly_brief_{today()}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(brief)
    print(f"[bold magenta]Saved report:[/bold magenta] {filename}")
    save_memory("weekly_brief", brief[:1800], source=filename, confidence=0.75, tags=["brief", "strategy"])
    return brief


def generate_research_gap_analysis(limit: int = 20) -> str:
    ensure_dirs()
    profile = load_profile()
    papers = get_top_papers(limit=limit)
    if not papers:
        return "No papers found yet. Run: python agent.py run"
    paper_text = "\n\n".join(json.dumps(p, indent=2, ensure_ascii=False) for p in papers)
    system = "You are a skeptical research strategist. Compare across papers and identify fundable gaps."
    prompt = f"""
Profile:
{mission_context(profile)}

Papers:
{paper_text}

Do not summarize one paper at a time. Compare patterns.
Return markdown sections:
# Research Gap Analysis
## 1. Best Gap Candidate
## 2. Evidence Pattern
## 3. Why It Matters
## 4. Fit to My Strengths
## 5. Possible Research Question
## 6. Grant Concept
## 7. Reviewer Attack Points
## 8. Next 3 Actions
"""
    gap = ask_llm(system, prompt, temperature=0.25, num_ctx=12000)
    filename = os.path.join(REPORT_DIR, f"research_gap_analysis_{today()}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(gap)
    print(f"[bold magenta]Saved gap analysis:[/bold magenta] {filename}")
    save_memory("research_gap", gap[:1800], source=filename, confidence=0.7, tags=["gap", "grant"])
    return gap


PROTECTED_TOPICS = [
    "OLED", "TADF", "organic semiconductor", "quantum chemistry", "red-NIR emission",
    "lanthanide complex", "charge transport", "device degradation",
]
NEVER_IN_FILTERS = [
    "oled", "tadf", "organic", "semiconductor", "quantum", "lanthanide",
    "device fabrication", "device physics", "charge transport", "red-nir", "nir",
    "exciplex", "phosphorescent", "emission", "emitter",
]
REQUIRED_WEIGHT_KEYS = [
    "relevance", "novelty", "grant_potential", "collaboration_potential",
    "industry_potential", "teaching_public_value",
]


def _sanitize_profile(updated: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    existing = list(updated.get("topics", []) or [])
    for t in PROTECTED_TOPICS:
        if t not in existing:
            existing.append(t)
    seen_topics, deduped_topics = set(), []
    for t in existing:
        t = str(t).strip()
        if t and t.lower() not in seen_topics:
            seen_topics.add(t.lower())
            deduped_topics.append(t)
    updated["topics"] = deduped_topics

    raw, orig_w = updated.get("weights", {}) or {}, original.get("weights", {}) or {}
    clean: Dict[str, float] = {}
    for k in REQUIRED_WEIGHT_KEYS:
        try:
            clean[k] = float(str(raw.get(k, orig_w.get(k, 1 / len(REQUIRED_WEIGHT_KEYS)))).split()[0])
        except Exception:
            clean[k] = float(orig_w.get(k, 1 / len(REQUIRED_WEIGHT_KEYS)))
    total = sum(clean.values()) or 1.0
    updated["weights"] = {k: round(v / total, 4) for k, v in clean.items()}

    for key in ("watchlist", "brief_instruction", "identity"):
        if key not in updated and key in original:
            updated[key] = original[key]

    seen_filters, deduped_filters = set(), []
    for f in updated.get("negative_filters", []) or []:
        f_str = str(f).strip()
        f_lower = f_str.lower()
        if not f_str or f_lower in seen_filters or any(bad in f_lower for bad in NEVER_IN_FILTERS):
            continue
        seen_filters.add(f_lower)
        deduped_filters.append(f_str)
    updated["negative_filters"] = deduped_filters
    return updated


def evolve_profile_from_feedback(feedback_text: str, *, require_approval: bool = False) -> None:
    init_db()
    profile = load_profile()
    if require_approval:
        print("[yellow]Profile evolution requested. Review feedback before applying:[/yellow]")
        print(feedback_text)
        answer = input("Apply this profile update? Type YES to continue: ")
        if answer.strip() != "YES":
            print("[yellow]Cancelled.[/yellow]")
            return

    system = "You improve a YAML research profile conservatively. Preserve core identity and protected topics."
    prompt = f"""
Current YAML profile:
{mission_context(profile)}

Feedback:
{feedback_text}

Rules:
- Keep keys: identity, topics, weights, watchlist, negative_filters, brief_instruction.
- Weights must be plain numbers only.
- Never remove OLED, TADF, organic semiconductor, red-NIR emission, lanthanide complex, charge transport.
- Add negative_filters only for durable irrelevant patterns.
- Return only valid YAML.
"""
    yaml_text = clean_yaml(ask_llm(system, prompt, temperature=0.1, num_ctx=8192))
    try:
        updated = yaml.safe_load(yaml_text)
        required = {"identity", "topics", "weights", "watchlist", "negative_filters"}
        if not required.issubset(set(updated.keys())):
            raise ValueError(f"Updated YAML missing keys: {required - set(updated.keys())}")
        updated = _sanitize_profile(updated, profile)
        save_profile(updated)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO feedback (feedback_text, created_at) VALUES (?, ?)", (feedback_text, now()))
        conn.commit()
        conn.close()
        save_memory("profile_evolution", feedback_text, "feedback", confidence=0.8, tags=["profile", "evolution"])
        print("[bold green]Profile evolved successfully.[/bold green]")
    except Exception as e:
        print("[red]Could not safely update profile.[/red]")
        print(e)
        print("\nModel output was:\n")
        print(yaml_text)


def generate_self_feedback(limit: int = 20) -> str:
    profile = load_profile()
    papers = get_top_papers(limit=limit)
    if not papers:
        return ""
    system = "You are the self-improvement module. Critique curation and workflow quality."
    prompt = f"""
Profile:
{mission_context(profile)}

Top papers:
{json.dumps(papers, indent=2, ensure_ascii=False)}

Write 3-6 sentences of concrete feedback to improve future outputs.
Focus on low-value patterns, missed grant angles, weak evidence, and scoring calibration.
Do not output YAML or JSON.
"""
    return ask_llm(system, prompt, temperature=0.3, num_ctx=8192).strip()


def auto_evolve(limit: int = 20, apply: bool = False) -> None:
    print("[bold blue]── Self-Evolution Cycle ──[/bold blue]")
    feedback = generate_self_feedback(limit=limit)
    if not feedback:
        print("[yellow]No papers in database. Run: python agent.py run[/yellow]")
        return
    ensure_dirs()
    fb_file = os.path.join(REPORT_DIR, f"self_feedback_{today()}.md")
    with open(fb_file, "w", encoding="utf-8") as f:
        f.write(f"# Self-Generated Feedback — {today()}\n\n{feedback}")
    print("[bold yellow]Self-generated feedback:[/bold yellow]")
    print(feedback)
    print(f"[dim]Feedback saved → {fb_file}[/dim]")
    if apply:
        evolve_profile_from_feedback(feedback, require_approval=True)
    else:
        print("[yellow]Profile not changed. Re-run with --apply to update after approval.[/yellow]")


def run_full_loop(max_per_topic: int = 3, brief_limit: int = 12, gap_limit: int = 20, apply_evolution: bool = False) -> None:
    print("[bold cyan]══ Research Evolution Agent — Agentic Loop ══[/bold cyan]\n")
    print("[bold]Phase 1/4  Fetching and scoring papers…[/bold]")
    collect_and_score(max_per_topic=max_per_topic)
    print("\n[bold]Phase 2/4  Generating weekly brief…[/bold]")
    generate_weekly_brief(limit=brief_limit)
    print("\n[bold]Phase 3/4  Generating gap analysis…[/bold]")
    generate_research_gap_analysis(limit=gap_limit)
    print("\n[bold]Phase 4/4  Self-evolution review…[/bold]")
    auto_evolve(limit=gap_limit, apply=apply_evolution)
    print("\n[bold cyan]══ Loop complete. ══[/bold cyan]")


def show_top(limit: int = 10) -> None:
    papers = get_top_papers(limit=limit)
    if not papers:
        print("[yellow]No papers yet. Run: python agent.py run[/yellow]")
        return
    for i, p in enumerate(papers, start=1):
        print(f"\n[bold cyan]{i}. {p['title']}[/bold cyan]")
        print(f"Score: {p['total_score']} | Action: {p['recommended_action']} | Risk: {p.get('risk', 5)}")
        print(f"Date: {p['published_date']} | Source: {p['source']}")
        print(f"Authors: {p['authors']}")
        print(f"Commentary: {p['agent_commentary']}")
        print(f"Evidence gaps: {p.get('evidence_gaps', '')}")
        print(f"URL: {p['url']}")


def show_tasks(limit: int = 20) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, agent, priority, effort, risk, status, created_at FROM tasks ORDER BY status, priority DESC, id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("[yellow]No tasks yet. Run: python agent.py think 'your idea'[/yellow]")
        return
    for r in rows:
        print(f"[bold]{r[0]}.[/bold] {r[1]} | agent={r[2]} | P={r[3]} E={r[4]} R={r[5]} | {r[6]} | {r[7]}")


def print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Research Evolution Agent powered by local Ollama/Qwen3")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Search and score new papers.")
    run_parser.add_argument("--max-per-topic", type=int, default=3)

    brief_parser = sub.add_parser("brief", help="Generate weekly research brief.")
    brief_parser.add_argument("--limit", type=int, default=12)

    top_parser = sub.add_parser("top", help="Show top scored papers.")
    top_parser.add_argument("--limit", type=int, default=10)

    gap_parser = sub.add_parser("gap", help="Generate research gap analysis.")
    gap_parser.add_argument("--limit", type=int, default=20)

    think_parser = sub.add_parser("think", help="Run the agentic multi-role workflow on an idea, paper note, or proposal fragment.")
    think_parser.add_argument("text", type=str)
    think_parser.add_argument("--no-communication", action="store_true")

    feedback_parser = sub.add_parser("feedback", help="Evolve profile from explicit feedback.")
    feedback_parser.add_argument("text", type=str)
    feedback_parser.add_argument("--apply", action="store_true", help="Ask for approval, then apply the profile update.")

    evolve_parser = sub.add_parser("evolve", help="Self-critique curation and optionally update profile after approval.")
    evolve_parser.add_argument("--limit", type=int, default=20)
    evolve_parser.add_argument("--apply", action="store_true")

    loop_parser = sub.add_parser("loop", help="Full agentic cycle: run → brief → gap → self-evolution review.")
    loop_parser.add_argument("--max-per-topic", type=int, default=3)
    loop_parser.add_argument("--brief-limit", type=int, default=12)
    loop_parser.add_argument("--gap-limit", type=int, default=20)
    loop_parser.add_argument("--apply-evolution", action="store_true")

    tasks_parser = sub.add_parser("tasks", help="Show agent-created task ledger.")
    tasks_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if args.command == "run":
        collect_and_score(max_per_topic=args.max_per_topic)
    elif args.command == "brief":
        print("\n" + generate_weekly_brief(limit=args.limit))
    elif args.command == "top":
        show_top(limit=args.limit)
    elif args.command == "gap":
        print("\n" + generate_research_gap_analysis(limit=args.limit))
    elif args.command == "think":
        print_json(run_agentic_workflow(args.text, include_communication=not args.no_communication))
    elif args.command == "feedback":
        evolve_profile_from_feedback(args.text, require_approval=not args.apply)
    elif args.command == "evolve":
        auto_evolve(limit=args.limit, apply=args.apply)
    elif args.command == "loop":
        run_full_loop(
            max_per_topic=args.max_per_topic,
            brief_limit=args.brief_limit,
            gap_limit=args.gap_limit,
            apply_evolution=args.apply_evolution,
        )
    elif args.command == "tasks":
        show_tasks(limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
