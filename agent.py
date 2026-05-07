import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import textwrap
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests
import yaml
from dotenv import load_dotenv
from ollama import chat
from rich import print


DB_PATH = "memory.db"
PROFILE_PATH = "research_profile.yaml"
REPORT_DIR = "reports"


def load_settings() -> Dict[str, str]:
    load_dotenv()
    return {
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "openalex_api_key": os.getenv("OPENALEX_API_KEY", "").strip(),
    }


def load_profile() -> Dict[str, Any]:
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_profile(profile: Dict[str, Any]) -> None:
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)


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
            total_score REAL,

            agent_commentary TEXT,
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

    conn.commit()
    conn.close()


def reconstruct_openalex_abstract(inv_index: Dict[str, List[int]] | None) -> str:
    if not inv_index:
        return ""

    positioned_words = []
    for word, positions in inv_index.items():
        for pos in positions:
            positioned_words.append((pos, word))

    positioned_words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positioned_words)


def search_openalex(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    settings = load_settings()

    params = {
        "search": query,
        "per-page": max_results,
        "sort": "publication_date:desc",
    }

    if settings["openalex_api_key"]:
        params["api_key"] = settings["openalex_api_key"]

    response = requests.get(
        "https://api.openalex.org/works",
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    papers = []

    for item in data.get("results", []):
        title = item.get("title") or item.get("display_name") or ""
        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

        primary_location = item.get("primary_location") or {}
        url = (
            item.get("doi")
            or primary_location.get("landing_page_url")
            or item.get("id")
            or ""
        )

        authorships = item.get("authorships") or []
        authors = []
        for a in authorships[:8]:
            author = a.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)

        papers.append(
            {
                "source": "OpenAlex",
                "title": title.strip(),
                "abstract": abstract.strip(),
                "url": url,
                "published_date": item.get("publication_date", ""),
                "authors": ", ".join(authors),
                "cited_by_count": item.get("cited_by_count", 0),
            }
        )

    time.sleep(0.2)  # polite pause; well within OpenAlex 100 req/s limit
    return papers


def arxiv_query_from_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return "all:OLED"
    # Use a quoted phrase for multi-word topics so arXiv treats them as a unit,
    # preventing single-word splits that pull in completely unrelated papers.
    if " " in topic:
        return f'all:"{topic}"'
    return f"all:{topic}"


def search_arxiv(topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
    params = {
        "search_query": arxiv_query_from_topic(topic),
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(
        "https://export.arxiv.org/api/query",
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns)
        abstract = entry.findtext("atom:summary", default="", namespaces=ns)
        url = entry.findtext("atom:id", default="", namespaces=ns)
        published = entry.findtext("atom:published", default="", namespaces=ns)

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(name)

        papers.append(
            {
                "source": "arXiv",
                "title": " ".join(title.split()),
                "abstract": " ".join(abstract.split()),
                "url": url,
                "published_date": published[:10],
                "authors": ", ".join(authors[:8]),
                "cited_by_count": 0,
            }
        )

    time.sleep(3)  # arXiv requires >= 3 s between requests (single connection)
    return papers


def get_ollama_content(response: Any) -> str:
    try:
        return response.message.content
    except AttributeError:
        return response["message"]["content"]


def clean_json(text: str) -> Dict[str, Any]:
    text = text.strip()

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


def score_paper(profile: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    settings = load_settings()

    prompt = f"""
You are a Research Evolution Agent for a researcher working on:
{profile["topics"]}

The agent's mission:
{profile["identity"]["mission"]}

Negative filters:
{profile.get("negative_filters", [])}

Score the paper from 0 to 10 for:

1. relevance
2. novelty
3. grant_potential
4. collaboration_potential
5. industry_potential
6. teaching_public_value

Then recommend exactly one action:
- read_now
- save_for_later
- use_for_grant
- contact_author
- use_for_teaching
- use_for_linkedin
- ignore

Paper:
Title: {paper["title"]}
Authors: {paper.get("authors", "")}
Source: {paper["source"]}
Published: {paper["published_date"]}
Citations: {paper.get("cited_by_count", 0)}
Abstract: {paper["abstract"]}

Return only valid JSON:
{{
  "relevance": 0,
  "novelty": 0,
  "grant_potential": 0,
  "collaboration_potential": 0,
  "industry_potential": 0,
  "teaching_public_value": 0,
  "commentary": "short critical explanation",
  "recommended_action": "read_now"
}}
"""

    response = chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    )

    content = get_ollama_content(response)

    try:
        data = clean_json(content)
    except Exception:
        data = {
            "relevance": 5,
            "novelty": 5,
            "grant_potential": 5,
            "collaboration_potential": 5,
            "industry_potential": 5,
            "teaching_public_value": 5,
            "commentary": "LLM output could not be parsed. Manual review required.",
            "recommended_action": "save_for_later",
        }

    required = [
        "relevance",
        "novelty",
        "grant_potential",
        "collaboration_potential",
        "industry_potential",
        "teaching_public_value",
    ]

    for key in required:
        try:
            data[key] = float(data.get(key, 5))
        except Exception:
            data[key] = 5.0

    data["commentary"] = str(data.get("commentary", "")).strip()
    data["recommended_action"] = str(
        data.get("recommended_action", "save_for_later")
    ).strip()

    return data


def calculate_total(profile: Dict[str, Any], scores: Dict[str, Any]) -> float:
    w = profile["weights"]

    total = (
        w["relevance"] * scores["relevance"]
        + w["novelty"] * scores["novelty"]
        + w["grant_potential"] * scores["grant_potential"]
        + w["collaboration_potential"] * scores["collaboration_potential"]
        + w["industry_potential"] * scores["industry_potential"]
        + w["teaching_public_value"] * scores["teaching_public_value"]
    )

    return round(total, 2)


def make_unique_key(paper: Dict[str, Any]) -> str:
    base = paper.get("url") or paper.get("title") or ""
    return re.sub(r"\s+", " ", base.lower()).strip()


def save_scored_paper(
    paper: Dict[str, Any],
    scores: Dict[str, Any],
    total_score: float,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR IGNORE INTO papers (
            unique_key, source, title, abstract, url, published_date,
            authors, cited_by_count,
            relevance, novelty, grant_potential, collaboration_potential,
            industry_potential, teaching_public_value, total_score,
            agent_commentary, recommended_action, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            make_unique_key(paper),
            paper.get("source", ""),
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("url", ""),
            paper.get("published_date", ""),
            paper.get("authors", ""),
            int(paper.get("cited_by_count", 0) or 0),
            scores["relevance"],
            scores["novelty"],
            scores["grant_potential"],
            scores["collaboration_potential"],
            scores["industry_potential"],
            scores["teaching_public_value"],
            total_score,
            scores["commentary"],
            scores["recommended_action"],
            dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )

    conn.commit()
    conn.close()


def collect_and_score(max_per_topic: int = 3) -> None:
    init_db()
    profile = load_profile()

    all_papers = []

    for topic in profile["topics"]:
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

    seen = set()
    unique_papers = []

    for paper in all_papers:
        key = make_unique_key(paper)
        if not paper.get("title") or key in seen:
            continue
        seen.add(key)
        unique_papers.append(paper)

    print(f"[bold green]Found {len(unique_papers)} unique papers.[/bold green]")

    for paper in unique_papers:
        title = paper["title"][:100]
        print(f"[bold yellow]Scoring:[/bold yellow] {title}")

        scores = score_paper(profile, paper)
        total = calculate_total(profile, scores)
        save_scored_paper(paper, scores, total)

        print(
            f"  Score={total} | Action={scores['recommended_action']} | "
            f"{scores['commentary'][:120]}"
        )


def get_top_papers(limit: int = 12) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            title, abstract, url, published_date, authors, source,
            cited_by_count, relevance, novelty, grant_potential,
            collaboration_potential, industry_potential,
            teaching_public_value, total_score,
            agent_commentary, recommended_action
        FROM papers
        ORDER BY total_score DESC, published_date DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    papers = []
    for r in rows:
        papers.append(
            {
                "title": r[0],
                "abstract": r[1],
                "url": r[2],
                "published_date": r[3],
                "authors": r[4],
                "source": r[5],
                "cited_by_count": r[6],
                "relevance": r[7],
                "novelty": r[8],
                "grant_potential": r[9],
                "collaboration_potential": r[10],
                "industry_potential": r[11],
                "teaching_public_value": r[12],
                "total_score": r[13],
                "agent_commentary": r[14],
                "recommended_action": r[15],
            }
        )

    return papers


def generate_weekly_brief(limit: int = 12) -> str:
    init_db()
    profile = load_profile()
    settings = load_settings()
    papers = get_top_papers(limit=limit)

    if not papers:
        return "No papers found yet. Run: python agent.py run"

    paper_text = "\n\n".join(
        [
            textwrap.dedent(
                f"""
                Title: {p["title"]}
                Source: {p["source"]}
                Date: {p["published_date"]}
                Authors: {p["authors"]}
                Score: {p["total_score"]}
                Action: {p["recommended_action"]}
                Commentary: {p["agent_commentary"]}
                URL: {p["url"]}
                """
            ).strip()
            for p in papers
        ]
    )

    prompt = f"""
You are my Research Evolution Agent.

My research profile:
{yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)}

Create a Weekly Research Evolution Brief.

Use this format:

# Weekly Research Evolution Brief

## 1. Executive Summary
Give a concise strategic summary.

## 2. Top 5 Papers or Opportunities
For each:
- Title
- Why it matters
- Connection to my work
- Recommended action

## 3. Emerging Research Gap
Identify one realistic research gap I may be positioned to attack.

## 4. Grant Angle
Suggest one grant proposal angle.

## 5. Collaboration Angle
Suggest one type of collaborator or institution.

## 6. Industry Angle
Suggest one industry application or company angle.

## 7. Teaching / LinkedIn Angle
Suggest one public-facing explanation or post idea.

## 8. One High-Leverage Action This Week
Give one specific action.

Papers:
{paper_text}
"""

    response = chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.25,
            "num_ctx": 8192,
        },
    )

    brief = get_ollama_content(response)

    os.makedirs(REPORT_DIR, exist_ok=True)
    filename = os.path.join(
        REPORT_DIR,
        f"weekly_brief_{dt.date.today().isoformat()}.md",
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(brief)

    print(f"[bold magenta]Saved report:[/bold magenta] {filename}")
    return brief


def clean_yaml(text: str) -> str:
    text = text.strip()

    fenced = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    return text


# Topics that must never be dropped by any LLM-driven evolution.
PROTECTED_TOPICS = [
    "OLED",
    "TADF",
    "organic semiconductor",
    "quantum chemistry",
    "red-NIR emission",
    "lanthanide complex",
    "charge transport",
    "device degradation",
]

# Substrings that must never appear in negative_filters — the LLM occasionally
# adds core research topics as filters when it misreads the curation stats.
NEVER_IN_FILTERS = [
    "oled", "tadf", "organic", "semiconductor", "quantum", "lanthanide",
    "device fabrication", "device physics", "charge transport", "red-nir",
    "nir", "exciplex", "phosphorescent", "emission", "emitter",
]

# Weight keys that must always be present.
REQUIRED_WEIGHT_KEYS = [
    "relevance",
    "novelty",
    "grant_potential",
    "collaboration_potential",
    "industry_potential",
    "teaching_public_value",
]


def _sanitize_profile(updated: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    """Guard rails applied after every LLM-driven profile update."""

    # 1. Restore any protected topics that were dropped.
    existing = updated.get("topics", [])
    for t in PROTECTED_TOPICS:
        if t not in existing:
            existing.append(t)
    updated["topics"] = existing

    # 2. Ensure all required weight keys exist (fall back to original values).
    raw = updated.get("weights", {})
    orig_w = original.get("weights", {})
    clean: Dict[str, float] = {}
    for k in REQUIRED_WEIGHT_KEYS:
        raw_val = raw.get(k, orig_w.get(k, 1.0 / len(REQUIRED_WEIGHT_KEYS)))
        try:
            clean[k] = float(str(raw_val).split()[0])  # strip inline comments
        except (ValueError, TypeError):
            clean[k] = orig_w.get(k, 1.0 / len(REQUIRED_WEIGHT_KEYS))

    # 3. Renormalise weights to exactly 1.0.
    total_w = sum(clean.values()) or 1.0
    updated["weights"] = {k: round(v / total_w, 4) for k, v in clean.items()}

    # 4. Preserve watchlist and brief_instruction if LLM dropped them.
    for key in ("watchlist", "brief_instruction", "identity"):
        if key not in updated and key in original:
            updated[key] = original[key]

    # 5. Deduplicate negative_filters, remove any that overlap with core topics.
    seen_filters: set = set()
    deduped = []
    for f in updated.get("negative_filters", []):
        f_stripped = f.strip()
        f_lower = f_stripped.lower()
        if f_lower in seen_filters:
            continue
        # Reject any filter whose text contains a protected research keyword.
        if any(bad in f_lower for bad in NEVER_IN_FILTERS):
            continue
        seen_filters.add(f_lower)
        deduped.append(f_stripped)
    updated["negative_filters"] = deduped

    # 6. Deduplicate topics, preserve order.
    seen_topics: set = set()
    deduped_topics = []
    for t in updated.get("topics", []):
        if t.strip().lower() not in seen_topics:
            seen_topics.add(t.strip().lower())
            deduped_topics.append(t.strip())
    updated["topics"] = deduped_topics

    return updated


def evolve_profile_from_feedback(feedback_text: str) -> None:
    init_db()
    profile = load_profile()
    settings = load_settings()

    prompt = f"""
You are improving my Research Evolution Agent.

Current YAML research profile:
{yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)}

User feedback:
{feedback_text}

Update the YAML profile carefully.

Rules:
- Keep the same top-level keys: identity, topics, weights, watchlist, negative_filters, brief_instruction.
- Weights must be plain numbers only — no text, no comments, no parentheses.
- Adjust weights only when justified by the feedback.
- Add topics only if they are durable research interests.
- Add to negative_filters when feedback identifies irrelevant outputs.
- Never remove OLED, TADF, organic semiconductor, or red-NIR emission from topics.
- Return only valid YAML, nothing else.
"""

    response = chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_ctx": 4096},
    )

    yaml_text = clean_yaml(get_ollama_content(response))

    try:
        updated = yaml.safe_load(yaml_text)

        required_keys = {"identity", "topics", "weights", "watchlist", "negative_filters"}
        if not required_keys.issubset(set(updated.keys())):
            raise ValueError(f"Updated YAML missing keys: {required_keys - set(updated.keys())}")

        updated = _sanitize_profile(updated, profile)
        save_profile(updated)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO feedback (feedback_text, created_at) VALUES (?, ?)",
            (feedback_text, dt.datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()

        w = updated["weights"]
        print("[bold green]Profile evolved successfully.[/bold green]")
        print(f"  Topics  : {len(updated['topics'])} ({', '.join(updated['topics'][:4])}…)")
        print(f"  Weights : relevance={w['relevance']:.2f}  novelty={w['novelty']:.2f}  "
              f"grant={w['grant_potential']:.2f}  (sum={sum(w.values()):.3f})")

    except Exception as e:
        print("[red]Could not safely update profile.[/red]")
        print(e)
        print("\nModel output was:\n")
        print(yaml_text)


def generate_self_feedback(limit: int = 20) -> str:
    """LLM critiques the agent's own paper curation and returns plain-English feedback."""
    profile = load_profile()
    settings = load_settings()
    papers = get_top_papers(limit=limit)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT total_score, recommended_action, source, title FROM papers"
    )
    all_rows = cur.fetchall()
    conn.close()

    if not all_rows:
        return ""

    scores = [r[0] for r in all_rows]
    avg_score = sum(scores) / len(scores)
    action_counts: Dict[str, int] = {}
    for r in all_rows:
        action_counts[r[1]] = action_counts.get(r[1], 0) + 1
    high = sum(1 for s in scores if s >= 6.0)
    low  = sum(1 for s in scores if s < 3.0)

    paper_summary = "\n".join(
        f"  {p['total_score']:.2f} | {p['recommended_action']:<18} | {p['title'][:80]}"
        for p in papers
    )

    prompt = f"""You are the self-improvement module of the Research Evolution Agent.

Current research profile:
{yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)}

Curation statistics from the database:
  Total papers scored : {len(all_rows)}
  Average score       : {avg_score:.2f} / 10
  High-relevance (≥6) : {high}
  Low-relevance (<3)  : {low}
  Action breakdown    : {action_counts}

Top {limit} papers by score:
{paper_summary}

Task: Write specific, actionable feedback (3-6 sentences) to improve the research profile.

Focus on:
- What topics or paper types are producing too many low-score results
- What is being missed or under-weighted (red-NIR TADF, lanthanide, device fabrication, grants)
- Whether negative_filters should be tightened
- Whether any weights should shift to reflect the researcher's priorities

Rules:
- Be concrete — reference actual patterns you see in the paper list above.
- Do NOT output JSON or YAML. Output plain English feedback only.
- Write as if a researcher is reviewing the agent's weekly output and giving guidance.
"""

    response = chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_ctx": 4096},
    )

    return get_ollama_content(response).strip()


def auto_evolve(limit: int = 20) -> None:
    """Full self-evolution cycle: critique curation → generate feedback → update profile."""
    init_db()

    print("[bold blue]── Self-Evolution Cycle ──[/bold blue]")
    print("[dim]Step 1/2  Generating self-critique…[/dim]")

    feedback = generate_self_feedback(limit=limit)

    if not feedback:
        print("[yellow]No papers in database. Run: python agent.py run[/yellow]")
        return

    print("\n[bold yellow]Self-generated feedback:[/bold yellow]")
    print(feedback)
    print()

    # Save the auto-generated feedback to reports/ for audit trail.
    os.makedirs(REPORT_DIR, exist_ok=True)
    fb_file = os.path.join(
        REPORT_DIR,
        f"self_feedback_{dt.date.today().isoformat()}.md",
    )
    with open(fb_file, "w", encoding="utf-8") as f:
        f.write(f"# Self-Generated Feedback — {dt.date.today().isoformat()}\n\n")
        f.write(feedback)
    print(f"[dim]Feedback saved → {fb_file}[/dim]\n")

    print("[dim]Step 2/2  Applying feedback to profile…[/dim]")
    evolve_profile_from_feedback(feedback)


def run_full_loop(max_per_topic: int = 3, brief_limit: int = 12, gap_limit: int = 20) -> None:
    """Full autonomous cycle: run → brief → gap → self-evolve."""
    print("[bold cyan]══ Research Evolution Agent — Full Loop ══[/bold cyan]\n")

    print("[bold]Phase 1/4  Fetching and scoring papers…[/bold]")
    collect_and_score(max_per_topic=max_per_topic)
    print()

    print("[bold]Phase 2/4  Generating weekly brief…[/bold]")
    generate_weekly_brief(limit=brief_limit)
    print()

    print("[bold]Phase 3/4  Generating gap analysis…[/bold]")
    generate_research_gap_analysis(limit=gap_limit)
    print()

    print("[bold]Phase 4/4  Self-evolving profile…[/bold]")
    auto_evolve(limit=gap_limit)
    print()

    print("[bold cyan]══ Loop complete. Profile updated for next run. ══[/bold cyan]")


def show_top(limit: int = 10) -> None:
    papers = get_top_papers(limit=limit)

    if not papers:
        print("[yellow]No papers yet. Run: python agent.py run[/yellow]")
        return

    for i, p in enumerate(papers, start=1):
        print(f"\n[bold cyan]{i}. {p['title']}[/bold cyan]")
        print(f"Score: {p['total_score']} | Action: {p['recommended_action']}")
        print(f"Date: {p['published_date']} | Source: {p['source']}")
        print(f"Authors: {p['authors']}")
        print(f"Commentary: {p['agent_commentary']}")
        print(f"URL: {p['url']}")


def generate_research_gap_analysis(limit: int = 20) -> str:
    init_db()
    profile = load_profile()
    settings = load_settings()
    papers = get_top_papers(limit=limit)

    if not papers:
        return "No papers found yet. Run: python agent.py run"

    paper_text = "\n\n".join(
        [
            f"""
Title: {p["title"]}
Date: {p["published_date"]}
Authors: {p["authors"]}
Score: {p["total_score"]}
Abstract: {p["abstract"][:1200]}
Commentary: {p["agent_commentary"]}
"""
            for p in papers
        ]
    )

    prompt = f"""
You are my Research Evolution Agent.

My research profile:
{yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)}

Analyze these papers to identify research gaps.

Do not merely summarize. Compare across papers.

Find:
1. Repeated limitations across the papers
2. Overcrowded research areas
3. Underexplored combinations of topics
4. Missing methods or missing measurements
5. Gaps connected to OLEDs, TADF, organic semiconductors, quantum chemistry, red-NIR emission, lanthanide complexes, photobiomodulation, device fabrication, charge transport, and grants
6. Which gap best fits my strengths
7. Which gap could become a grant proposal
8. Which gap could become a publishable paper
9. Which gap has industry or biomedical relevance
10. One concrete next action

Return this format:

# Research Gap Analysis

## 1. Best Research Gap Candidate
State the gap clearly.

## 2. Evidence from Papers
Explain which patterns in the papers support this gap.

## 3. Why This Gap Matters
Explain scientific, technological, grant, or industry importance.

## 4. Why I Am Positioned to Work on It
Connect to my OLED/device/organic electronics background.

## 5. Possible Research Question
Write one strong research question.

## 6. Possible Grant Angle
Write one grant concept.

## 7. Risk or Weakness
Explain why this gap may be hard or uncertain.

## 8. Next Action
Give one concrete action this week.

Papers:
{paper_text}
"""

    response = chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.25,
            "num_ctx": 8192,
        },
    )

    gap_analysis = get_ollama_content(response)

    os.makedirs(REPORT_DIR, exist_ok=True)
    filename = os.path.join(
        REPORT_DIR,
        f"research_gap_analysis_{dt.date.today().isoformat()}.md",
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(gap_analysis)

    print(f"[bold magenta]Saved gap analysis:[/bold magenta] {filename}")
    return gap_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Evolution Agent")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Search and score new papers.")
    run_parser.add_argument("--max-per-topic", type=int, default=3)

    brief_parser = sub.add_parser("brief", help="Generate weekly research brief.")
    brief_parser.add_argument("--limit", type=int, default=12)

    top_parser = sub.add_parser("top", help="Show top scored papers.")
    top_parser.add_argument("--limit", type=int, default=10)

    feedback_parser = sub.add_parser("feedback", help="Evolve profile from feedback.")
    feedback_parser.add_argument("text", type=str)

    gap_parser = sub.add_parser("gap", help="Generate research gap analysis.")
    gap_parser.add_argument("--limit", type=int, default=20)

    evolve_parser = sub.add_parser("evolve", help="Self-evolve: agent critiques its own curation and updates profile.")
    evolve_parser.add_argument("--limit", type=int, default=20)

    loop_parser = sub.add_parser("loop", help="Full autonomous cycle: run → brief → gap → self-evolve.")
    loop_parser.add_argument("--max-per-topic", type=int, default=3)
    loop_parser.add_argument("--brief-limit", type=int, default=12)
    loop_parser.add_argument("--gap-limit",   type=int, default=20)

    args = parser.parse_args()

    if args.command == "run":
        collect_and_score(max_per_topic=args.max_per_topic)
    elif args.command == "brief":
        brief = generate_weekly_brief(limit=args.limit)
        print("\n" + brief)
    elif args.command == "top":
        show_top(limit=args.limit)
    elif args.command == "feedback":
        evolve_profile_from_feedback(args.text)
    elif args.command == "gap":
        gap = generate_research_gap_analysis(limit=args.limit)
        print("\n" + gap)
    elif args.command == "evolve":
        auto_evolve(limit=args.limit)
    elif args.command == "loop":
        run_full_loop(
            max_per_topic=args.max_per_topic,
            brief_limit=args.brief_limit,
            gap_limit=args.gap_limit,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
