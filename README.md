# Research Evolution Agent

A local-first AI research assistant that helps a researcher continuously discover, score, prioritize, and strategically interpret new papers from OpenAlex and arXiv.

The agent is designed for researchers working in **OLEDs, TADF, organic semiconductors, red-NIR emission, lanthanide complexes, device physics, quantum chemistry, materials discovery, and related grant/collaboration opportunities**. It uses a local Ollama model to evaluate papers against a configurable research profile, stores the results in a SQLite memory database, generates weekly research briefs, identifies research gaps, and can evolve its own profile from feedback.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Core Capabilities](#core-capabilities)
- [How the Agent Works](#how-the-agent-works)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Ollama Setup](#ollama-setup)
- [Environment Variables](#environment-variables)
- [Research Profile Configuration](#research-profile-configuration)
- [Usage](#usage)
- [Command Reference](#command-reference)
- [Generated Outputs](#generated-outputs)
- [Database Design](#database-design)
- [Scoring System](#scoring-system)
- [Recommended Action Labels](#recommended-action-labels)
- [Self-Evolution System](#self-evolution-system)
- [Workflow Examples](#workflow-examples)
- [Troubleshooting](#troubleshooting)
- [Suggested `.gitignore`](#suggested-gitignore)
- [Development Notes](#development-notes)
- [Roadmap Ideas](#roadmap-ideas)
- [Limitations](#limitations)
- [Security and Privacy](#security-and-privacy)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## What This Project Does

The **Research Evolution Agent** automates part of the research-intelligence workflow:

1. Reads your research profile from `research_profile.yaml`.
2. Searches for recent papers from:
   - OpenAlex
   - arXiv
3. Uses a local Ollama-hosted language model to score each paper.
4. Saves scored papers into a local SQLite database called `memory.db`.
5. Generates strategic Markdown reports inside the `reports/` directory.
6. Produces research-gap analysis based on the top-ranked papers.
7. Updates the research profile from direct feedback or self-generated feedback.
8. Supports a full autonomous loop: search → score → brief → gap analysis → self-evolution.

The project is especially useful if you want a repeatable, local, customizable assistant that helps answer questions such as:

- Which recent papers are most relevant to my research direction?
- Which papers should I read now, save, ignore, or use for grant development?
- What emerging research gap fits my strengths?
- Which paper could support a collaboration or industry conversation?
- What weekly research brief can I generate from the literature?
- How should my research profile evolve after reviewing the agent’s output?

---

## Core Capabilities

### 1. Literature Discovery

The agent searches papers using your configured research topics. For each topic, it queries:

- **OpenAlex** using the `/works` endpoint.
- **arXiv** using the arXiv Atom API.

The agent retrieves metadata such as:

- title
- abstract
- authors
- source
- publication date
- URL or DOI
- citation count, where available

### 2. Local LLM-Based Scoring

Each paper is evaluated using an Ollama-hosted local model. By default, the model is:

```text
llama3.1:8b
```

The model scores each paper on:

- relevance
- novelty
- grant potential
- collaboration potential
- industry potential
- teaching/public communication value

### 3. SQLite Research Memory

The agent stores results in a local SQLite database:

```text
memory.db
```

This allows the system to remember previously scored papers and retrieve top-ranked papers later for briefs, gap analysis, and self-evolution.

### 4. Weekly Research Brief Generation

The agent can generate a Markdown report containing:

- executive summary
- top papers or opportunities
- emerging research gap
- grant angle
- collaboration angle
- industry angle
- teaching or LinkedIn angle
- one high-leverage action for the week

Generated briefs are saved in:

```text
reports/weekly_brief_YYYY-MM-DD.md
```

### 5. Research Gap Analysis

The agent can compare top papers and produce a strategic gap analysis with:

- best research-gap candidate
- evidence from papers
- why the gap matters
- why you are positioned to work on it
- possible research question
- possible grant angle
- risk or weakness
- concrete next action

Generated gap analyses are saved in:

```text
reports/research_gap_analysis_YYYY-MM-DD.md
```

### 6. Human Feedback Profile Evolution

You can provide feedback such as:

```bash
python agent.py feedback "Prioritize red-NIR OLED device papers more strongly and ignore purely theoretical work with no device relevance."
```

The agent will use the local LLM to update `research_profile.yaml` while preserving core profile structure and protected research topics.

### 7. Self-Evolution

The agent can critique its own paper curation, generate feedback, save that feedback as an audit trail, and then update the research profile.

Self-generated feedback is saved as:

```text
reports/self_feedback_YYYY-MM-DD.md
```

---

## How the Agent Works

The full conceptual pipeline is:

```text
research_profile.yaml
        │
        ▼
Load research identity, topics, weights, watchlist, and filters
        │
        ▼
Search OpenAlex and arXiv for each topic
        │
        ▼
Deduplicate retrieved papers
        │
        ▼
Score each paper using local Ollama model
        │
        ▼
Calculate weighted total score
        │
        ▼
Save results to SQLite database memory.db
        │
        ├──► Show top papers
        ├──► Generate weekly brief
        ├──► Generate research-gap analysis
        └──► Self-evolve research profile
```

The project is designed to be **local-first**. The LLM runs locally through Ollama, while paper metadata is retrieved from public scholarly APIs.

---

## Repository Structure

Recommended GitHub repository layout:

```text
research-evolution-agent/
│
├── agent.py                 # Main command-line agent
├── research_profile.yaml    # User research identity, topics, weights, watchlist, filters
├── requirements.txt         # Python dependencies
├── setup_ollama.ps1         # Windows PowerShell setup script for Ollama + llama3.1:8b
├── test_ollama.py           # Ollama smoke test using Python client
├── README.md                # Project documentation
│
├── reports/                 # Generated Markdown reports; created automatically
│   ├── weekly_brief_YYYY-MM-DD.md
│   ├── research_gap_analysis_YYYY-MM-DD.md
│   └── self_feedback_YYYY-MM-DD.md
│
├── memory.db                # SQLite database; generated automatically
└── .env                     # Optional local environment variables; not committed
```

### Main Files

| File | Purpose |
|---|---|
| `agent.py` | Main CLI application. Searches papers, scores them, stores results, generates reports, and evolves the profile. |
| `research_profile.yaml` | Defines your research mission, topics, scoring weights, watchlist, negative filters, and preferred output style. |
| `requirements.txt` | Lists required Python packages. |
| `setup_ollama.ps1` | Installs Ollama on Windows, pulls `llama3.1:8b`, and runs a CLI smoke test. |
| `test_ollama.py` | Verifies that the Ollama Python client can connect to the local model. |
| `memory.db` | Generated SQLite database storing papers and feedback. |
| `reports/` | Generated Markdown reports. |

---

## Prerequisites

### Required

- Python 3.10 or newer recommended
- Internet connection for OpenAlex and arXiv searches
- Ollama installed locally
- At least one Ollama model pulled locally
- Git, if you plan to upload this project to GitHub

### Recommended Hardware

For `llama3.1:8b`, a practical development machine should have:

- 16 GB RAM recommended
- 8 GB RAM may work but can be slow
- GPU acceleration optional but helpful
- Several GB of free disk space for the model and generated outputs

The setup script notes that `llama3.1:8b` is approximately several GB in size.

---

## Quick Start

For Windows users:

```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/research-evolution-agent.git
cd research-evolution-agent

# 2. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull the local model
powershell -NoProfile -ExecutionPolicy Bypass -File setup_ollama.ps1

# 5. Test Ollama Python integration
python test_ollama.py

# 6. Run the agent
python agent.py run --max-per-topic 3

# 7. Show top scored papers
python agent.py top --limit 10

# 8. Generate a weekly brief
python agent.py brief --limit 12

# 9. Generate a research-gap analysis
python agent.py gap --limit 20
```

For macOS or Linux users, install Ollama manually, pull the model, then run the Python commands:

```bash
ollama pull llama3.1:8b
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_ollama.py
python agent.py run --max-per-topic 3
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/research-evolution-agent.git
cd research-evolution-agent
```

Replace `YOUR_USERNAME` with your GitHub username or organization name.

### 2. Create a Python Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The current dependency list is:

```text
requests
pyyaml
python-dotenv
ollama
rich
```

### Dependency Roles

| Package | Role |
|---|---|
| `requests` | Makes HTTP requests to OpenAlex and arXiv. |
| `pyyaml` | Reads and writes the YAML research profile. |
| `python-dotenv` | Loads optional environment variables from `.env`. |
| `ollama` | Connects Python to the local Ollama model. |
| `rich` | Provides formatted terminal output. |

---

## Ollama Setup

### Windows Automated Setup

The repository includes:

```text
setup_ollama.ps1
```

Run it from PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup_ollama.ps1
```

The script performs five steps:

1. Installs Ollama.
2. Verifies the Ollama version.
3. Pulls `llama3.1:8b`.
4. Lists available Ollama models.
5. Runs a quick CLI smoke test.

After the script finishes, run:

```bash
python test_ollama.py
```

### Manual Ollama Setup

If you do not use the PowerShell script:

```bash
ollama pull llama3.1:8b
ollama list
```

If Ollama is not already running, start it:

```bash
ollama serve
```

Then test:

```bash
python test_ollama.py
```

Expected successful output includes a message similar to:

```text
[PASS] Model responded correctly. Ollama is working!
```

---

## Environment Variables

The agent can run without a `.env` file because it defaults to:

```text
OLLAMA_MODEL=llama3.1:8b
OPENALEX_API_KEY=
```

However, you may create a `.env` file for customization:

```bash
cp .env.example .env
```

If no `.env.example` exists yet, create `.env` manually:

```text
OLLAMA_MODEL=llama3.1:8b
OPENALEX_API_KEY=
```

### Supported Variables

| Variable | Required | Default | Description |
|---|---:|---|---|
| `OLLAMA_MODEL` | No | `llama3.1:8b` | Local Ollama model used for scoring, briefs, gap analysis, and profile evolution. |
| `OPENALEX_API_KEY` | No | empty | Optional OpenAlex API key. The code runs without it, but you may add one if needed. |

### Example Using a Different Ollama Model

```text
OLLAMA_MODEL=qwen2.5:7b
OPENALEX_API_KEY=
```

Then pull the model:

```bash
ollama pull qwen2.5:7b
```

---

## Research Profile Configuration

The research profile is stored in:

```text
research_profile.yaml
```

It contains six top-level sections:

```yaml
identity:
  name: Research Evolution Agent
  mission: Help evolve my research direction in OLEDs, TADF, organic semiconductors,
    quantum chemistry, device physics, grants, collaborators, and industry applications.

topics:
  - OLED
  - TADF
  - organic semiconductor
  - red-NIR emission
  - lanthanide complex
  - exciplex system
  - charge transport
  - molecular vibration
  - device degradation
  - machine learning materials discovery
  - quantum chemistry

weights:
  relevance: 0.339
  novelty: 0.2542
  grant_potential: 0.1695
  collaboration_potential: 0.1271
  industry_potential: 0.0254
  teaching_public_value: 0.0848

watchlist:
  researchers: []
  institutions: []
  companies:
    - OLEDWorks
    - Universal Display Corporation
    - Samsung Display
    - LG Display
    - Merck

negative_filters:
  - purely incremental
  - not related to organic electronics
  - no device relevance
  - photobiomodulation
  - non-device-related research

brief_instruction:
  output_style: "strategic, concise, critical, action-oriented"
```

### `identity`

Defines the name and mission of the agent.

Use this section to describe what the agent is ultimately trying to help you achieve.

Example:

```yaml
identity:
  name: Research Evolution Agent
  mission: Help evolve my research direction toward grant-ready OLED and organic semiconductor research.
```

### `topics`

Each topic is used as a search query seed. The agent searches both OpenAlex and arXiv for each topic.

Good topic examples:

```yaml
- red-NIR TADF OLED
- lanthanide organic complex
- charge transport organic semiconductor
- device degradation OLED
- quantum chemistry emitter design
```

Avoid overly broad topics if they produce too many irrelevant papers.

### `weights`

Weights determine how the total score is calculated.

The current score formula is:

```text
total_score =
    relevance × relevance_weight
  + novelty × novelty_weight
  + grant_potential × grant_potential_weight
  + collaboration_potential × collaboration_potential_weight
  + industry_potential × industry_potential_weight
  + teaching_public_value × teaching_public_value_weight
```

Current weights:

| Criterion | Weight | Approximate Percentage |
|---|---:|---:|
| relevance | 0.339 | 33.90% |
| novelty | 0.2542 | 25.42% |
| grant potential | 0.1695 | 16.95% |
| collaboration potential | 0.1271 | 12.71% |
| teaching/public value | 0.0848 | 8.48% |
| industry potential | 0.0254 | 2.54% |

The weights should sum to approximately `1.0`.

### `watchlist`

The profile currently supports watchlists for:

- researchers
- institutions
- companies

The code currently stores the watchlist inside the profile and includes it in strategic prompts. Future extensions could use this list for targeted author, institution, or company matching.

### `negative_filters`

These help the LLM avoid irrelevant or low-priority work.

Examples:

```yaml
negative_filters:
  - purely incremental
  - not related to organic electronics
  - no device relevance
```

Important note: the current profile includes `photobiomodulation` as a negative filter, while the gap-analysis prompt still mentions photobiomodulation as a possible relevance area. If photobiomodulation is strategically important to your research, consider removing it from `negative_filters`. If it is not currently a priority, keep it as a filter.

### `brief_instruction`

This guides the tone and style of generated reports.

Example:

```yaml
brief_instruction:
  output_style: "strategic, concise, critical, action-oriented"
```

---

## Usage

All main functionality is accessed through:

```bash
python agent.py <command> [options]
```

Before running commands, activate your virtual environment.

### Search and Score Papers

```bash
python agent.py run
```

With a custom number of papers per topic:

```bash
python agent.py run --max-per-topic 5
```

This command:

1. Initializes the SQLite database if needed.
2. Loads `research_profile.yaml`.
3. Searches OpenAlex and arXiv for every topic.
4. Deduplicates results.
5. Scores each paper using the local Ollama model.
6. Saves scored papers to `memory.db`.

### Show Top Papers

```bash
python agent.py top
```

With a custom limit:

```bash
python agent.py top --limit 15
```

This prints the highest-scoring papers from `memory.db`, including:

- title
- total score
- recommended action
- publication date
- source
- authors
- agent commentary
- URL

### Generate Weekly Research Brief

```bash
python agent.py brief
```

With a custom number of papers:

```bash
python agent.py brief --limit 12
```

This creates a report in:

```text
reports/weekly_brief_YYYY-MM-DD.md
```

The brief is also printed to the terminal.

### Generate Research Gap Analysis

```bash
python agent.py gap
```

With a custom number of top papers:

```bash
python agent.py gap --limit 20
```

This creates:

```text
reports/research_gap_analysis_YYYY-MM-DD.md
```

### Evolve Profile from Human Feedback

```bash
python agent.py feedback "Prioritize papers with device fabrication, red-NIR emission, and clear grant relevance."
```

The agent will ask the local LLM to update `research_profile.yaml` while preserving core profile structure and protected topics.

### Self-Evolve the Profile

```bash
python agent.py evolve
```

With a custom analysis limit:

```bash
python agent.py evolve --limit 20
```

This command:

1. Reviews the current database of scored papers.
2. Generates self-critique feedback.
3. Saves that feedback in `reports/`.
4. Applies the feedback to `research_profile.yaml`.

### Run the Full Autonomous Loop

```bash
python agent.py loop
```

With custom limits:

```bash
python agent.py loop --max-per-topic 3 --brief-limit 12 --gap-limit 20
```

This runs four phases:

```text
Phase 1: Fetch and score papers
Phase 2: Generate weekly brief
Phase 3: Generate research-gap analysis
Phase 4: Self-evolve profile
```

---

## Command Reference

| Command | Purpose | Example |
|---|---|---|
| `run` | Search and score new papers. | `python agent.py run --max-per-topic 3` |
| `brief` | Generate weekly research brief. | `python agent.py brief --limit 12` |
| `top` | Show top scored papers. | `python agent.py top --limit 10` |
| `feedback` | Update profile from user feedback. | `python agent.py feedback "Prioritize device papers."` |
| `gap` | Generate research-gap analysis. | `python agent.py gap --limit 20` |
| `evolve` | Self-critique and update profile. | `python agent.py evolve --limit 20` |
| `loop` | Run the full autonomous cycle. | `python agent.py loop --max-per-topic 3 --brief-limit 12 --gap-limit 20` |

---

## Generated Outputs

### SQLite Database

```text
memory.db
```

Stores:

- scored papers
- metadata
- scoring dimensions
- total score
- commentary
- recommended actions
- feedback history

### Reports Directory

```text
reports/
```

The agent creates this automatically.

Possible files:

```text
weekly_brief_YYYY-MM-DD.md
research_gap_analysis_YYYY-MM-DD.md
self_feedback_YYYY-MM-DD.md
```

### Profile Updates

The file below may be modified by feedback or self-evolution commands:

```text
research_profile.yaml
```

It is recommended to commit profile changes intentionally so you can track how your research strategy evolves over time.

---

## Database Design

The agent initializes a SQLite database with two tables:

### `papers`

Stores retrieved, scored, and ranked papers.

Key columns:

| Column | Description |
|---|---|
| `id` | Auto-incremented internal ID. |
| `unique_key` | Unique deduplication key based on paper URL or title. |
| `source` | Source of the paper, such as `OpenAlex` or `arXiv`. |
| `title` | Paper title. |
| `abstract` | Paper abstract. |
| `url` | DOI, landing page, arXiv URL, or OpenAlex ID. |
| `published_date` | Publication date. |
| `authors` | Up to several author names stored as a comma-separated string. |
| `cited_by_count` | Citation count when available. |
| `relevance` | LLM score from 0 to 10. |
| `novelty` | LLM score from 0 to 10. |
| `grant_potential` | LLM score from 0 to 10. |
| `collaboration_potential` | LLM score from 0 to 10. |
| `industry_potential` | LLM score from 0 to 10. |
| `teaching_public_value` | LLM score from 0 to 10. |
| `total_score` | Weighted total score. |
| `agent_commentary` | Short LLM-generated explanation. |
| `recommended_action` | Suggested next action. |
| `status` | Defaults to `new`. |
| `created_at` | Timestamp when saved. |

### `feedback`

Stores human or generated feedback used to evolve the research profile.

| Column | Description |
|---|---|
| `id` | Auto-incremented internal ID. |
| `feedback_text` | Feedback used to update the profile. |
| `created_at` | Timestamp when feedback was stored. |

---

## Scoring System

Each paper is scored on a 0–10 scale for six dimensions.

### 1. Relevance

Measures how directly the paper connects to your research profile.

High relevance examples:

- OLED device physics
- red-NIR TADF emitters
- lanthanide complexes for organic electronics
- organic semiconductor charge transport
- quantum chemistry for emitter design

Low relevance examples:

- general chemistry with no device connection
- unrelated biomedical papers
- materials papers with no organic electronics angle

### 2. Novelty

Measures whether the paper appears to introduce a new method, insight, material design, mechanism, or strategic direction.

### 3. Grant Potential

Measures whether the paper could support:

- proposal framing
- research gap development
- preliminary literature review
- strategic funding angle
- interdisciplinary grant concept

### 4. Collaboration Potential

Measures whether the paper suggests useful authors, institutions, or expertise areas for collaboration.

### 5. Industry Potential

Measures whether the paper has possible connection to:

- OLED display technology
- organic electronics manufacturing
- photonics
- biomedical light-emitting devices
- materials companies
- device engineering

### 6. Teaching/Public Value

Measures whether the paper could be used for:

- teaching examples
- student discussion
- public science communication
- LinkedIn posts
- research storytelling

---

## Recommended Action Labels

The scoring prompt restricts recommended actions to exactly one of the following:

| Action | Meaning |
|---|---|
| `read_now` | High-priority paper worth immediate attention. |
| `save_for_later` | Useful but not urgent. |
| `use_for_grant` | Strong fit for proposal development. |
| `contact_author` | Good basis for potential collaboration. |
| `use_for_teaching` | Useful for lectures, student examples, or education. |
| `use_for_linkedin` | Good for public-facing explanation or professional communication. |
| `ignore` | Low relevance or low strategic value. |

---

## Self-Evolution System

The self-evolution system has two levels:

### 1. Feedback-Based Evolution

Run:

```bash
python agent.py feedback "Your feedback here"
```

The LLM receives:

- the current YAML profile
- the user feedback
- rules for safe updating

It then returns updated YAML, which is sanitized before being saved.

### 2. Automatic Self-Evolution

Run:

```bash
python agent.py evolve
```

The agent:

1. Reviews curation statistics from the database.
2. Calculates patterns such as average score and action breakdown.
3. Looks at top papers.
4. Generates 3–6 sentences of profile-improvement feedback.
5. Saves that feedback to `reports/`.
6. Applies it to the YAML profile.

### Guard Rails

The profile sanitizer protects important research directions from being accidentally removed.

Protected topics include:

```text
OLED
TADF
organic semiconductor
quantum chemistry
red-NIR emission
lanthanide complex
charge transport
device degradation
```

The sanitizer also:

- ensures required weight keys are present
- normalizes weights to sum to approximately 1.0
- preserves key sections if the LLM drops them
- deduplicates negative filters
- removes invalid negative filters that overlap with core protected research topics
- deduplicates topics while preserving order

---

## Workflow Examples

### Example 1: Weekly Research Intelligence Routine

```bash
python agent.py run --max-per-topic 3
python agent.py top --limit 10
python agent.py brief --limit 12
python agent.py gap --limit 20
```

Recommended cadence:

- Run once per week.
- Review top papers manually.
- Use `feedback` to correct agent behavior.
- Commit useful changes to `research_profile.yaml`.

### Example 2: Grant Proposal Preparation

```bash
python agent.py run --max-per-topic 5
python agent.py gap --limit 30
python agent.py feedback "Increase grant_potential weighting and prioritize papers that connect red-NIR OLEDs with device fabrication and measurable biomedical relevance."
python agent.py brief --limit 15
```

Use this when preparing:

- grant proposals
- research concept notes
- collaboration emails
- strategic lab directions

### Example 3: Collaboration Discovery

```bash
python agent.py run --max-per-topic 4
python agent.py top --limit 20
python agent.py feedback "Prioritize papers with clear author or institution collaboration potential, especially in OLED materials, device fabrication, and quantum chemistry."
python agent.py evolve --limit 20
```

### Example 4: Full Autonomous Cycle

```bash
python agent.py loop --max-per-topic 3 --brief-limit 12 --gap-limit 20
```

Use this when you want a complete end-to-end cycle.

---

## Example Output

### Terminal Output During Scoring

```text
Searching OpenAlex: OLED
Searching arXiv: OLED
Found 42 unique papers.
Scoring: High-efficiency red-NIR TADF OLED emitters...
  Score=8.45 | Action=use_for_grant | Strong fit with red-NIR OLED device strategy...
```

### Example Top-Paper Display

```text
1. Example Paper Title
Score: 8.45 | Action: use_for_grant
Date: 2026-01-15 | Source: OpenAlex
Authors: A. Researcher, B. Scientist
Commentary: Strong strategic fit because...
URL: https://doi.org/...
```

### Example Report Files

```text
reports/weekly_brief_2026-05-06.md
reports/research_gap_analysis_2026-05-06.md
reports/self_feedback_2026-05-06.md
```

---

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'ollama'`

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install Ollama Python client directly:

```bash
pip install ollama
```

### Problem: `Is the Ollama service running?`

Start Ollama:

```bash
ollama serve
```

Then retry:

```bash
python test_ollama.py
```

### Problem: Model Not Found

If you see an error saying the model is not available:

```bash
ollama pull llama3.1:8b
```

If you use another model, update `.env`:

```text
OLLAMA_MODEL=your-model-name
```

### Problem: OpenAlex Request Fails

Possible causes:

- internet connection issue
- temporary OpenAlex API issue
- timeout
- invalid API key, if configured

The agent catches OpenAlex failures for each topic and continues running other searches.

### Problem: arXiv Request Fails

Possible causes:

- internet connection issue
- temporary arXiv API issue
- search query too broad or malformed

The agent catches arXiv failures for each topic and continues.

### Problem: No Papers Found

Try increasing the number of papers per topic:

```bash
python agent.py run --max-per-topic 10
```

Or edit `research_profile.yaml` to include more searchable terms.

For example, instead of:

```yaml
- device physics
```

use:

```yaml
- OLED device physics
- organic semiconductor charge transport
- TADF emitter device degradation
```

### Problem: Too Many Irrelevant Papers

Tighten your topics and negative filters.

Example feedback command:

```bash
python agent.py feedback "The results are too broad. Reduce emphasis on general chemistry and prioritize OLED device fabrication, red-NIR TADF, and organic semiconductor charge transport."
```

### Problem: JSON Parsing Fails During Scoring

The code includes a fallback. If the LLM output cannot be parsed as JSON, the paper receives default midpoint scores and is marked for manual review.

To reduce parsing failures:

- use a model with reliable instruction-following
- keep paper abstracts within context limits
- reduce temperature
- use the default `format="json"` behavior already configured in the code

### Problem: Profile Evolution Produces Bad YAML

The code attempts to clean fenced YAML and validate required top-level keys. If the update cannot be safely applied, it prints the model output and does not overwrite the profile.

Suggested fix:

```bash
python agent.py feedback "Make only a small update: increase grant_potential slightly and add 'red-NIR OLED device fabrication' as a topic."
```

---

## Suggested `.gitignore`

Before uploading to GitHub, create a `.gitignore` file to avoid committing local databases, reports, virtual environments, and secrets.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python

# Virtual environments
.venv/
venv/
env/

# Local environment variables
.env
.env.*
!.env.example

# SQLite database
memory.db
*.sqlite
*.sqlite3

# Generated reports
reports/

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/

# Logs
*.log
```

You may choose to commit selected reports if you want to show examples, but avoid committing private research notes by default.

---

## Suggested `.env.example`

For GitHub, consider adding:

```text
OLLAMA_MODEL=llama3.1:8b
OPENALEX_API_KEY=
```

Commit `.env.example`, but do not commit `.env`.

---

## Development Notes

### Code Organization

The current implementation is a single-file CLI application in `agent.py`. Major function groups include:

| Function Area | Representative Functions |
|---|---|
| Settings/profile loading | `load_settings`, `load_profile`, `save_profile` |
| Database initialization | `init_db` |
| Paper search | `search_openalex`, `search_arxiv` |
| Paper scoring | `score_paper`, `calculate_total` |
| Persistence | `save_scored_paper`, `get_top_papers` |
| Report generation | `generate_weekly_brief`, `generate_research_gap_analysis` |
| Profile evolution | `evolve_profile_from_feedback`, `auto_evolve`, `_sanitize_profile` |
| CLI routing | `main` |

### Design Philosophy

The project favors:

- local LLM inference
- transparent YAML configuration
- lightweight SQLite persistence
- Markdown outputs
- human-in-the-loop research strategy
- iterative profile evolution

### Why SQLite?

SQLite is appropriate because:

- no external database server is needed
- results are stored locally
- the schema is simple
- it is easy to inspect, back up, or migrate later

### Why Markdown Reports?

Markdown is suitable because:

- it is readable in GitHub
- it can be version controlled
- it can be converted into PDF, Word, slides, or web pages later
- it is easy to edit manually

---

## Roadmap Ideas

Possible future improvements:

### Search Improvements

- Add Semantic Scholar support.
- Add Crossref support.
- Add PubMed support for biomedical directions.
- Add Europe PMC support.
- Add author-specific search.
- Add institution-specific search.
- Add company-watchlist matching.

### Scoring Improvements

- Add confidence scores.
- Add reasoned scoring traces in a separate audit column.
- Add novelty comparison against previous database entries.
- Add penalty for papers matching negative filters.
- Add domain-specific scoring templates for OLED, TADF, quantum chemistry, and device fabrication.

### Database Improvements

- Add migration support.
- Add tags table.
- Add authors table.
- Add institutions table.
- Add many-to-many relationships between papers and topics.
- Add status workflow such as `new`, `read`, `cited`, `used_for_grant`, `contacted_author`.

### Reporting Improvements

- Generate HTML dashboards.
- Generate PowerPoint summaries.
- Generate BibTeX exports.
- Generate Zotero-compatible exports.
- Generate weekly email summaries.
- Generate grant-outline drafts.

### User Interface Improvements

- Add Streamlit dashboard.
- Add FastAPI backend.
- Add web-based paper review interface.
- Add command-line filters for source, score, action, and date.

### Automation Improvements

- Add scheduled weekly runs.
- Add GitHub Actions for non-LLM checks.
- Add local cron or Windows Task Scheduler examples.
- Add email notification after report generation.

---

## Limitations

This project is useful, but it should not replace expert judgment.

Important limitations:

1. **LLM scores are heuristic.** They are useful for triage, not final evaluation.
2. **Paper search depends on public API metadata.** Missing abstracts or metadata may affect scoring quality.
3. **arXiv and OpenAlex coverage differs by discipline.** Some OLED/device papers may be better represented in publisher databases not currently queried.
4. **The local model may produce imperfect JSON or YAML.** The code includes parsing and validation safeguards, but manual review is still important.
5. **Profile evolution should be reviewed.** The agent can update `research_profile.yaml`, but you should inspect changes before committing them.
6. **Negative filters need careful tuning.** Overly broad filters may suppress valuable papers.
7. **Citation counts may not be available from all sources.** arXiv entries are assigned `0` citation count in the current implementation.

---

## Security and Privacy

### Local Files

The following are local and may contain private research information:

```text
memory.db
reports/
research_profile.yaml
.env
```

Be careful before committing them to a public repository.

### Recommended Public Repository Policy

Safe to commit:

```text
agent.py
requirements.txt
setup_ollama.ps1
test_ollama.py
README.md
.env.example
.gitignore
```

Review before committing:

```text
research_profile.yaml
```

Usually do not commit:

```text
.env
memory.db
reports/
```

### API Keys

Do not commit your `.env` file.

If you use an OpenAlex API key, keep it private:

```text
OPENALEX_API_KEY=your_private_key_here
```

---

## GitHub Upload Guide

### 1. Initialize Git

```bash
git init
```

### 2. Add Files

```bash
git add agent.py requirements.txt setup_ollama.ps1 test_ollama.py research_profile.yaml README.md .gitignore
```

If you create `.env.example`, add it too:

```bash
git add .env.example
```

### 3. Commit

```bash
git commit -m "Initial commit: Research Evolution Agent"
```

### 4. Create a GitHub Repository

Create a new repository on GitHub, for example:

```text
research-evolution-agent
```

### 5. Connect Remote Repository

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/research-evolution-agent.git
git push -u origin main
```

---

## Testing Checklist

Before publishing, run:

```bash
python test_ollama.py
python agent.py --help
python agent.py run --max-per-topic 1
python agent.py top --limit 5
python agent.py brief --limit 5
python agent.py gap --limit 5
```

Expected checks:

- Ollama imports successfully.
- The model appears in `ollama list`.
- The test prompt receives a response.
- `memory.db` is created.
- Papers are retrieved or failures are gracefully printed.
- Scores are saved.
- Reports are created in `reports/`.

---

## Minimal Example Session

```bash
# Activate environment
source .venv/bin/activate

# Confirm local model works
python test_ollama.py

# Run discovery and scoring
python agent.py run --max-per-topic 2

# Inspect top results
python agent.py top --limit 5

# Create a weekly brief
python agent.py brief --limit 5

# Create gap analysis
python agent.py gap --limit 10

# Improve the agent
python agent.py feedback "Prioritize papers with device-level OLED validation and reduce broad unrelated materials papers."
```

Windows PowerShell equivalent:

```powershell
.\.venv\Scripts\Activate.ps1
python test_ollama.py
python agent.py run --max-per-topic 2
python agent.py top --limit 5
python agent.py brief --limit 5
python agent.py gap --limit 10
python agent.py feedback "Prioritize papers with device-level OLED validation and reduce broad unrelated materials papers."
```

---

## Example Research Strategy Use Case

A researcher focused on OLEDs and organic semiconductor devices may use the system weekly as follows:

1. Run `python agent.py run --max-per-topic 3` every Monday.
2. Review `python agent.py top --limit 10`.
3. Generate `python agent.py brief --limit 12`.
4. Convert the best opportunity into:
   - one paper-reading priority
   - one grant idea
   - one collaboration email
   - one LinkedIn post
5. Run `python agent.py feedback "..."` after manual review.
6. Commit meaningful profile updates.

This creates a long-term memory of how your research direction evolves.

---

## License

No license file is currently included.

Before publishing publicly, choose a license. Common options:

- MIT License for a permissive open-source project.
- Apache License 2.0 for a permissive license with explicit patent language.
- GPLv3 if you want derivative works to remain open source.
- Private repository if the project contains sensitive research strategy.

Suggested default for broad reuse:

```text
MIT License
```

Add a `LICENSE` file if you want others to clearly understand how they may use the code.

---

## Acknowledgements

This project uses:

- Ollama for local LLM inference.
- OpenAlex for scholarly metadata search.
- arXiv for preprint search.
- SQLite for local research memory.
- Python, PyYAML, Requests, python-dotenv, Ollama Python client, and Rich.

---

## Project Status

Current status: **working local prototype**.

The system is suitable for personal research intelligence workflows, strategic literature scanning, grant ideation, and research-profile evolution. It is not yet a polished multi-user platform, but it is structured clearly enough for extension into a dashboard, scheduled automation, or broader research assistant system.
