# HarknessAI Prompt Catalogue Generator

A Python tool that generates high-quality, self-contained LLM prompts for English literature courses. For each course in the catalogue, it produces three sets of five prompts:

1. **Harkness Discussion Scripts** -- prompts that generate realistic student dialogue around a Harkness table
2. **Essay Prompts** -- detailed, thesis-specific prompts that direct an LLM to write literary essays
3. **Teacher Lesson Plans** -- prompts that generate complete 45-60 minute class session plans

Every generated prompt is **fully self-contained**: you can hand any single prompt to an LLM with zero additional context and get a complete, usable output.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Output Format](#output-format)
- [Prompt Design](#prompt-design)
- [Architecture](#architecture)
- [Error Handling](#error-handling)

---

## Overview

The generator processes 64 English courses from the Lawrenceville School catalogue. For each course, it:

1. Identifies the key literary texts (books, plays, poems) from the course description
2. Searches for real quotes, passages, and scholarly context using the **Exa API**
3. Uses **OpenAI** to generate 15 self-contained prompts (5 Harkness + 5 essay + 5 lesson plan)
4. Saves structured JSON output organized by run number

The sourced passages ground every prompt in specific textual evidence -- page numbers, act/scene/line references, and verbatim quotes -- so the downstream LLM outputs are anchored in real literary content rather than vague generalities.

---

## How It Works

```
english_courses.py          Exa API                    OpenAI API
       |                      |                            |
       v                      v                            v
  64 courses ──> Extract ──> Search for ──> Generate 15 ──> Save to
  with metadata  texts &     quotes &       self-contained  output/run_NNN/
                 authors     passages       prompts         {CODE}.json
```

### Pipeline per course:

1. **Extract Texts** -- Parse course description and tags to identify books, authors, plays, and poems (up to 3 per course)
2. **Fetch Sourced Content** -- Query Exa for passages, quotes, themes, and historical context for each identified work
3. **Fallback** -- If Exa returns fewer than 3 usable quotes, fall back to OpenAI web search
4. **Generate Prompts** -- Call OpenAI to produce 5 prompts for each of the 3 types (Harkness, essay, lesson plan)
5. **Save Output** -- Write JSON to `output/run_NNN/{COURSE_CODE}.json`

---

## Project Structure

```
harknessai_prompts_from_catalogue/
├── english_courses.py      # Source data: 64 course definitions (DO NOT MODIFY)
├── api_exa.py              # Exa API client -- search and content retrieval
├── api_openai.py           # OpenAI client -- prompt generation + Harkness method guide
├── main.py                 # Orchestrator -- iterates courses, coordinates APIs, saves output
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore
├── instructions.txt        # Original build specification
├── README.md               # This file
└── output/                 # Auto-created; one subfolder per run
    ├── run_001/
    │   ├── HU201.json
    │   ├── EN301.json
    │   ├── EN421.json
    │   └── ...
    ├── run_002/
    │   └── ...
    └── ...
```

Each run of `main.py` creates a new numbered subdirectory (`run_001`, `run_002`, etc.) so outputs from different runs are never overwritten.

---

## Setup

### 1. Clone and enter the project

```bash
cd harknessai_prompts_from_catalogue
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv-harknessai-prompts-from-catalogue
source .venv-harknessai-prompts-from-catalogue/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `openai` -- OpenAI API client for prompt generation and web search fallback
- `exa-py` -- Exa API client for passage/quote sourcing
- `python-dotenv` -- Environment variable loading from `.env`

### 4. Configure API keys

Create a `.env` file in the project root:

```env
EXA_API_KEY=your-exa-api-key-here
OAI_KEY=your-openai-api-key-here
OAL_MODEL=gpt-4o
```

| Variable | Description |
|----------|-------------|
| `EXA_API_KEY` | API key for [Exa](https://exa.ai) (neural search for sourcing quotes) |
| `OAI_KEY` | API key for [OpenAI](https://platform.openai.com) (prompt generation) |
| `OAL_MODEL` | OpenAI model to use (default: `gpt-4o`) |

---

## Usage

### Process all 64 courses

```bash
python main.py
```

This takes a while (~2-3 minutes per course due to API calls and rate limiting). Progress is printed to stderr:

```
[1/64] Processing HU201: Humanities - English
  [HU201] Extracting texts...
  [HU201] Generating Harkness prompts...
  [HU201] Generating essay prompts...
  [HU201] Generating lesson plan prompts...
Generated: HU201 — Humanities - English

[2/64] Processing EN301: English III
...
```

### Process a single course (for testing)

```bash
python main.py --course EN301
```

### Output location

```
output/run_001/EN301.json
output/run_001/EN421.json
...
```

Each run auto-increments: `run_001`, `run_002`, `run_003`, etc.

---

## Output Format

Each course produces a single JSON file with this structure:

```json
{
  "course_code": "EN301",
  "course_title": "English III",
  "generated_at": "2026-03-23T14:30:00.000000+00:00",
  "sourced_quotes": [
    "\"Tomorrow, and tomorrow, and tomorrow / Creeps in this petty pace...\" — Macbeth, Shakespeare (source: folger.edu)",
    "..."
  ],
  "harkness_discussion_prompts": [
    {
      "title": "Macbeth - Dagger Soliloquy",
      "prompt": "You are to write a realistic Harkness discussion DIALOGUE SCRIPT for EN301..."
    }
  ],
  "essay_prompts": [
    {
      "title": "Macbeth: Power Corrupts Moral Order",
      "prompt": "In EN301 English III, write a 5-paragraph argumentative literary essay..."
    }
  ],
  "lesson_plan_prompts": [
    {
      "title": "Lesson: Macbeth Act II Scene 1 — Banquo and Fleance",
      "prompt": "You are an experienced English teacher at Lawrenceville School..."
    }
  ]
}
```

Each array contains exactly **5 prompt objects** with `title` and `prompt` keys.

Optional keys that may appear:
- `"exa_fallback_used": true` -- Exa returned insufficient results; OpenAI web search was used
- `"errors": { "harkness_discussion_prompts": "error message" }` -- partial failure for one prompt type

---

## Prompt Design

### Design Principle: Self-Contained Prompts

Every prompt includes all context needed for an LLM to produce complete output with no other input. This means each prompt embeds:

- The course code, title, and full description
- The school context (Lawrenceville School, boarding school, grades 9-12)
- The specific text, scene, chapter, or poem being addressed
- Sourced quotes and passages with attribution
- Detailed output format instructions

### Harkness Discussion Prompts

Each prompt instructs an LLM to generate a **realistic dialogue script** -- an actual conversation between 6-8 named students sitting around a Harkness table. The prompt embeds the full Harkness discussion method guide so the LLM understands:

- **Student role**: students drive the conversation, cite text directly, build on and challenge each other
- **Teacher role**: opens with one rich question, then observes; intervenes only 2-3 times to redirect or deepen
- **Discussion arc**: entry/orientation, exploration of competing readings, complication/tension, synthesis
- **What makes a good Harkness question**: text-anchored, genuinely contested, generative, accessible

The output is formatted as a play script with stage directions (`[pause]`, `[flipping pages]`, `[turning to Sofia]`), not a teacher planning document.

### Essay Prompts

Each prompt directs an LLM (acting as a student writer) to produce a literary essay. Prompts specify:

- Essay type (argumentative, comparative, close-reading, etc.)
- The exact thesis to argue
- At least 3 specific passages with verbatim quotes and references
- A named counterargument to address and rebut
- Structural guidance (paragraph count, 600-1000 word target)
- Voice and register (formal academic, personal-reflective, etc.)

### Lesson Plan Prompts

Each prompt targets **one specific 45-60 minute class session** on one narrow topic (a single scene, poem, chapter, or concept). The output includes:

- Session focus with exact text references
- Learning objectives (3-4 observable outcomes)
- Entry activity (5-8 min warm-up)
- Mini-lesson (10 min direct instruction)
- Close reading / discussion activity (20-25 min)
- Exit assessment (5-10 min)
- Differentiation notes (scaffolding + extension)
- Materials list

---

## Architecture

### File Responsibilities

| File | Role | Imports from |
|------|------|-------------|
| `english_courses.py` | Data source (64 courses) | Nothing |
| `api_exa.py` | Exa search and quote extraction | Nothing (stateless) |
| `api_openai.py` | OpenAI prompt generation | Nothing (stateless) |
| `main.py` | Orchestration only | `english_courses`, `api_exa`, `api_openai` |

- `api_exa.py` and `api_openai.py` are **stateless** -- no globals, no cross-imports between them
- All course-specific logic flows in as parameters, not hardcoded
- Only `main.py` imports from `english_courses.py`

### Text Identification

`main.py` contains a `KNOWN_TEXTS` mapping of ~60 literary works commonly taught in English courses. The `extract_texts_from_course()` function matches course descriptions/tags against this list to identify up to 3 works per course for Exa sourcing.

### JSON Parsing

GPT output is requested as a JSON array but doesn't always comply (especially for long Harkness prompts). The parser in `api_openai.py` uses a 4-stage fallback:

1. **Direct JSON parse** -- try parsing the cleaned response as-is
2. **Array extraction** -- find a JSON array `[...]` within surrounding text
3. **Truncated JSON repair** -- close dangling strings/objects/arrays from token-limit truncation
4. **Regex extraction** -- extract `"title"` and `"prompt"` values directly from raw text

---

## Error Handling

- Each course is wrapped in a try/except -- a single course failure doesn't crash the full run
- If Exa API is unavailable, OpenAI web search is used as a fallback for sourcing
- If one prompt type fails (e.g., lesson plans), the other types are still saved
- Partial failures are recorded in an `"errors"` key in the output JSON
- A 2-second sleep between courses prevents API rate limiting
