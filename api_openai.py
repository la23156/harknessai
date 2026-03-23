"""api_openai.py — All OpenAI API interactions: prompt construction and generation.

The key design principle: each generated "prompt" must be FULLY SELF-CONTAINED.
When the prompt is given to any LLM with no other context, the LLM should be able
to produce a complete Harkness discussion script, essay, or lesson plan.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TypedDict

import openai
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("OAL_MODEL", "gpt-4o")


class PromptItem(TypedDict):
    title: str
    prompt: str


def get_openai_client() -> openai.OpenAI:
    """Initialize and return OpenAI client using OAI_KEY from environment."""
    api_key = os.environ.get("OAI_KEY")
    if not api_key:
        raise EnvironmentError("OAI_KEY not set in environment.")
    return openai.OpenAI(api_key=api_key)


def fetch_web_context(query: str, client: openai.OpenAI) -> str:
    """Use OpenAI responses API with web search to fetch context. Fallback — never crashes."""
    try:
        response = client.responses.create(
            model=MODEL,
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        return block.text
        return ""
    except Exception as exc:
        print(f"[api_openai] fetch_web_context error: {exc}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Harkness method guide — embedded verbatim into every Harkness prompt
# ---------------------------------------------------------------------------

HARKNESS_METHOD_GUIDE = """
THE HARKNESS DISCUSSION METHOD
===============================
Harkness discussion is a student-led, collaborative inquiry method developed at
Phillips Exeter Academy. It is the cornerstone pedagogy at the Lawrenceville School
and other leading independent schools. Here is how it works:

SETUP: Students sit in an oval or rectangle (the "Harkness table") so every
participant can see every other participant. The teacher sits at the table as an
equal presence but does NOT moderate or call on students.

STUDENT ROLE: Students drive the conversation. They are expected to:
  - Reference the text directly (open books on the table, cite passages)
  - Build on, challenge, and extend each other's ideas
  - Ask follow-up questions of each other (not the teacher)
  - Tolerate productive silence — not every pause needs filling
  - Disagree respectfully and with textual evidence

TEACHER ROLE: The teacher is an observer and occasional catalyst. The teacher:
  - Opens with a single, rich question and then steps back
  - Maps the discussion (tracks who speaks, who builds on whom)
  - Intervenes ONLY to redirect a stalled discussion, push deeper, or surface
    a neglected thread — never to provide "the answer"
  - Closes the discussion with a synthesis move that helps students name what
    they collectively discovered

WHAT MAKES A GOOD HARKNESS QUESTION:
  - It is TEXT-ANCHORED: students must have the text open to answer it
  - It is GENUINELY CONTESTED: reasonable readers can disagree
  - It is GENERATIVE: it opens up multiple lines of inquiry, not one right path
  - It is ACCESSIBLE: every student can enter the conversation, but no student
    can exhaust it quickly
  - It avoids yes/no framing and avoids questions with obvious "right" answers

DISCUSSION ARC: A good Harkness session typically moves through:
  1. ENTRY — students orient to the question, offer initial readings
  2. EXPLORATION — competing interpretations surface, evidence is marshalled
  3. COMPLICATION — students discover tensions, contradictions, or layers
  4. SYNTHESIS — the group begins to articulate a collective (not unanimous) insight
"""


def _quotes_block(sourced_quotes: list[str]) -> str:
    """Format sourced quotes for embedding in prompts."""
    if not sourced_quotes:
        return "(No pre-sourced passages available — use your knowledge of the texts.)"
    lines = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sourced_quotes))
    return (
        "SOURCED PASSAGES — use these as textual anchors. Where a passage is\n"
        "usable, embed it verbatim with its attribution:\n\n"
        f"{lines}"
    )


# ---------------------------------------------------------------------------
# Generators — each produces 5 SELF-CONTAINED prompts
# ---------------------------------------------------------------------------

def generate_harkness_prompts(
    course: dict,
    sourced_quotes: list[str],
    client: openai.OpenAI,
) -> list[PromptItem]:
    """Generate FIVE self-contained Harkness discussion script prompts."""

    system_msg = f"""You are a meta-prompt engineer. Your job is to produce FIVE standalone LLM prompts.

Each prompt you write will later be given — BY ITSELF, with no other context — to an LLM.
That LLM must be able to produce a complete, realistic Harkness discussion DIALOGUE SCRIPT
from the prompt alone — an actual scripted conversation between named students sitting around
the Harkness table, with the teacher occasionally intervening.

The output should read like a play script or transcript: real student voices debating,
agreeing, disagreeing, citing the text, building on each other's ideas. NOT a teacher
planning document. NOT bullet points of discussion goals. An actual dialogue.

Therefore each prompt must contain ALL necessary context: a thorough explanation of the
Harkness method (so the LLM understands the pedagogy and can write realistic dialogue),
the course context, the specific text/passage focus, sourced quotes, and exact output
format requested.

COURSE INFORMATION (embed relevant parts into each prompt):
  Code: {course['code']}
  Title: {course['title']}
  Description: {course['description']}
  Tags: {', '.join(course.get('tags', []))}

{_quotes_block(sourced_quotes)}

HARKNESS METHOD GUIDE (embed this into each prompt so the receiving LLM understands the method
and can write dialogue that authentically follows it):
{HARKNESS_METHOD_GUIDE}
"""

    user_msg = """Write FIVE self-contained prompts. Each prompt will be given to an LLM on its own
to generate a Harkness discussion DIALOGUE SCRIPT. Each prompt must target a DIFFERENT text,
theme, or moment from the course.

EACH PROMPT MUST INCLUDE ALL OF THE FOLLOWING EMBEDDED WITHIN IT:
1. A thorough description of the Harkness discussion method (use the guide above) so the
   LLM understands how students and teacher behave in this format
2. The school context (Lawrenceville School, boarding school, grades 9-12, strong readers,
   undergraduate seminar level)
3. The specific course information
4. The specific text, scene, chapter, or poem being discussed
5. Relevant sourced quotes/passages embedded verbatim for students to cite in dialogue
6. Instructions telling the LLM to produce an ACTUAL DIALOGUE SCRIPT in this format:

   The script must be a realistic 20-30 minute Harkness table discussion written as a
   transcript/play script with:
   - 6-8 named students (use first names like "Maya", "James", "Sofia", etc.) with
     distinct voices and perspectives
   - The TEACHER who opens with a single rich question, then steps back and only
     intervenes 2-3 times to redirect or deepen
   - Students must quote and reference the text directly in their dialogue (open books
     on the table), cite specific passages, page numbers, or act/scene/line references
   - The conversation must show: students building on each other's ideas, at least one
     genuine interpretive disagreement, moments of productive silence or uncertainty,
     a student changing their mind or refining their position
   - The discussion should follow a natural arc: entry/orientation, exploration of
     competing readings, complication/tension, and movement toward (but not necessarily
     arriving at) synthesis
   - Stage directions in brackets for key moments: [pause], [flipping pages],
     [nodding], [turning to address another student], etc.
   - The dialogue should feel like real high school students who are smart and engaged
     but still teenagers — not like professors

Return your response as a JSON array of exactly 5 objects, each with keys "title" (short label)
and "prompt" (the full self-contained prompt text). Return ONLY valid JSON, no markdown fences."""

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=16000,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _parse_json_items(raw, "Harkness Discussion")


def generate_essay_prompts(
    course: dict,
    sourced_quotes: list[str],
    client: openai.OpenAI,
) -> list[PromptItem]:
    """Generate FIVE self-contained essay prompts."""

    system_msg = f"""You are a meta-prompt engineer. Your job is to produce FIVE standalone LLM prompts.

Each prompt you write will later be given — BY ITSELF, with no other context — to an LLM
acting as a student writer. That LLM must be able to produce a complete, polished literary
essay from the prompt alone. Therefore each prompt must contain ALL necessary context:
the course description, the specific text focus, exact passages/quotes to engage with,
the thesis to argue, structural guidance, and voice/register instructions.

Be EXTREMELY specific. Do not leave room for vague or generic responses. Name the exact
passages the LLM must engage with. The LLM receiving your prompt should know precisely
what position to take and what evidence to use.

COURSE INFORMATION (embed relevant parts into each prompt):
  Code: {course['code']}
  Title: {course['title']}
  Description: {course['description']}
  Tags: {', '.join(course.get('tags', []))}

{_quotes_block(sourced_quotes)}
"""

    user_msg = """Write FIVE self-contained essay prompts. Each prompt will be given to an LLM on its own
to generate a literary essay. Each prompt must address a DIFFERENT text, theme, or analytical
angle from the course.

EACH PROMPT MUST INCLUDE ALL OF THE FOLLOWING EMBEDDED WITHIN IT:
1. The course context (one sentence situating the essay within the course)
2. ESSAY TYPE declaration (argumentative, comparative, close-reading, etc.)
3. CENTRAL ARGUMENT TASK — the exact thesis to argue (be specific, not open-ended)
4. REQUIRED TEXTUAL EVIDENCE — at least 3 specific passages with quotes embedded verbatim,
   including act/scene/line or chapter/page references
5. COUNTERARGUMENT REQUIREMENT — name the opposing interpretation to address and rebut
6. STRUCTURAL GUIDANCE — paragraph count, word target (600-1000 words), required sections
7. VOICE AND REGISTER — formal academic, personal-reflective, journalistic, etc.
8. Each prompt should be written as a single cohesive paragraph of instructions, not bullet points

Return your response as a JSON array of exactly 5 objects, each with keys "title" (short label)
and "prompt" (the full self-contained prompt text). Return ONLY valid JSON, no markdown fences."""

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=8000,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _parse_json_items(raw, "Essay Prompt")


def generate_lesson_plan_prompts(
    course: dict,
    sourced_quotes: list[str],
    client: openai.OpenAI,
) -> list[PromptItem]:
    """Generate FIVE self-contained lesson plan prompts."""

    system_msg = f"""You are a meta-prompt engineer. Your job is to produce FIVE standalone LLM prompts.

Each prompt you write will later be given — BY ITSELF, with no other context — to an LLM
acting as an experienced English teacher. That LLM must be able to produce a complete,
ready-to-use 45-60 minute class session plan from the prompt alone. Therefore each prompt
must contain ALL necessary context: the school and course information, the specific narrow
topic for that single class session, relevant sourced quotes/passages, and the exact
output format requested.

CRITICAL: Each lesson plan targets ONE specific 45-60 minute class session on ONE narrow
topic — a single scene, poem, chapter, or concept. NOT a whole unit. NOT a multi-day plan.
A single day. Be granular about timing.

COURSE INFORMATION (embed relevant parts into each prompt):
  Code: {course['code']}
  Title: {course['title']}
  Description: {course['description']}
  Tags: {', '.join(course.get('tags', []))}

{_quotes_block(sourced_quotes)}
"""

    user_msg = """Write FIVE self-contained lesson plan prompts. Each prompt will be given to an LLM on its
own to generate a complete class session plan. Each prompt must target a DIFFERENT specific,
narrow topic from the course (a single scene, poem, chapter, or concept).

EACH PROMPT MUST INCLUDE ALL OF THE FOLLOWING EMBEDDED WITHIN IT:
1. School context (Lawrenceville School, boarding school, grades 9-12, capable readers,
   undergraduate seminar level)
2. The course information
3. The specific narrow session focus (exact text, chapter, scene, poem, or concept with
   line/page references)
4. Relevant sourced quotes/passages embedded verbatim
5. Instructions for the LLM to produce a plan with these sections:
   A) SESSION FOCUS — the exact narrow topic with references
   B) LEARNING OBJECTIVES (3-4 observable outcomes)
   C) ENTRY ACTIVITY (5-8 min) — specific warm-up tied to the focus
   D) DIRECT INSTRUCTION / MINI-LESSON (10 min) — what the teacher teaches, referencing
      specific passage or technique
   E) CLOSE READING / DISCUSSION ACTIVITY (20-25 min) — central activity with exact
      passages, method (Harkness, Socratic, pair-share, etc.), and 2-3 lens questions
   F) ASSESSMENT / EXIT ACTIVITY (5-10 min) — exit ticket, written response, etc.
   G) DIFFERENTIATION NOTES — one scaffolding suggestion, one extension suggestion
   H) MATERIALS LIST — texts, handouts, multimedia needed

Return your response as a JSON array of exactly 5 objects, each with keys "title"
(e.g. "Lesson: Macbeth Act II Scene 2 — The Dagger Soliloquy") and "prompt"
(the full self-contained prompt text). Return ONLY valid JSON, no markdown fences."""

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=8000,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _parse_json_items(raw, "Lesson Plan")


# ---------------------------------------------------------------------------
# JSON parser — much more robust than text splitting
# ---------------------------------------------------------------------------

def _parse_json_items(raw: str, fallback_prefix: str) -> list[PromptItem]:
    """Parse GPT JSON output into a list of 5 PromptItem dicts.

    Handles markdown fences, truncated JSON, and falls back to regex extraction.
    """
    import re

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # --- Attempt 1: direct JSON parse ---
    items = _try_json_parse(cleaned, fallback_prefix)
    if items:
        return items

    # --- Attempt 2: find JSON array in text ---
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        items = _try_json_parse(match.group(), fallback_prefix)
        if items:
            return items

    # --- Attempt 3: repair truncated JSON ---
    # GPT often runs out of tokens mid-array, producing: [{ ... }, { ... }, { ...
    # Try closing the last string, object, and array
    items = _try_repair_truncated_json(cleaned, fallback_prefix)
    if items:
        return items

    # --- Attempt 4: regex extraction of title/prompt pairs ---
    print(f"[api_openai] WARNING: JSON parse failed for {fallback_prefix}, using regex extraction", file=sys.stderr)
    return _regex_extract_items(cleaned, fallback_prefix)


def _try_json_parse(text: str, fallback_prefix: str) -> list[PromptItem] | None:
    """Try to parse text as a JSON array of {title, prompt} objects."""
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            items: list[PromptItem] = []
            for item in data[:5]:
                if isinstance(item, dict):
                    items.append(PromptItem(
                        title=item.get("title", f"{fallback_prefix} {len(items)+1}"),
                        prompt=item.get("prompt", ""),
                    ))
            if any(len(i["prompt"]) > 50 for i in items):
                while len(items) < 5:
                    items.append(PromptItem(title=f"{fallback_prefix} {len(items)+1}", prompt=""))
                return items[:5]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _try_repair_truncated_json(text: str, fallback_prefix: str) -> list[PromptItem] | None:
    """Try to repair JSON that was truncated mid-generation."""
    import re
    # Find where the array starts
    start = text.find("[")
    if start == -1:
        return None

    fragment = text[start:]

    # Try progressively aggressive repairs
    repairs = [
        fragment + '"}]',         # truncated mid-string value
        fragment + '"}]',
        fragment + '" }]',
        fragment + '"}}\n]',
        fragment + '\n}]',
        fragment + ']',
    ]
    for attempt in repairs:
        items = _try_json_parse(attempt, fallback_prefix)
        if items:
            print(f"[api_openai] Repaired truncated JSON for {fallback_prefix} ({len(items)} items recovered)", file=sys.stderr)
            return items

    # More aggressive: find all complete objects and wrap in array
    objects = re.findall(r'\{\s*"title"\s*:\s*"[^"]*"\s*,\s*"prompt"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', fragment, re.DOTALL)
    if objects:
        try:
            arr_text = "[" + ",".join(objects) + "]"
            items = _try_json_parse(arr_text, fallback_prefix)
            if items:
                print(f"[api_openai] Extracted {len(items)} complete JSON objects for {fallback_prefix}", file=sys.stderr)
                return items
        except Exception:
            pass

    return None


def _regex_extract_items(raw: str, fallback_prefix: str) -> list[PromptItem]:
    """Last-resort: extract prompt items using regex and text patterns."""
    import re
    items: list[PromptItem] = []

    # Try to find "title": "..." and "prompt": "..." pairs
    title_pattern = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
    prompt_pattern = re.compile(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)

    titles = title_pattern.findall(raw)
    prompts = prompt_pattern.findall(raw)

    if titles and prompts:
        for i in range(min(len(titles), len(prompts), 5)):
            # Unescape JSON string escapes
            title = titles[i].replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            prompt = prompts[i].replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            items.append(PromptItem(title=title.strip(), prompt=prompt.strip()))

    if not items:
        # Final fallback: split on --- or ### or numbered headers
        sections = re.split(r'\n(?=---|\#{2,3}\s|\d+\.\s+\*\*)', raw)
        sections = [s.strip().lstrip("-#").strip() for s in sections if s.strip() and len(s.strip()) > 50]

        for i, section in enumerate(sections[:5]):
            lines = section.splitlines()
            title = f"{fallback_prefix} {i + 1}"
            body_start = 0

            for j, line in enumerate(lines[:5]):
                stripped = line.strip()
                low = stripped.lower()
                if low.startswith("title:") or low.startswith("**title"):
                    title = stripped.split(":", 1)[-1].strip().strip("*").strip()
                    body_start = j + 1
                    break
                elif j == 0:
                    candidate = stripped.lstrip("0123456789.) ").strip("*#").strip()
                    if candidate and len(candidate) < 120:
                        title = candidate
                        body_start = j + 1
                        break

            prompt_text = "\n".join(lines[body_start:]).strip()
            items.append(PromptItem(title=title, prompt=prompt_text))

    while len(items) < 5:
        items.append(PromptItem(title=f"{fallback_prefix} {len(items)+1}", prompt=""))
    return items[:5]
