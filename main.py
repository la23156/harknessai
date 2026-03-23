"""main.py — Orchestrator: iterates courses, calls APIs, saves output JSON files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from english_courses import courses  # noqa: E402
from api_exa import search_book_passages, search_work_context, extract_usable_quotes
from api_openai import (
    get_openai_client,
    fetch_web_context,
    generate_harkness_prompts,
    generate_essay_prompts,
    generate_lesson_plan_prompts,
)

OUTPUT_BASE = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Known text extraction helpers
# ---------------------------------------------------------------------------

# Heuristic map: terms commonly found in descriptions/tags → (title, author)
KNOWN_TEXTS: list[tuple[str, str]] = [
    # Shakespeare
    ("Macbeth", "Shakespeare"),
    ("Hamlet", "Shakespeare"),
    ("King Lear", "Shakespeare"),
    ("Othello", "Shakespeare"),
    ("A Midsummer Night's Dream", "Shakespeare"),
    ("Much Ado About Nothing", "Shakespeare"),
    ("The Tempest", "Shakespeare"),
    ("Twelfth Night", "Shakespeare"),
    ("Romeo and Juliet", "Shakespeare"),
    ("The Merchant of Venice", "Shakespeare"),
    # American literature
    ("Invisible Man", "Ralph Ellison"),
    ("Song of Solomon", "Toni Morrison"),
    ("Beloved", "Toni Morrison"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("Moby Dick", "Herman Melville"),
    ("Moby-Dick", "Herman Melville"),
    ("Their Eyes Were Watching God", "Zora Neale Hurston"),
    ("The Adventures of Huckleberry Finn", "Mark Twain"),
    ("Adventures of Huckleberry Finn", "Mark Twain"),
    ("The Scarlet Letter", "Nathaniel Hawthorne"),
    ("The Sun Also Rises", "Ernest Hemingway"),
    ("A Farewell to Arms", "Ernest Hemingway"),
    ("For Whom the Bell Tolls", "Ernest Hemingway"),
    ("The Old Man and the Sea", "Ernest Hemingway"),
    ("On the Road", "Jack Kerouac"),
    ("Howl", "Allen Ginsberg"),
    ("Paterson", "William Carlos Williams"),
    ("Leaves of Grass", "Walt Whitman"),
    # British literature
    ("Heart of Darkness", "Joseph Conrad"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Pride and Prejudice", "Jane Austen"),
    ("Wuthering Heights", "Emily Brontë"),
    ("Great Expectations", "Charles Dickens"),
    ("Middlemarch", "George Eliot"),
    ("Ulysses", "James Joyce"),
    ("Mrs Dalloway", "Virginia Woolf"),
    ("To the Lighthouse", "Virginia Woolf"),
    # Drama
    ("A Raisin in the Sun", "Lorraine Hansberry"),
    ("Death of a Salesman", "Arthur Miller"),
    ("The Cherry Orchard", "Anton Chekhov"),
    ("Uncle Vanya", "Anton Chekhov"),
    ("The Three Sisters", "Anton Chekhov"),
    ("The Seagull", "Anton Chekhov"),
    # Short fiction / essays authors (searched by author + course context)
    ("Sonny's Blues", "James Baldwin"),
    ("The Fire Next Time", "James Baldwin"),
    ("A Good Man Is Hard to Find", "Flannery O'Connor"),
    ("Everything That Rises Must Converge", "Flannery O'Connor"),
    ("The Canterbury Tales", "Geoffrey Chaucer"),
    # Science fiction
    ("Kindred", "Octavia Butler"),
    ("Parable of the Sower", "Octavia Butler"),
    ("1984", "George Orwell"),
    ("Brave New World", "Aldous Huxley"),
    ("The Handmaid's Tale", "Margaret Atwood"),
    ("Fahrenheit 451", "Ray Bradbury"),
    # Other
    ("The Bible", ""),
    ("Wizard of Oz", "L. Frank Baum"),
    ("The Wonderful Wizard of Oz", "L. Frank Baum"),
]


def extract_texts_from_course(course: dict) -> list[tuple[str, str]]:
    """Return up to 3 (title, author) tuples from course description and tags."""
    combined = (course.get("description", "") + " " + " ".join(course.get("tags", []))).lower()

    found: list[tuple[str, str]] = []
    for title, author in KNOWN_TEXTS:
        if title.lower() in combined or (author and author.split()[-1].lower() in combined):
            found.append((title, author))
        if len(found) >= 3:
            break

    return found


def get_topics_from_course(course: dict) -> list[str]:
    """Extract 2-3 topic keywords from course tags."""
    tags = course.get("tags", [])
    # Filter out very generic tags
    skip = {"yearlong", "ncaa", "honors"}
    topics = [t for t in tags if t.lower() not in skip]
    return topics[:3] if topics else ["themes", "character", "plot"]


# ---------------------------------------------------------------------------
# Core per-course processing
# ---------------------------------------------------------------------------

def process_course(course: dict, openai_client) -> dict:
    """Run the full pipeline for one course and return the output dict."""
    code = course["code"]
    title = course["title"]

    print(f"  [{code}] Extracting texts...", file=sys.stderr)
    texts = extract_texts_from_course(course)
    topics = get_topics_from_course(course)

    # --- Step 2: Fetch Exa content ---
    all_exa_results: list[dict] = []
    exa_available = True

    for work_title, author in texts:
        for topic in topics[:2]:  # limit to 2 topics per work to control cost
            results = search_book_passages(work_title, author, topic)
            all_exa_results.extend(results)
        for ctx_type in ("themes", "historical context"):
            results = search_work_context(work_title, author, ctx_type)
            all_exa_results.extend(results)

    sourced_quotes = extract_usable_quotes(all_exa_results, max_quotes=8)

    # Fallback: if fewer than 3 quotes, try OpenAI web search
    if len(sourced_quotes) < 3:
        if not texts:
            exa_available = False
            fallback_query = (
                f"key quotes and passages from texts in: {course['description'][:200]}"
            )
        else:
            exa_available = False
            fallback_query = " and ".join(
                f'"{t}" by {a}' for t, a in texts[:2] if a
            ) or course["description"][:200]
            fallback_query += " key quotes passages literary analysis"

        print(f"  [{code}] Exa returned <3 quotes; falling back to OpenAI web search.", file=sys.stderr)
        web_text = fetch_web_context(fallback_query, openai_client)
        if web_text:
            # Extract sentences that look like quotes
            sentences = re.split(r'(?<=[.!?])\s+', web_text)
            for s in sentences:
                if len(s) > 40 and ('"' in s or "'" in s):
                    sourced_quotes.append(s[:300])
                if len(sourced_quotes) >= 6:
                    break

    # --- Step 3: Generate prompts ---
    harkness: list[dict] = []
    essays: list[dict] = []
    lessons: list[dict] = []
    errors: dict[str, str] = {}

    print(f"  [{code}] Generating Harkness prompts...", file=sys.stderr)
    try:
        harkness = generate_harkness_prompts(course, sourced_quotes, openai_client)
    except Exception as exc:
        errors["harkness_discussion_prompts"] = str(exc)
        print(f"  [{code}] ERROR generating Harkness prompts: {exc}", file=sys.stderr)

    print(f"  [{code}] Generating essay prompts...", file=sys.stderr)
    try:
        essays = generate_essay_prompts(course, sourced_quotes, openai_client)
    except Exception as exc:
        errors["essay_prompts"] = str(exc)
        print(f"  [{code}] ERROR generating essay prompts: {exc}", file=sys.stderr)

    print(f"  [{code}] Generating lesson plan prompts...", file=sys.stderr)
    try:
        lessons = generate_lesson_plan_prompts(course, sourced_quotes, openai_client)
    except Exception as exc:
        errors["lesson_plan_prompts"] = str(exc)
        print(f"  [{code}] ERROR generating lesson plan prompts: {exc}", file=sys.stderr)

    output: dict = {
        "course_code": code,
        "course_title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sourced_quotes": sourced_quotes,
        "harkness_discussion_prompts": harkness,
        "essay_prompts": essays,
        "lesson_plan_prompts": lessons,
    }
    if not exa_available:
        output["exa_fallback_used"] = True
    if errors:
        output["errors"] = errors

    return output


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def get_next_run_dir() -> Path:
    """Determine the next run directory: output/run_001, run_002, etc."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p for p in OUTPUT_BASE.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    if existing:
        last_num = int(existing[-1].name.split("_")[1])
        next_num = last_num + 1
    else:
        next_num = 1
    run_dir = OUTPUT_BASE / f"run_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_output(data: dict, run_dir: Path) -> Path:
    """Write course output JSON into the run directory."""
    path = run_dir / f"{data['course_code']}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HarknessAI prompt catalogue.")
    parser.add_argument(
        "--course",
        metavar="CODE",
        help="Process a single course by code (e.g. EN301). Omit to run all.",
    )
    args = parser.parse_args()

    openai_client = get_openai_client()
    run_dir = get_next_run_dir()
    print(f"Run directory: {run_dir.resolve()}", file=sys.stderr)

    target_courses: list[dict]
    if args.course:
        target_courses = [c for c in courses if c["code"] == args.course]
        if not target_courses:
            print(f"ERROR: Course code '{args.course}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        target_courses = list(courses)

    total = len(target_courses)
    for idx, course in enumerate(target_courses, start=1):
        code = course["code"]
        title = course["title"]
        print(f"\n[{idx}/{total}] Processing {code}: {title}", file=sys.stderr)
        try:
            data = process_course(course, openai_client)
            path = save_output(data, run_dir)
            print(f"Generated: {code} — {title}")
        except Exception as exc:
            print(f"  [{code}] FATAL ERROR: {exc}", file=sys.stderr)
            # Write partial error JSON so the run is resumable
            error_data = {
                "course_code": code,
                "course_title": title,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            try:
                save_output(error_data, run_dir)
            except Exception:
                pass

        if idx < total:
            time.sleep(2)  # rate-limit buffer between courses

    print(f"\nDone. Output written to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
