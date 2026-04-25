import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


load_dotenv()

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))
GEN_MODEL = os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash")
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "index"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_SOURCE_CHARS = int(os.getenv("WIKI_UPDATE_MAX_SOURCE_CHARS", "6000"))


def normalize_heading(text: str) -> str:
    normalized = (text or "").casefold().strip()
    normalized = re.sub(r"^[0-9]+[.\)]\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_gemini_api_key() -> str | None:
    candidates = [
        ("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "").strip()),
        ("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", "").strip()),
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "").strip()),
    ]
    for name, value in candidates:
        if value:
            if name == "OPENAI_API_KEY":
                print("[INFO] Using Gemini key from OPENAI_API_KEY. GEMINI_API_KEY is preferred.")
            return value
    print("[ERROR] Gemini API key not found")
    return None


def build_client():
    if genai is None:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai")
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing Gemini API key")
    return genai.Client(api_key=api_key)


def load_index(index_dir: Path):
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Index files not found. Run build_index.py first. Missing: {index_path} or {metadata_path}")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata


def embed_query(client, text: str):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIM),
    )
    vector = np.array([result.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(vector)
    return vector


def retrieve_hits(index, metadata, query_text: str, top_k: int, wiki_only: bool = False):
    query_vector = embed_query(client, query_text)
    pool = min(max(top_k * 6, 24), len(metadata))
    scores, indices = index.search(query_vector, pool)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        if wiki_only and item.get("source_kind") != "wiki" and item.get("doc_type") != "wiki":
            continue
        item["score"] = float(score)
        hits.append(item)
        if len(hits) >= top_k:
            break
    return hits


def load_source_text(args) -> tuple[str, str]:
    if args.source_file:
        path = Path(args.source_file)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return path.name, text
    if args.text:
        return "inline_input", args.text
    raise ValueError("Provide --source-file or text")


def unique_wiki_pages(metadata):
    seen = {}
    for item in metadata:
        if item.get("source_kind") == "wiki" or item.get("doc_type") == "wiki":
            path = item["path"]
            if path not in seen:
                seen[path] = {
                    "path": path,
                    "title": item.get("title") or Path(path).stem,
                    "folder": item.get("folder", ""),
                }
    return list(seen.values())


def wiki_page_lookup(metadata):
    pages = {}
    for page in unique_wiki_pages(metadata):
        pages[page["path"]] = page
        pages[page["title"]] = page
        pages[Path(page["path"]).stem] = page
    return pages


def render_hits(hits):
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(
            f"[{i}] {hit.get('title', Path(hit['path']).stem)} | "
            f"path={hit['path']}#chunk-{hit['chunk_id']} | "
            f"score={hit['score']:.4f}\n{hit['text']}"
        )
    return "\n\n".join(lines)


def build_prompt(source_name: str, source_text: str, wiki_pages, evidence_hits, wiki_hits):
    wiki_catalog = "\n".join(f"- {page['title']} | {page['path']}" for page in wiki_pages[:40])
    return f"""
You are a PepCLIP knowledge-base maintenance assistant.
Your task is not to answer freely. Your task is to propose how new content should be merged into a canonical wiki page.

Read the input content, relevant evidence, and current wiki page candidates, then output JSON only.

Rules:
1. Prefer updating an existing canonical page instead of creating a duplicate page.
2. Distinguish fact / problem / hypothesis / plan / result / comparison.
3. Base the proposal only on the input and provided evidence.
4. Output valid JSON only. Do not use markdown code fences.

JSON schema:
{{
  "suggested_page_path": "existing canonical page path, or empty string",
  "suggested_page_title": "page title",
  "content_type": "fact|problem|hypothesis|plan|result|comparison",
  "reason": "why this page should be updated",
  "section_title": "section title to write under",
  "proposed_markdown": "markdown to append, concise but complete",
  "evidence_paths": ["path#chunk-x", "..."]
}}

Input source: {source_name}
Input content:
{source_text[:MAX_SOURCE_CHARS]}

Existing canonical wiki pages:
{wiki_catalog}

Most relevant current wiki hits:
{render_hits(wiki_hits)}

Most relevant evidence hits:
{render_hits(evidence_hits)}
""".strip()


def generate_proposal(client, prompt: str):
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


def normalize_page_reference(raw: str, pages_lookup: dict):
    candidate = (raw or "").strip()
    if not candidate:
        return None

    if "|" in candidate:
        parts = [part.strip() for part in candidate.split("|") if part.strip()]
        for part in reversed(parts):
            if part in pages_lookup:
                return pages_lookup[part]
            for key, value in pages_lookup.items():
                if isinstance(key, str) and part == key:
                    return value
                if isinstance(key, str) and part.endswith(key):
                    return value

    if candidate in pages_lookup:
        return pages_lookup[candidate]

    for key, value in pages_lookup.items():
        if isinstance(key, str) and candidate.endswith(key):
            return value

    return None


def apply_guardrails(proposal: dict, pages_lookup: dict):
    normalized = proposal.copy()
    matched_page = normalize_page_reference(normalized.get("suggested_page_path", ""), pages_lookup)

    if matched_page is None:
        matched_page = normalize_page_reference(normalized.get("suggested_page_title", ""), pages_lookup)

    if matched_page is not None:
        normalized["suggested_page_path"] = matched_page["path"]
        normalized["suggested_page_title"] = matched_page["title"]

        page_name = Path(matched_page["path"]).stem.lower()
        content_type = (normalized.get("content_type") or "").strip().lower()

        if "next_experiments" in page_name or "next experiments" in page_name:
            if content_type in {"result", "fact", ""}:
                normalized["content_type"] = "plan"
        elif "known_limitations" in page_name or "known limitations" in page_name:
            if content_type in {"result", "fact", "plan", ""}:
                normalized["content_type"] = "problem"
        elif "comparison" in page_name and content_type in {"result", "fact", ""}:
            normalized["content_type"] = "comparison"

    return normalized


def append_update(target_path: Path, proposal: dict):
    section_title = (proposal.get("section_title") or "").strip()
    proposed_markdown = proposal["proposed_markdown"].strip()
    if section_title:
        markdown_lines = proposed_markdown.splitlines()
        if markdown_lines:
            first_line = markdown_lines[0].strip()
            if first_line.startswith("#"):
                first_heading = first_line.lstrip("#").strip()
                if first_heading.casefold() == section_title.casefold():
                    proposed_markdown = "\n".join(markdown_lines[1:]).strip()
    content_block = "\n" + proposed_markdown.rstrip() + "\n"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fallback_block = (
        "\n\n## Proposed Updates\n"
        f"\n### {timestamp} | {section_title or 'Update'}\n"
        f"{proposed_markdown.rstrip()}\n"
    )

    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else f"# {proposal['suggested_page_title']}\n"

    if section_title:
        lines = existing.splitlines()
        insert_at = None
        normalized_section = normalize_heading(section_title)

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_text = stripped.lstrip("#").strip()
                normalized_heading = normalize_heading(heading_text)
                if normalized_heading == normalized_section:
                    insert_at = idx + 1
                    while insert_at < len(lines) and not lines[insert_at].strip():
                        insert_at += 1
                    break
                if normalized_section and (
                    normalized_section in normalized_heading
                    or normalized_heading in normalized_section
                ):
                    insert_at = idx + 1
                    while insert_at < len(lines) and not lines[insert_at].strip():
                        insert_at += 1
                    break

        if insert_at is not None:
            next_heading = len(lines)
            for idx in range(insert_at, len(lines)):
                if lines[idx].strip().startswith("#"):
                    next_heading = idx
                    break
            before = "\n".join(lines[:next_heading]).rstrip()
            after = "\n".join(lines[next_heading:]).lstrip()
            updated = before + content_block
            if after:
                updated += "\n\n" + after
            target_path.write_text(updated.rstrip() + "\n", encoding="utf-8")
            return

    if "## Proposed Updates" in existing:
        updated = existing.rstrip() + "\n" + fallback_block.replace("\n\n## Proposed Updates\n", "\n")
    else:
        updated = existing.rstrip() + fallback_block
    target_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def write_update_log(vault_root: Path, source_name: str, target_path: Path, proposal: dict):
    now = datetime.now()
    logs_dir = vault_root / "PepCLIP-Brain" / "09 Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"wiki_updates_{now.strftime('%Y-%m')}.md"

    rel_target = str(target_path.relative_to(vault_root)) if target_path.is_relative_to(vault_root) else str(target_path)
    evidence = proposal.get("evidence_paths", [])
    evidence_lines = "\n".join(f"- {item}" for item in evidence) if evidence else "- none"
    section_title = proposal.get("section_title") or "Update"
    title = proposal.get("suggested_page_title") or Path(target_path).stem

    entry = (
        f"\n## {now.strftime('%Y-%m-%d %H:%M')} | {title}\n"
        f"- source: `{source_name}`\n"
        f"- target: `{rel_target}`\n"
        f"- section: `{section_title}`\n"
        f"- content_type: `{proposal.get('content_type', '')}`\n"
        f"- reason: {proposal.get('reason', '')}\n"
        f"- evidence:\n{evidence_lines}\n\n"
        f"### Proposed Markdown\n{proposal.get('proposed_markdown', '').rstrip()}\n"
    )

    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
    else:
        existing = f"# Wiki Update Log - {now.strftime('%Y-%m')}\n"
    log_path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")


def resolve_target(vault_root: Path, proposal: dict, explicit_target: str | None):
    if explicit_target:
        return Path(explicit_target)
    page_path = proposal.get("suggested_page_path", "").strip()
    if page_path:
        return vault_root / page_path
    return None


def print_proposal(proposal: dict):
    print("\n=== Update Proposal ===")
    print(f"Target page: {proposal.get('suggested_page_title', '')}")
    print(f"Target path: {proposal.get('suggested_page_path', '')}")
    print(f"Content type: {proposal.get('content_type', '')}")
    print(f"Reason: {proposal.get('reason', '')}")
    print(f"Section title: {proposal.get('section_title', '')}")
    print("\n--- Proposed Markdown ---\n")
    print(proposal.get("proposed_markdown", ""))
    print("\n--- Evidence ---")
    for item in proposal.get("evidence_paths", []):
        print(f"- {item}")


def main():
    parser = argparse.ArgumentParser(description="Propose or write updates into canonical wiki pages")
    parser.add_argument("text", nargs="?", help="New content to compile into the wiki")
    parser.add_argument("--source-file", help="Read new content from a file")
    parser.add_argument("--target", help="Explicit canonical page path to write into")
    parser.add_argument("--write", action="store_true", help="Append the proposal to the target page")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of evidence hits to retrieve")
    args = parser.parse_args()

    source_name, source_text = load_source_text(args)
    source_text = source_text.strip()
    if not source_text:
        raise ValueError("Input content is empty")

    vault_root = Path(os.getenv("RAG_VAULT", r"E:\notebook\ltw"))
    index, metadata = load_index(INDEX_DIR)
    wiki_pages = unique_wiki_pages(metadata)
    pages_lookup = wiki_page_lookup(metadata)

    evidence_hits = retrieve_hits(index, metadata, source_text, args.top_k, wiki_only=False)
    wiki_hits = retrieve_hits(index, metadata, source_text, min(5, args.top_k), wiki_only=True)

    prompt = build_prompt(source_name, source_text, wiki_pages, evidence_hits, wiki_hits)
    proposal = generate_proposal(client, prompt)
    proposal = apply_guardrails(proposal, pages_lookup)
    print_proposal(proposal)

    if args.write:
        target_path = resolve_target(vault_root, proposal, args.target)
        if target_path is None:
            raise ValueError("No writable target page resolved. Use --target explicitly.")
        append_update(target_path, proposal)
        write_update_log(vault_root, source_name, target_path, proposal)
        print(f"\n[OK] Appended proposal to: {target_path}")


if __name__ == "__main__":
    client = build_client()
    main()
