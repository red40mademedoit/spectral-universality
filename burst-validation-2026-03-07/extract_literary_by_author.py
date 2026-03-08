#!/usr/bin/env python3
"""
Extract Greek text from canonical-greekLit TEI XML, organized by author.

Output: ~/data/greek/literary/{author_slug}/*.txt
One .txt file per work, containing only Greek text content.

Dialect classification included for quarantine purposes.
"""

import os
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# ── Configuration ────────────────────────────────────────────

CANONICAL_GREEKLIT = "/mnt/storage/knowledge-rag/scraped/github_greek/canonical-greekLit/data"
FIRST1K_GREEK = "/mnt/storage/knowledge-rag/scraped/github_greek/First1KGreek/data"
OUTPUT_DIR = os.path.expanduser("~/data-greek-literary")

# TEI namespace
TEI = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI}

# Greek Unicode detection
GREEK_RE = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')

# Author → dialect classification
# This is the quarantine map. Authors are grouped by primary dialect.
DIALECT_MAP = {
    # Koine / Imperial (1st c BCE – 4th c CE) — THE PRIMARY TEST GROUP
    "plutarch": "koine",
    "lucian-of-samosata": "koine",
    "aristides-aelius": "koine",
    "galen": "koine",
    "epictetus": "koine",
    "marcus-aurelius": "koine",
    "appianus-of-alexandria": "koine",
    "arrian": "koine",
    "cassius-dio-cocceianus": "koine",
    "dio-chrysostom": "koine",
    "diogenes-laertius": "koine",
    "flavius-josephus": "koine",
    "julian-emperor-of-rome": "koine",
    "philostratus-the-athenian": "koine",
    "pausanias": "koine",
    "strabo": "koine",
    "achilles-tatius": "koine",
    "chariton-of-aphrodisias": "koine",
    "longus": "koine",
    "xenophon-of-ephesus": "koine",
    "aelian": "koine",
    "athenaeus-of-naucratis": "koine",
    "diodorus-siculus": "koine",
    "polybius": "koine",
    "onasander": "koine",
    "aretaeus-of-cappadocia": "koine",

    # Koine / Christian-Patristic
    "new-testament": "koine-christian",
    "clement-of-alexandria": "koine-christian",
    "eusebius-of-caesarea": "koine-christian",
    "basil-saint-bishop-of-caesarea": "koine-christian",
    "john-of-damascus-saint": "koine-christian",

    # Attic (5th–4th c BCE)
    "demosthenes": "attic",
    "lysias": "attic",
    "isocrates": "attic",
    "isaeus": "attic",
    "plato": "attic",
    "aristotle": "attic",
    "xenophon": "attic",
    "aeschines": "attic",
    "andocides": "attic",
    "antiphon": "attic",
    "dinarchus": "attic",
    "hyperides": "attic",
    "lycurgus": "attic",
    "demades": "attic",
    "theophrastus": "attic",
    "aeneas-tacticus": "attic",

    # Attic / Drama
    "euripides": "attic-drama",
    "sophocles": "attic-drama",
    "aeschylus": "attic-drama",
    "aristophanes": "attic-drama",

    # Ionic
    "hippocrates": "ionic",
    "herodotus": "ionic",

    # Epic / Archaic
    "homer": "epic",
    "homeric-hymns": "epic",
    "hesiod": "epic",

    # Doric / Lyric
    "pindar": "doric",
    "bacchylides": "doric",

    # Hellenistic
    "callimachus": "hellenistic",
    "apollonius-rhodius": "hellenistic",
    "theocritus": "hellenistic",

    # Historiography (Attic-adjacent)
    "thucydides": "attic",

    # Late Antique / Byzantine
    "nonnus-of-panopolis": "late-antique",
    "quintus-smyrnaeus": "late-antique",
    "colluthus-of-lycopolis": "late-antique",
    "procopius": "late-antique",
    "proclus": "late-antique",
    "zonaras": "late-antique",
}

# Minimum Greek characters for a text to be worth keeping
MIN_GREEK_CHARS = 100


def slugify(name):
    """Convert author name to filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[,.]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


def extract_text_from_tei(xml_path):
    """Extract all text content from TEI XML body."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None, None, None

    # Get title
    title_el = root.find(f".//{{{TEI}}}titleStmt/{{{TEI}}}title")
    title = title_el.text if title_el is not None and title_el.text else ""

    # Get author
    author_el = root.find(f".//{{{TEI}}}titleStmt/{{{TEI}}}author")
    author = author_el.text if author_el is not None and author_el.text else ""

    # Extract text from body
    body = root.find(f".//{{{TEI}}}body")
    if body is None:
        body = root.find(f".//{{{TEI}}}text")
    if body is None:
        return author, title, ""

    # Get all text content, preserving paragraph breaks
    parts = []
    for elem in body.iter():
        if elem.text:
            parts.append(elem.text.strip())
        if elem.tail:
            parts.append(elem.tail.strip())

    text = "\n".join(p for p in parts if p)

    # Count Greek chars
    greek_chars = len(GREEK_RE.findall(text))

    return author, title, text if greek_chars >= MIN_GREEK_CHARS else ""


def process_corpus(corpus_dir, output_dir, corpus_name):
    """Process all TEI XML files in a corpus directory."""
    stats = {
        "total_xml": 0,
        "extracted": 0,
        "skipped_no_greek": 0,
        "skipped_parse_error": 0,
        "authors": {},
    }

    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"  {corpus_name}: directory not found at {corpus_dir}")
        return stats

    # Find all Greek XML files (not __cts__.xml)
    xml_files = sorted(corpus_path.rglob("*grc*.xml"))
    xml_files = [f for f in xml_files if "__cts__" not in f.name]
    stats["total_xml"] = len(xml_files)
    print(f"  {corpus_name}: found {len(xml_files)} Greek XML files")

    for xml_path in xml_files:
        author, title, text = extract_text_from_tei(xml_path)

        if author is None:
            stats["skipped_parse_error"] += 1
            continue

        if not text:
            stats["skipped_no_greek"] += 1
            continue

        # Determine author slug
        if author:
            author_slug = slugify(author)
        else:
            # Fallback: use CTS __cts__.xml from parent dirs
            for parent in xml_path.parents:
                cts = parent / "__cts__.xml"
                if cts.exists():
                    try:
                        cts_tree = ET.parse(cts)
                        gn = cts_tree.getroot().find(f".//{{{TEI}}}groupname")
                        if gn is None:
                            # Try without namespace
                            gn = cts_tree.getroot().find(".//groupname")
                        if gn is not None and gn.text:
                            author_slug = slugify(gn.text)
                            author = gn.text
                            break
                    except:
                        pass
            else:
                author_slug = "unknown"
                author = "Unknown"

        # Get dialect
        dialect = DIALECT_MAP.get(author_slug, "unknown")

        # Create output directory: {output_dir}/{dialect}/{author_slug}/
        author_dir = Path(output_dir) / dialect / author_slug
        author_dir.mkdir(parents=True, exist_ok=True)

        # Write text file
        work_name = xml_path.stem.replace(".", "_")
        out_path = author_dir / f"{work_name}.txt"
        out_path.write_text(text, encoding="utf-8")

        stats["extracted"] += 1
        if author_slug not in stats["authors"]:
            stats["authors"][author_slug] = {
                "name": author,
                "dialect": dialect,
                "works": 0,
                "total_chars": 0,
            }
        stats["authors"][author_slug]["works"] += 1
        stats["authors"][author_slug]["total_chars"] += len(text)

    return stats


def main():
    print("=" * 70)
    print("Greek Literary Text Extraction — By Author & Dialect")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process canonical-greekLit
    print("\n--- canonical-greekLit (Perseus) ---")
    stats1 = process_corpus(CANONICAL_GREEKLIT, OUTPUT_DIR, "canonical-greekLit")

    # Process First1KGreek
    print("\n--- First1KGreek ---")
    stats2 = process_corpus(FIRST1K_GREEK, OUTPUT_DIR, "First1KGreek")

    # Combined stats
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    all_authors = {}
    for s in [stats1, stats2]:
        for slug, info in s.get("authors", {}).items():
            if slug in all_authors:
                all_authors[slug]["works"] += info["works"]
                all_authors[slug]["total_chars"] += info["total_chars"]
            else:
                all_authors[slug] = dict(info)

    print(f"\nTotal XML files: {stats1['total_xml'] + stats2['total_xml']}")
    print(f"Extracted: {stats1['extracted'] + stats2['extracted']}")
    print(f"Skipped (no Greek): {stats1['skipped_no_greek'] + stats2['skipped_no_greek']}")
    print(f"Skipped (parse error): {stats1['skipped_parse_error'] + stats2['skipped_parse_error']}")
    print(f"Unique authors: {len(all_authors)}")

    # Print by dialect
    by_dialect = {}
    for slug, info in sorted(all_authors.items(), key=lambda x: -x[1]["works"]):
        d = info["dialect"]
        if d not in by_dialect:
            by_dialect[d] = []
        by_dialect[d].append((slug, info))

    print(f"\n{'Dialect':<20} {'Authors':>8} {'Works':>8} {'Chars':>12}")
    print("-" * 52)
    for dialect in sorted(by_dialect.keys()):
        authors = by_dialect[dialect]
        total_works = sum(a[1]["works"] for a in authors)
        total_chars = sum(a[1]["total_chars"] for a in authors)
        print(f"{dialect:<20} {len(authors):>8} {total_works:>8} {total_chars:>12,}")

    print(f"\n{'Author':<35} {'Dialect':<18} {'Works':>6} {'Chars':>10}")
    print("-" * 75)
    for slug, info in sorted(all_authors.items(), key=lambda x: -x[1]["works"]):
        if info["works"] >= 3:  # only show authors with 3+ works
            print(f"{info['name'][:34]:<35} {info['dialect']:<18} {info['works']:>6} {info['total_chars']:>10,}")

    # Save stats
    import json
    stats_path = os.path.join(OUTPUT_DIR, "_extraction_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "output_dir": OUTPUT_DIR,
            "sources": ["canonical-greekLit", "First1KGreek"],
            "authors": all_authors,
            "by_dialect": {d: [a[0] for a in authors] for d, authors in by_dialect.items()},
        }, f, indent=2, default=str)

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Stats: {stats_path}")

    # Show directory structure
    print(f"\nDirectory structure:")
    for dialect_dir in sorted(Path(OUTPUT_DIR).iterdir()):
        if dialect_dir.is_dir() and not dialect_dir.name.startswith("_"):
            n_authors = len(list(dialect_dir.iterdir()))
            print(f"  {dialect_dir.name}/  ({n_authors} authors)")


if __name__ == "__main__":
    main()
