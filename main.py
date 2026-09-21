import argparse
from pathlib import Path

from book_organizer.metadata.authors import build_author_mapping
from book_organizer.metadata.extractor import extract_metadata
from book_organizer.metadata.fetcher import fetch_online_metadata, find_isbn_in_filename
from book_organizer.metadata.genres import classify_genre
from book_organizer.metadata.llm import extract_text_snippet, guess_from_filename, guess_from_text
from book_organizer.metadata.mathcer import resolve_match
from book_organizer.metadata.models import BookMetadata
from book_organizer.metadata.writer import apply_metadata, rename_book
from book_organizer.report import write_report, write_author_mapping_report
from book_organizer.scanner import scan_books
from config import load_config


def gather_metadata(book_path: Path, config: dict) -> tuple[BookMetadata, BookMetadata | None]:
    """Phase 1: figure out what each book is. No writes/renames yet — author
    and genre standardization both need the whole library gathered first."""
    old_meta = extract_metadata(book_path)
    isbn = old_meta.isbn or find_isbn_in_filename(book_path)
    use_llm, llm_model = config["use_llm"], config["llm_model"]

    search_title, search_authors, llm_genre = old_meta.title, old_meta.authors, None
    if use_llm and not isbn and not search_title:
        guess = guess_from_filename(book_path, llm_model)
        search_title = search_title or guess.title
        search_authors = search_authors or guess.authors

    new_meta = fetch_online_metadata(book_path, isbn=isbn, title=search_title, authors=search_authors)

    if use_llm and (not new_meta or not new_meta.title or not new_meta.authors or not new_meta.genre):
        snippet = extract_text_snippet(book_path)
        if snippet:
            new_meta = guess_from_text(book_path, snippet, llm_model)

    return old_meta, new_meta


def main():
    parser = argparse.ArgumentParser(description="Clean up a messy ebook library.")
    parser.add_argument("--report", type=Path, default=Path("cleanup_report.csv"))
    parser.add_argument("--author-report", type=Path, default=Path("author_mapping.csv"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = load_config()
    library = Path(config["library_path"] + "/Kolekcija Knjiga")
    books = scan_books(library, config["supported_extensions"])
    print(f"Found {len(books)} books.")

    # Phase 1: gather raw metadata for every book
    records = []
    for i, book in enumerate(books, 1):
        if i == 1000: break
        print(f"[{i}/{len(books)}] gathering metadata: {book.name}")
        records.append(gather_metadata(book, config))

    # Phase 2: standardize author names across the WHOLE library
    all_authors = [new.authors for _, new in records if new and new.authors]
    llm_model = config.get("llm_model") if config.get("use_llm") else None
    author_mapping = build_author_mapping(all_authors, llm_model)
    write_author_mapping_report(author_mapping, args.author_report)
    print(f"Resolved {len(all_authors)} author strings into "
          f"{len(set(author_mapping.values()))} canonical names "
          f"(see {args.author_report} to review)")

    for _, new in records:
        if new and new.authors in author_mapping:
            new.authors = author_mapping[new.authors]

    # # Phase 3: standardize genres onto the fixed taxonomy
    # taxonomy, llm_model = config["genre_taxonomy"], config["llm_model"]
    # for _, new in records:
    #     if new and new.genre:
    #         new.genre = classify_genre(new.genre, taxonomy, llm_model)
    #
    # # Phase 4: confidence check, apply, report
    # threshold, use_llm = config["confidence_threshold"], config["use_llm"]
    # rows = []
    # for old_meta, new_meta in records:
    #     if not new_meta or not new_meta.title:
    #         rows.append((old_meta, None, "no_match", ""))
    #         continue
    #     score, accepted = resolve_match(old_meta, new_meta, threshold, use_llm, llm_model)
    #     if accepted:
    #         action = "would_rename_and_tag" if args.dry_run else "renamed_and_tagged"
    #         if not args.dry_run:
    #             apply_metadata(new_meta)
    #             rename_book(new_meta)
    #     else:
    #         action = "flagged_for_review"
    #     rows.append((old_meta, new_meta, action, round(score, 2)))
    #
    # write_report(rows, args.report)
    # print(f"\nDone. Report written to {args.report}")


if __name__ == "__main__":
    main()