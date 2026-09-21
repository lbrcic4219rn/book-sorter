import csv
from pathlib import Path

from book_organizer.metadata.models import BookMetadata

def write_author_mapping_report(mapping: dict[str, str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_variant", "canonical_name"])
        for original, canonical in sorted(mapping.items()):
            writer.writerow([original, canonical])
            
def write_report(
    rows: list[tuple[BookMetadata, BookMetadata | None, str, float | str]],
    report_path: Path,
) -> None:
    """rows: list of (old_meta, new_meta_or_None, action, confidence)."""
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "original_path",
                "old_title",
                "old_author",
                "new_title",
                "new_author",
                "new_genre",
                "action",
                "confidence",
            ]
        )
        for old, new, action, confidence in rows:
            writer.writerow(
                [
                    old.path,
                    old.title,
                    old.authors,
                    new.title if new else "",
                    new.authors if new else "",
                    new.genre if new else "",
                    action,
                    confidence,
                ]
            )
