"""Record → text for embedding (spec §5.4)."""

from typing import Any


def chunk_text(record: dict[str, Any]) -> str:
    """One `field: value` line per field, skipping null and empty strings."""
    return "\n".join(
        f"{field}: {value}"
        for field, value in record.items()
        if value is not None and value != ""
    )
