"""Load JSON / JSONL / CSV files into flat records (spec §5.1)."""

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

Record = dict[str, Any]

CSV_ENCODINGS = ("utf-8-sig", "cp949")

# Plain decimal literals only: int() / float() alone would also accept
# "1_000", "nan" and "inf", which are not numbers in a CSV cell.
_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?")
_LEADING_ZERO_RE = re.compile(r"[+-]?0\d")


class LoadError(Exception):
    """The file cannot produce any records; the build must fail."""


def load_file(path: str | Path) -> list[Record]:
    """Load one data file into flat records. Raises LoadError."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _read_json(path)
    elif suffix == ".jsonl":
        raw = _read_jsonl(path)
    elif suffix == ".csv":
        return _finish(path, _read_csv(path))
    else:
        raise LoadError(f"{path}: unsupported file type '{suffix}'")

    records = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            records.append(flatten(item))
        else:
            logger.warning("%s: record %d is not an object, skipped", path, i)
    return _finish(path, records)


def flatten(obj: dict[str, Any], prefix: str = "") -> Record:
    """Nested objects → dot keys; lists → JSON strings."""
    out: Record = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        elif isinstance(value, list):
            out[name] = json.dumps(value, ensure_ascii=False)
        else:
            out[name] = value
    return out


def _finish(path: Path, records: list[Record]) -> list[Record]:
    if not records:
        raise LoadError(f"{path}: no valid records")
    return records


def _read_text(path: Path, encoding: str) -> str:
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError as e:
        raise LoadError(f"{path}: not valid {encoding}: {e}") from e


def _read_json(path: Path) -> list[Any]:
    try:
        data = json.loads(_read_text(path, "utf-8-sig"))
    except json.JSONDecodeError as e:
        raise LoadError(f"{path}: invalid JSON: {e}") from e

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        arrays = [v for v in data.values() if isinstance(v, list)]
        if len(arrays) == 1:
            return arrays[0]
        raise LoadError(
            f"{path}: top-level object must have exactly one array field "
            f"(found {len(arrays)})"
        )
    raise LoadError(f"{path}: top level must be an array or an object")


def _read_jsonl(path: Path) -> list[Any]:
    items = []
    for lineno, line in enumerate(_read_text(path, "utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("%s:%d: invalid JSON, skipped: %s", path, lineno, e)
    return items


def _decode_csv(path: Path) -> str:
    data = path.read_bytes()
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise LoadError(f"{path}: cannot decode as {' or '.join(CSV_ENCODINGS)}")


def _read_csv(path: Path) -> list[Record]:
    reader = csv.DictReader(io.StringIO(_decode_csv(path), newline=""))
    header = reader.fieldnames
    if not header or not any(h.strip() for h in header):
        raise LoadError(f"{path}: CSV header is required")

    rows = []
    for row in reader:
        if None in row:  # more cells than header columns
            logger.warning("%s:%d: too many columns, skipped", path, reader.line_num)
            continue
        rows.append(row)

    numeric = {name for name in header if _is_numeric_column([r[name] for r in rows])}
    return [{k: _convert_cell(v, k in numeric) for k, v in row.items()} for row in rows]


def _is_numeric_column(values: list[str | None]) -> bool:
    """Every non-empty cell is a plain number and none has a leading zero.

    Deciding per column keeps one type per column and keeps codes such as
    "007" or zip code "01234" as strings.
    """
    cells = [v.strip() for v in values if v]
    return bool(cells) and all(
        (_INT_RE.fullmatch(c) or _FLOAT_RE.fullmatch(c))
        and not _LEADING_ZERO_RE.match(c)
        for c in cells
    )


def _convert_cell(value: str | None, numeric: bool) -> Any:
    """'' / missing → None; numeric columns → int, else float."""
    if value is None or value == "":
        return None
    if not numeric:
        return value
    text = value.strip()
    return int(text) if _INT_RE.fullmatch(text) else float(text)
