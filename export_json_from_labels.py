from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_CSV_PATH = Path("video_labels.csv")
DEFAULT_JSON_PATH = Path("exported_labels.json")


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _to_situation_lower(v: str) -> str:
    s = v.strip().lower()
    if s in {"affection", "intent", "attitude"}:
        return s
    return s


def _generic_label(rec: Dict[str, Any]) -> str:
    situation = _to_situation_lower(_safe_text(rec.get("situation", "")))
    if situation == "affection":
        return _safe_text(rec.get("label_Affection", ""))
    if situation == "intent":
        return _safe_text(rec.get("label_Intent", ""))
    if situation == "attitude":
        return _safe_text(rec.get("label_Attitude", ""))
    return ""


def _build_image_url(filename: str, base_image_url: str) -> str:
    if not filename:
        return ""
    base = base_image_url.strip()
    if not base:
        return filename
    return f"{base.rstrip('/')}/{filename}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export video_labels.csv back to input/output JSON format."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_PATH))
    parser.add_argument(
        "--base-image-url",
        default="",
        help="Optional prefix for image_url, e.g. https://xxx/path",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include rows with skipped=True.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    out_path = Path(args.output_json)
    base_image_url = args.base_image_url

    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)

    result: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        skipped = str(rec.get("skipped", "")).strip().lower() in {"true", "1", "yes"}
        if skipped and not args.include_skipped:
            continue

        filename = _safe_text(rec.get("filename", "")).strip()
        item = {
            "input": {
                "id": _safe_text(rec.get("id", "")).strip(),
                "text": _safe_text(rec.get("input_text", "")).strip(),
                "image_url": _build_image_url(filename, base_image_url),
            },
            "output": {
                "subject": _safe_text(rec.get("subject", "")).strip(),
                "target": _safe_text(rec.get("target", "")).strip(),
                "situation": _to_situation_lower(_safe_text(rec.get("situation", ""))),
                "mechanism": _safe_text(rec.get("mechanism", "")).strip(),
                "label": _generic_label(rec).strip(),
                "domain": _safe_text(rec.get("domain", "")).strip(),
                "culture": _safe_text(rec.get("culture", "")).strip(),
                "rationale": _safe_text(rec.get("rationale", "")).strip(),
            },
        }
        result.append(item)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported: {len(result)}")
    print(f"Output JSON: {out_path}")


if __name__ == "__main__":
    main()

