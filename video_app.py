from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image


# =========================
# Configuration
# =========================
MEDIA_DIR = Path("images")  # Directory containing images and videos
LABELS_CSV = Path("video_labels.csv")

SITUATION_OPTIONS = ["Affection", "Intent", "Attitude"]
MECHANISM_OPTIONS = ["Mechanism A", "Socio-Cultural Context Dependency", "Mechanism C"]
DOMAIN_OPTIONS = ["NULL", "NULL", "NULL"]
CULTURE_OPTIONS = ["NULL", "NULL", "NULL"]
Affection_OPTIONS = ["Happy", "Sad", "Disgusted", "Angry", "Fearful", "Bad"]
ATTITUDE_OPTIONS = [
    "Supportive", "Appreciative", "Sympathetic", "Neutral", "Indifferent",
    "Disapproving", "Skeptical", "Concerned", "Dismissive", "Contemptuous", "Hostile"
]


# =========================
# CSV Column Definitions
# =========================
CSV_COLUMNS = [
    "filename",
    "id",
    "input_text",  # ✅ NEW
    "subject",
    "target",
    "situation",
    "mechanism",
    "domain",
    "culture",
    "label_Affection",
    "label_Intent",
    "label_Attitude",
    "rationale",
    "skipped",
]


# =========================
# Helpers
# =========================
def _safe_choice(value: Any, options: List[str], allow_empty: bool = False) -> Any:
    """
    Ensure default value for selectbox / radio / pills is valid:
    - If value is in options, return value
    - Otherwise:
        - If allow_empty is True, return "" (or None)
        - Otherwise return options[0] (or "" if options is empty)
    """
    if value in options:
        return value
    if allow_empty:
        return ""
    return options[0] if options else ""


def _normalize_choice_in_state(key: str, options: List[str], allow_empty: bool = False) -> None:
    """Normalize st.session_state[key] to a valid value in options."""
    current = st.session_state.get(key, "")
    st.session_state[key] = _safe_choice(current, options, allow_empty=allow_empty)


def _safe_text(v: Any) -> str:
    """Convert any value into a safe string for Streamlit text_input/text_area."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _ensure_text_state(keys: List[str]) -> None:
    """Force session_state keys to valid strings (prevents Streamlit widget crashes)."""
    for k in keys:
        st.session_state[k] = _safe_text(st.session_state.get(k, ""))


def _clear_bad_widget_state(keys: List[str]) -> None:
    """
    Streamlit can persist a corrupted widget state (non-str) across reruns.
    Remove those keys before widgets are created to avoid crashes.
    """
    for k in keys:
        if k in st.session_state and not isinstance(st.session_state.get(k), str):
            del st.session_state[k]


def _supported_media_files() -> List[Path]:
    """Load supported image and video files."""
    if not MEDIA_DIR.exists():
        return []
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    video_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    supported_ext = image_ext | video_ext
    files = [p for p in MEDIA_DIR.iterdir() if p.is_file() and p.suffix.lower() in supported_ext]
    return sorted(files, key=lambda p: p.name.lower())


def _is_image(file_path: Path) -> bool:
    """Check if file is an image format."""
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return file_path.suffix.lower() in image_ext


def _is_video(file_path: Path) -> bool:
    """Check if file is a video format."""
    video_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return file_path.suffix.lower() in video_ext


def _load_labels_df() -> pd.DataFrame:
    if LABELS_CSV.exists():
        try:
            # ✅ IMPORTANT: prevent empty cells from becoming NaN(float)
            df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig", keep_default_na=False)
            for col in CSV_COLUMNS:
                if col not in df.columns:
                    df[col] = "" if col != "skipped" else False
            return df[CSV_COLUMNS].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=CSV_COLUMNS)


def _labels_index(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """filename -> row dict (for easy loading)"""
    if df.empty:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        record: Dict[str, Any] = {}
        for k in CSV_COLUMNS:
            if k == "skipped":
                s = str(row.get(k, "")).strip().lower()
                record[k] = s in ("true", "1", "yes")
            else:
                record[k] = _safe_text(row.get(k, ""))
        out[str(row["filename"])] = record
    return out


def _upsert_label(df: pd.DataFrame, record: Dict[str, Any]) -> pd.DataFrame:
    filename = str(record["filename"])
    if df.empty or "filename" not in df.columns:
        return pd.DataFrame([record], columns=CSV_COLUMNS)

    mask = df["filename"].astype(str) == filename
    if mask.any():
        idx = df.index[mask][0]
        for k in CSV_COLUMNS:
            df.at[idx, k] = record.get(k, "" if k != "skipped" else False)
        return df

    return pd.concat([df, pd.DataFrame([record], columns=CSV_COLUMNS)], ignore_index=True)


def _save_labels_df(df: pd.DataFrame) -> None:
    df.to_csv(LABELS_CSV, index=False, encoding="utf-8-sig")


def _rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _init_session_state() -> None:
    defaults = {
        "current_index": 0,
        "id": "",
        "input_text": "",  # ✅ NEW
        "subject": "",
        "target": "",
        "situation": SITUATION_OPTIONS[0] if SITUATION_OPTIONS else "",
        "mechanism": MECHANISM_OPTIONS[0] if MECHANISM_OPTIONS else "",
        "domain": DOMAIN_OPTIONS[0] if DOMAIN_OPTIONS else "",
        "culture": CULTURE_OPTIONS[0] if CULTURE_OPTIONS else "",
        "label_Affection": "",
        "label_Intent": "",
        "label_Attitude": "",
        "rationale": "",
        "last_loaded_filename": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Force text keys to safe strings
    _ensure_text_state(["id", "input_text", "subject", "target", "label_Intent", "rationale"])

    # Normalize choice keys
    _normalize_choice_in_state("situation", SITUATION_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism", MECHANISM_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("domain", DOMAIN_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("culture", CULTURE_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Affection", Affection_OPTIONS, allow_empty=True)
    _normalize_choice_in_state("label_Attitude", ATTITUDE_OPTIONS, allow_empty=True)


def _load_record_into_inputs(record: Optional[Dict[str, Any]]) -> None:
    """Load saved record into input fields (or clear them)."""
    if not record:
        st.session_state.id = ""
        st.session_state.input_text = ""
        st.session_state.subject = ""
        st.session_state.target = ""
        st.session_state.situation = SITUATION_OPTIONS[0] if SITUATION_OPTIONS else ""
        st.session_state.mechanism = MECHANISM_OPTIONS[0] if MECHANISM_OPTIONS else ""
        st.session_state.domain = DOMAIN_OPTIONS[0] if DOMAIN_OPTIONS else ""
        st.session_state.culture = CULTURE_OPTIONS[0] if CULTURE_OPTIONS else ""
        st.session_state.label_Affection = ""
        st.session_state.label_Intent = ""
        st.session_state.label_Attitude = ""
        st.session_state.rationale = ""
        return

    st.session_state.id = _safe_text(record.get("id", ""))
    st.session_state.input_text = _safe_text(record.get("input_text", ""))
    st.session_state.subject = _safe_text(record.get("subject", ""))
    st.session_state.target = _safe_text(record.get("target", ""))
    st.session_state.situation = _safe_text(record.get("situation", ""))
    st.session_state.mechanism = _safe_text(record.get("mechanism", ""))
    st.session_state.domain = _safe_text(record.get("domain", ""))
    st.session_state.culture = _safe_text(record.get("culture", ""))
    st.session_state.label_Affection = _safe_text(record.get("label_Affection", ""))
    st.session_state.label_Intent = _safe_text(record.get("label_Intent", ""))
    st.session_state.label_Attitude = _safe_text(record.get("label_Attitude", ""))
    st.session_state.rationale = _safe_text(record.get("rationale", ""))

    _normalize_choice_in_state("situation", SITUATION_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("mechanism", MECHANISM_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("domain", DOMAIN_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("culture", CULTURE_OPTIONS, allow_empty=False)
    _normalize_choice_in_state("label_Affection", Affection_OPTIONS, allow_empty=True)
    _normalize_choice_in_state("label_Attitude", ATTITUDE_OPTIONS, allow_empty=True)


def _get_image_meta(image_path: Path) -> Tuple[int, int]:
    """Get image width and height (only valid for image files)."""
    try:
        with Image.open(image_path) as im:
            w, h = im.size
        return w, h
    except Exception:
        return 0, 0


def main() -> None:
    st.set_page_config(page_title="Media Annotation Tool", layout="wide")

    st.markdown(
        """
<style>
/* Adjust top padding to prevent header overlap */
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
div[data-testid="stVerticalBlock"] { gap: 0.2rem; }
/* Make primary button more prominent */
div[data-testid="stButton"] button[kind="primary"] { font-weight: 700; }
/* Compact metadata styling */
.meta { color: rgba(49, 51, 63, 0.7); font-size: 0.85rem; margin-top: 0.2rem; }
/* Reduce spacing in form elements */
.stTextInput, .stSelectbox, .stTextArea { margin-bottom: 0.15rem; }
/* Compact subheader */
h3 { margin-top: 0.3rem; margin-bottom: 0.3rem; }
/* Limit media size */
.stImage img { max-width: 100%; max-height: 420px; object-fit: contain; }
.stVideo video { max-width: 100%; max-height: 420px; height: auto; }
/* Compact right panel elements - maximize space efficiency */
div[data-testid="column"]:last-child .stSubheader { font-size: 1rem; margin-bottom: 0.2rem; margin-top: 0.2rem; }
div[data-testid="column"]:last-child .stButton button {
    padding: 0.05rem 0.2rem !important;
    font-size: 0.52rem !important;
    min-height: 20px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    width: 100%;
}
div[data-testid="column"]:last-child .stButton button * {
    font-size: 0.52rem !important;
    line-height: 1 !important;
    white-space: nowrap !important;
}
div[data-testid="column"]:last-child .stTextInput,
div[data-testid="column"]:last-child .stSelectbox,
div[data-testid="column"]:last-child .stTextArea { margin-bottom: 0.15rem; }
div[data-testid="column"]:last-child .stTextInput input,
div[data-testid="column"]:last-child .stSelectbox select,
div[data-testid="column"]:last-child .stTextArea textarea { font-size: 0.85rem; padding: 0.3rem 0.4rem; min-height: 32px; }
div[data-testid="column"]:last-child label { font-size: 0.8rem; margin-bottom: 0.1rem; }
div[data-testid="column"]:last-child [data-testid="stProgress"] { margin-bottom: 0.15rem; height: 0.5rem; display: block !important; visibility: visible !important; }
div[data-testid="column"]:last-child [data-testid="stProgress"] > div { height: 0.5rem !important; }
div[data-testid="column"]:last-child .stCaption { font-size: 0.75rem; margin-top: 0.1rem; }
div[data-testid="column"]:last-child .stMarkdown { margin-bottom: 0.2rem; }
div[data-testid="column"]:last-child .stDivider { margin: 0.3rem 0; }
div[data-testid="column"]:last-child [data-testid="stVerticalBlock"] { gap: 0.2rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    # ✅ AUTO-CLEAR bad persisted widget state (must be before widgets are created)
    _clear_bad_widget_state(["input_text", "subject", "target", "label_Intent", "rationale"])

    _init_session_state()

    media_files = _supported_media_files()
    total = len(media_files)
    if total == 0:
        st.warning(
            "No media files found. Please create an `images/` directory and add images (jpg/png/webp) "
            "or videos (mp4/mov/avi/mkv/webm)."
        )
        st.stop()

    labels_df = _load_labels_df()
    by_name = _labels_index(labels_df)

    # Calculate progress: considered done if not skipped and at least one field is filled
    def _is_done(rec: Dict[str, Any]) -> bool:
        if bool(rec.get("skipped", False)):
            return True
        return any(
            str(rec.get(k, "")).strip()
            for k in [
                "input_text",
                "subject",
                "target",
                "situation",
                "mechanism",
                "domain",
                "culture",
                "label_Affection",
                "label_Intent",
                "label_Attitude",
                "rationale",
            ]
        )

    done_count = sum(1 for p in media_files if _is_done(by_name.get(p.name, {})))

    # ====== Two-column layout: Left 70% / Right 30% ======
    left, right = st.columns([0.6, 0.4], gap="small")

    current_index = int(st.session_state.current_index)
    current_index = max(0, min(current_index, total - 1))
    st.session_state.current_index = current_index
    current_path = media_files[current_index]

    # Auto-load previous annotations when switching files
    if st.session_state.last_loaded_filename != current_path.name:
        _load_record_into_inputs(by_name.get(current_path.name))
        st.session_state.last_loaded_filename = current_path.name

    # ✅ Final safety: make sure text keys are strings right before rendering widgets
    _ensure_text_state(["id", "input_text", "subject", "target", "label_Intent", "rationale"])

    # =========================
    # Left Column: Media Display + (RED BOX AREA) Input + Mechanism/Domain/Culture/Rationale
    # =========================
    with left:
        # This creates the red-box area at the right of the media
        media_col, input_col = st.columns([0.48, 0.52], gap="medium")

        with media_col:
            # Display media based on file type
            if _is_image(current_path):
                st.image(str(current_path), use_container_width=True)
                w, h = _get_image_meta(current_path)
                if w > 0 and h > 0:
                    st.markdown(
                        f'<div class="meta">File: <b>{current_path.name}</b> | Size: <b>{w}×{h}</b> | '
                        f'Index: <b>{current_index + 1}/{total}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="meta">File: <b>{current_path.name}</b> | Index: <b>{current_index + 1}/{total}</b></div>',
                        unsafe_allow_html=True,
                    )
            elif _is_video(current_path):
                st.video(str(current_path))
                st.markdown(
                    f'<div class="meta">File: <b>{current_path.name}</b> | Type: <b>Video</b> | '
                    f'Index: <b>{current_index + 1}/{total}</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="meta">File: <b>{current_path.name}</b> | Index: <b>{current_index + 1}/{total}</b></div>',
                    unsafe_allow_html=True,
                )

        with input_col:
            # ✅ Requirement: place Input in the red-box area (title + box like Subject)
            st.text_input("ID", key="id")
            st.text_input("Input", key="input_text")

        st.divider()
        st.selectbox("Mechanism", MECHANISM_OPTIONS, key="mechanism")
        st.selectbox("Domain", DOMAIN_OPTIONS, key="domain")
        st.selectbox("Culture", CULTURE_OPTIONS, key="culture")
        st.text_area("Rationale", key="rationale", height=50)

    # =========================
    # Right Column: Progress + Navigation + Form
    # =========================
    with right:
        with st.container():
            prog = done_count / total if total else 0.0
            st.progress(prog)
            st.caption(f"Progress: {done_count}/{total}")

            nav_cols = st.columns([1, 1, 1, 1], gap="small")
            with nav_cols[0]:
                prev_clicked = st.button("Previous", use_container_width=True)
            with nav_cols[1]:
                accept_clicked = st.button("Accept", use_container_width=True, type="primary")
            with nav_cols[2]:
                pending_clicked = st.button("Pending", use_container_width=True)
            with nav_cols[3]:
                abandon_clicked = st.button("Abandon", use_container_width=True)

        st.divider()

        with st.container():
            st.markdown("**Annotation Form**")

            st.text_input("Subject", key="subject")
            st.text_input("Target", key="target")
            st.selectbox("Situation", SITUATION_OPTIONS, key="situation")

            st.selectbox("Label: Affection", Affection_OPTIONS, key="label_Affection")
            st.text_input("Label: Intent", key="label_Intent")
            st.selectbox("Label: Attitude", ATTITUDE_OPTIONS, key="label_Attitude")

        st.caption(f"Current: `{current_path.name}`")

    # =========================
    # Event handling: Previous / Save & Next / Skip
    # =========================
    def _go(index: int) -> None:
        st.session_state.current_index = max(0, min(index, total - 1))
        _rerun()

    def _next_index(from_idx: int) -> int:
        return min(from_idx + 1, total - 1)

    if prev_clicked:
        _go(current_index - 1)

    if abandon_clicked:
        record = {
            "filename": current_path.name,
            "id": "",
            "input_text": "",
            "subject": "",
            "target": "",
            "situation": "",
            "mechanism": "",
            "domain": "",
            "culture": "",
            "label_Affection": "",
            "label_Intent": "",
            "label_Attitude": "",
            "rationale": "",
            "skipped": True,
        }
        labels_df = _upsert_label(labels_df, record)
        _save_labels_df(labels_df)
        st.session_state.last_loaded_filename = ""  # Force reload on next file
        _go(_next_index(current_index))

    if pending_clicked:
        _go(_next_index(current_index))

    if accept_clicked:
        record = {
            "filename": current_path.name,
            "id": st.session_state.id,
            "input_text": st.session_state.input_text,
            "subject": st.session_state.subject,
            "target": st.session_state.target,
            "situation": st.session_state.situation,
            "mechanism": st.session_state.mechanism,
            "domain": st.session_state.domain,
            "culture": st.session_state.culture,
            "label_Affection": st.session_state.label_Affection,
            "label_Intent": st.session_state.label_Intent,
            "label_Attitude": st.session_state.label_Attitude,
            "rationale": st.session_state.rationale,
            "skipped": False,
        }
        labels_df = _upsert_label(labels_df, record)
        _save_labels_df(labels_df)
        st.session_state.last_loaded_filename = ""  # Force reload on next file
        _go(_next_index(current_index))


if __name__ == "__main__":
    main()
