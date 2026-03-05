# :   — Audio / Transcript Ingestion Pipeline
# What to change:
#   1. Set WHISPER_MODEL in .env (default: "small").
#   2. Provide audio files or use bundled mock transcript.
#
# Run-time steps:
#   1) pip install openai-whisper   (requires ffmpeg on PATH)
#   2) python -m backend.ingest_audio --input path/to/audio.mp3
#
# TODO[USER_ACTION]: PROVIDE_PATH_TO_LECTURE_AUDIO or upload via UI.
# TODO[USER_ACTION]: IF YOU WANT TO USE CLOUD_ASR_SET_PROVIDER_AND_KEY
#   Set ASR_PROVIDER and ASR_API_KEY in .env; see docs for OpenAI/AssemblyAI.

"""
ingest_audio.py — Transcribe lecture audio with Whisper and produce
time-coded segment documents for indexing.

Falls back to mock transcript JSON if Whisper is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, model_name: str = WHISPER_MODEL) -> Dict[str, Any]:
    """Transcribe an audio file using local Whisper model.

    Args:
        audio_path: Path to the audio file (.mp3, .wav, etc.).
        model_name: Whisper model size (tiny, base, small, medium, large).

    Returns:
        Dict with keys: lecture_id, segments [{start_time, end_time, text}].
    """
    try:
        import whisper  # type: ignore
    except ImportError:
        logger.warning(
            "openai-whisper not installed. Returning empty transcript. "
            "Install with: pip install openai-whisper"
        )
        return _mock_transcript_stub(audio_path)

    logger.info("Loading Whisper model '%s' …", model_name)
    model = whisper.load_model(model_name)

    logger.info("Transcribing %s …", audio_path)
    result = model.transcribe(audio_path)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start_time": round(seg["start"], 2),
            "end_time": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    lecture_id = Path(audio_path).stem
    transcript = {
        "lecture_id": lecture_id,
        "course": "",  # TODO[USER_ACTION]: attach course metadata
        "duration_seconds": segments[-1]["end_time"] if segments else 0,
        "segments": segments,
    }

    # Persist JSON alongside the audio
    out_path = Path(audio_path).with_suffix(".transcript.json")
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info("Saved transcript to %s (%d segments)", out_path, len(segments))

    return transcript


def _mock_transcript_stub(audio_path: str) -> Dict[str, Any]:
    """Return a minimal mock transcript when Whisper is not available."""
    return {
        "lecture_id": Path(audio_path).stem,
        "course": "UNKNOWN",
        "duration_seconds": 0,
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "[Mock transcript — Whisper not available. Install openai-whisper.]",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Load existing transcript JSON
# ---------------------------------------------------------------------------

def load_transcript_json(json_path: str) -> Dict[str, Any]:
    """Load a pre-existing transcript JSON file.

    Args:
        json_path: Path to the transcript JSON.

    Returns:
        Transcript dict with lecture_id, segments, etc.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        "Loaded transcript '%s' with %d segments",
        data.get("lecture_id", "?"),
        len(data.get("segments", [])),
    )
    return data


# ---------------------------------------------------------------------------
# Canonical document conversion
# ---------------------------------------------------------------------------

def transcript_to_documents(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a transcript dict into canonical document dicts for indexing.

    Each segment becomes its own document so provenance can point to a
    specific timestamp.

    Args:
        transcript: Transcript dict (lecture_id, segments, …).

    Returns:
        List of document dicts.
    """
    docs: List[Dict[str, Any]] = []
    lecture_id = transcript.get("lecture_id", "unknown")
    course = transcript.get("course", "")

    for i, seg in enumerate(transcript.get("segments", [])):
        doc_id = f"{lecture_id}_seg{i}"
        docs.append({
            "doc_id": doc_id,
            "source_type": "lecture",
            "text": seg["text"],
            "metadata": {
                "course": course,
                "lecture_id": lecture_id,
                "segment_index": i,
                "start_time": seg.get("start_time", 0.0),
                "end_time": seg.get("end_time", 0.0),
                "filename": f"{lecture_id}.audio",
            },
        })
    return docs


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Transcribe lecture audio")
    parser.add_argument("--input", required=True,
                        help="Path to audio file or transcript JSON")
    parser.add_argument("--output", default="data/processed/",
                        help="Output directory")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.suffix == ".json":
        transcript = load_transcript_json(str(inp))
    else:
        transcript = transcribe_audio(str(inp))

    docs = transcript_to_documents(transcript)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{transcript.get('lecture_id', 'transcript')}_docs.json"
    with open(str(out_file), "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(docs)} segment documents to {out_file}")
