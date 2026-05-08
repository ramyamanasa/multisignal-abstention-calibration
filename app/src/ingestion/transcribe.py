"""Audio transcription using faster-whisper for lecture ingestion."""

import warnings
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "base") -> list[dict]:
    """Transcribe an audio file into timestamped text segments.

    Args:
        audio_path: Path to the audio file (e.g. .mp3, .wav, .m4a).
        model_size: Whisper model variant to load (e.g. "base", "small", "medium").

    Returns:
        A list of dicts with keys:
            start (float): Segment start time in seconds.
            end (float): Segment end time in seconds.
            text (str): Transcribed text for the segment.
    """
    if not Path(audio_path).exists():
        warnings.warn(f"Audio file not found: {audio_path}", UserWarning, stacklevel=2)
        return []

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)

    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]
