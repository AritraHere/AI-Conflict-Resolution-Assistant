import os
import pickle
from typing import List

from crewai.tools import tool

# Import the initialized models and predictors directly from our DRY-refactored ml_models.py
from ml_models import (
    predict_emotion,
    predict_conflict,
    _emotion_vectorizer,
    _emotion_clf,
    _conflict_vectorizer,
    _conflict_clf,
)

@tool("Analyze Emotional Timeline")
def analyze_emotional_timeline(conversation: str) -> str:
    """Take a raw multi-line conversation and classify emotion and conflict risk per line.

    The conversation should consist of lines like ``Speaker: utterance``. The
    tool splits on newlines, identifies the speaker, runs the lightweight
    predictors on the utterance, and returns a human-readable timeline.
    """
    timeline_lines: List[str] = []
    
    for raw in conversation.strip().splitlines():
        if ":" in raw:
            speaker, utterance = raw.split(":", 1)
            speaker = speaker.strip()
            text = utterance.strip()
        else:
            speaker = "Unknown"
            text = raw.strip()
            
        if text:
            # Run both the Emotion and Conflict ML pipelines
            emotion = predict_emotion(text, _emotion_vectorizer, _emotion_clf)
            conflict = predict_conflict(text, _conflict_vectorizer, _conflict_clf)
        else:
            emotion = "Neutral"
            conflict = "None"
            
        timeline_lines.append(
            f"[{speaker}] said '{text}' -> Detected Emotion: {emotion} | Conflict Risk Type: {conflict}"
        )
        
    return "\n".join(timeline_lines)