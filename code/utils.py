"""
utils.py - Utility functions.
"""

import hashlib
import re
from typing import List, Tuple
from pathlib import Path


def hash_text(text: str) -> str:
    """Hash text for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    # Remove special chars, lowercase, split
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def calculate_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard similarity between two texts."""
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(pattern, text)


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    pattern = r"https?://[^\s]+"
    return re.findall(pattern, text)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_confidence(score: float) -> str:
    """Format confidence score as percentage."""
    return f"{score * 100:.1f}%"


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def validate_csv_headers(headers: List[str], required: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate CSV headers.

    Returns:
        (is_valid, missing_headers)
    """
    header_set = set(headers)
    required_set = set(required)
    missing = list(required_set - header_set)

    return len(missing) == 0, missing
