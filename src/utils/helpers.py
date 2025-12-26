from jinja2 import Environment, FileSystemLoader, Template
from datetime import datetime
import os


def get_prompt_template(template_path: str) -> Template:
    """
    Reads and returns a Jinja2 template file.

    Args:
        template_path (str): Path to the Jinja2 template file.

    Returns:
        Template: The specified jinja template.
    """

    # Get directory and filename
    template_dir, template_file = os.path.split(template_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    return template


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


def split_text_by_words(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """
    Split text into overlapping chunks based on number of words.

    Args:
        text (str): Input text to split.
        chunk_size (int): Number of words per chunk.
        overlap_size (int): Number of overlapping words between chunks.

    Returns:
        List[str]: List of chunked text strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_size < 0:
        raise ValueError("overlap_size must be >= 0")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size")

    words = text.split()
    chunks = []

    start = 0
    n_words = len(words)

    while start < n_words:
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Move start forward, keeping overlap
        start = end - overlap_size

    return chunks
