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
