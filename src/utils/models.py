from langchain_ollama import ChatOllama
from typing import Literal

from src.config import MODEL_CONFIG


def get_llm(
    model_name: str,
    temperature: float = 0,
    reasoning: bool | Literal["low", "medium", "high"] | None = None,
    **kwargs,
) -> ChatOllama:
    """
    Returns a ChatOllama instance with the specified model name and parameters

    Parameters:
    - model_name (str): The name of the Ollama model to use.
    - temperature (float): The temperature setting for the model.
    - reasoning (bool|str): Whether to enable reasoning mode. Set it to a string for,
       'gpt-oss' models and valid values are 'low', 'medium', 'high'.
    - **kwargs: Additional keyword arguments to pass to the ChatOllama constructor.
    """
    if (
        model_name.startswith("gemma3")
        or model_name.startswith("llama3.1")
        or model_name.startswith("granite4")
        or model_name.startswith("mistral")
    ):
        reasoning = None  # Gemma3 models do not support reasoning mode

    return ChatOllama(
        model=model_name,
        validate_model_on_init=True,
        temperature=temperature,
        reasoning=reasoning,
        **kwargs,
    )


def get_model(
    temperature: float = MODEL_CONFIG["temperature"],
    reasoning: bool | Literal["low", "medium", "high"] | None = MODEL_CONFIG[
        "reasoning"
    ],
    **kwargs,
) -> ChatOllama:
    """
    Returns a default ChatOllama instance with predefined model configuration.
    Parameters:
    - temperature (float): The temperature setting for the model.
    - reasoning (bool|str): Whether to enable reasoning mode. Set it to a string for
       'gpt-oss' models and valid values are 'low', 'medium', 'high'.
    Returns:
    - ChatOllama: Configured ChatOllama instance.
    """
    model_reasoning = reasoning
    model_name = MODEL_CONFIG["model_name"]
    if not model_name.startswith("gpt"):
        if reasoning == "low":
            model_reasoning = None
        else:
            model_reasoning = True

    return get_llm(
        model_name=model_name,
        temperature=temperature,
        reasoning=model_reasoning,
        **kwargs,
    )
