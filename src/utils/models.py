from langchain_ollama import ChatOllama


def get_llm(
    model_name: str, temperature: float = 0, reasoning: bool | str = False, **kwargs
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
    if not reasoning and model_name.startswith("gpt-oss"):
        reasoning = "medium"
    if (
        model_name.startswith("gemma3")
        or model_name.startswith("llama3.1")
        or model_name.startswith("granite4")
        or model_name.startswith("mistral")
    ):
        reasoning = None  # Gemma3 models do not support reasoning mode
    return ChatOllama(
        model=model_name, temperature=temperature, reasoning=reasoning, **kwargs
    )


MODEL = get_llm("gpt-oss", temperature=0)
