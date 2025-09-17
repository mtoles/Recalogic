import json
import logging
from typing import List, Dict, Callable, Any
from joblib import Memory
import openai

logger = logging.getLogger(__name__)
memory = Memory(".cache", verbose=0)  # cache dir

def make_hashable(messages: List[Dict[str, str]]) -> str:
    # Ensure stable hashing by sorting keys
    return json.dumps(messages, sort_keys=True)

@memory.cache
def _cached_api_call(messages_json: str, model_id: str) -> str:
    """Raw OpenAI call, cached by joblib."""
    messages = json.loads(messages_json)
    response = openai.chat.completions.create(
        model=model_id,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

def retry_with_fallback(
    messages: List[Dict[str, str]],
    validation_func: Callable[[str], bool],
    max_retries: int = 5,
    fallback_value: Any = None,
    model_id: str = "gpt-5-mini",
) -> Any:
    messages_json = make_hashable(messages)

    for attempt in range(max_retries):
        try:
            content = _cached_api_call(messages_json, model_id)
        except Exception as e:
            logger.error(f"API call failed on attempt {attempt+1}: {e}")
            continue

        if validation_func(content):
            return content
        else:
            logger.warning(f"Invalid response on attempt {attempt+1}, retrying...")

    logger.error(f"Failed after {max_retries} attempts")
    return fallback_value
