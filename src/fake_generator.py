"""
FakeMessageGenerator: produces random Messages that match LenSpec intervals.
Used to feed the TaskBuilder with fake tasks for the mock server.
"""

import random
from typing import Generator

from defination import Message, ConfigRegistry

# Word pool for generating random prompts
_WORDS = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "with", "as", "was", "on", "are", "be", "this", "have", "from", "or",
    "an", "by", "not", "but", "what", "all", "were", "when", "we", "there",
    "can", "had", "has", "will", "each", "about", "how", "up", "out", "them",
    "then", "she", "many", "some", "so", "these", "would", "other", "into", "more",
    "model", "data", "system", "input", "output", "function", "process", "result",
    "value", "type", "class", "method", "object", "string", "number", "list",
    "array", "index", "element", "node", "tree", "graph", "path", "query",
    "server", "client", "request", "response", "token", "prompt", "generate",
]


def _random_prompt(target_len: int) -> str:
    """Generate a random prompt of approximately target_len characters."""
    words = []
    current_len = 0
    while current_len < target_len:
        word = random.choice(_WORDS)
        words.append(word)
        current_len += len(word) + 1
    return " ".join(words)[:target_len]


def generate_messages(
    registry: ConfigRegistry,
    msg_id_start: int = 0,
) -> Generator[Message, None, None]:
    """
    Infinite generator of fake Messages.

    Cycles through LenSpec intervals to produce a balanced stream
    of messages that the TaskBuilder can bucket.
    """
    lenspecs = list(registry._lenspec.items())
    msg_id = msg_id_start

    while True:
        # Pick a random LenSpec interval
        label, spec = random.choice(lenspecs)
        target_len = random.randint(spec.prompt_min, spec.prompt_max - 1)

        prompt = _random_prompt(target_len)

        msg = Message(
            ID=msg_id,
            prompt=prompt,
        )
        msg_id += 1
        yield msg
