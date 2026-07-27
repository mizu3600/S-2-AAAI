import hashlib

from s2rag.generation.prompt_builder import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_prompt,
)
from s2rag.providers import DeepSeekClient


SHARED_DEEPSEEK_GENERATION_PROTOCOL = "shared_deepseek_v1"


class EvidenceGenerator:
    protocol = SHARED_DEEPSEEK_GENERATION_PROTOCOL

    def __init__(self, client=None):
        self.client = client or DeepSeekClient()

    def generate(self, question: str, context: str) -> str:
        return self.client.complete(SYSTEM_PROMPT, build_prompt(question, context))

    def manifest(self) -> dict:
        config = (
            self.client.generation_config()
            if hasattr(self.client, "generation_config")
            else {}
        )
        return {
            "generation_protocol": self.protocol,
            "prompt_sha256": hashlib.sha256(
                f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE}".encode()
            ).hexdigest(),
            **config,
        }
