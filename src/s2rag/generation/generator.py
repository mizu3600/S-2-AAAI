import hashlib

from s2rag.generation.prompt_builder import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_prompt,
)
from s2rag.providers import DeepSeekClient


SHARED_DEEPSEEK_GENERATION_PROTOCOL = "unified_concise_deepseek_v1"


class EvidenceGenerator:
    protocol = SHARED_DEEPSEEK_GENERATION_PROTOCOL
    citation_capability = "none"

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
        prompt_material = (
            f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE}"
            if SYSTEM_PROMPT
            else USER_PROMPT_TEMPLATE
        )
        return {
            "generation_protocol": self.protocol,
            "prompt_sha256": hashlib.sha256(prompt_material.encode()).hexdigest(),
            "citation_capability": self.citation_capability,
            **config,
        }
