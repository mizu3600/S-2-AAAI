from qmshe.generation.prompt_builder import SYSTEM_PROMPT, build_prompt
from qmshe.providers import DeepSeekClient, ProviderError


class EvidenceGenerator:
    def __init__(self):
        try:
            self.client = DeepSeekClient()
        except ProviderError:
            self.client = None

    def generate(self, question: str, context: str) -> str:
        if not context.strip():
            return "证据不足，无法确定。"
        if self.client is not None:
            try:
                return self.client.complete(SYSTEM_PROMPT, build_prompt(question, context))
            except ProviderError:
                pass
        evidence_ids = [
            line.removeprefix("[Evidence ").removesuffix("]")
            for line in context.splitlines()
            if line.startswith("[Evidence ")
        ]
        original = [line.removeprefix("Original text: ") for line in context.splitlines() if line.startswith("Original text: ")]
        citations = " ".join(f"[{item}]" for item in evidence_ids)
        return f"{original[0] if original else '证据不足，无法确定。'} {citations}".strip()
