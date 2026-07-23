SYSTEM_PROMPT = """You answer only from the supplied evidence.
Every major claim must cite an Evidence ID in square brackets. Do not turn correlation into causation.
State conflicting conditions separately. If evidence is insufficient, say so explicitly.
Answer in the same language as the question."""


def build_prompt(question: str, context: str) -> str:
    return f"Question:\n{question}\n\nEvidence:\n{context}\n\nWrite the evidence-grounded answer."

