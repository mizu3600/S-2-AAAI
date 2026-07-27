SYSTEM_PROMPT = """You answer only from the supplied evidence.
Every major claim must cite an Evidence ID in square brackets. Do not turn correlation into causation.
State conflicting conditions separately. If evidence is insufficient, say so explicitly.
Answer in the same language as the question."""

USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\nEvidence:\n{context}\n\n"
    "Write the evidence-grounded answer."
)


def build_prompt(question: str, context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question, context=context)
