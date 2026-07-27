SYSTEM_PROMPT = ""

USER_PROMPT_TEMPLATE = (
    "Answer concisely using only the evidence.\n\n"
    "Question:\n{question}\n\nEvidence:\n{context}"
)


def build_prompt(question: str, context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question, context=context)
