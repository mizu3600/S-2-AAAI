def validate_answer_citations(answer: str, allowed_ids: list[str]) -> tuple[bool, list[str]]:
    missing = [evidence_id for evidence_id in allowed_ids if f"[{evidence_id}]" not in answer]
    return not missing, missing

