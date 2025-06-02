def rule_to_asp(rule: str):
    try:
        _, condition, _, outcome, direction = rule.split(" ", 4)
        condition = condition.replace(">", "").strip()
        predicate = outcome.replace(" ", "_").replace("(", "").replace(")", "").lower()
        direction_sym = "increase" if "increase" in direction else "decrease"
        return f"{direction_sym}({predicate}) :- high({condition})."
    except Exception as e:
        return f"% Failed to convert rule: {rule} -- {str(e)}"
