import re

def rule_to_asp_program(rules):
    """
    ブーリアンルールをASP形式のルールに変換する。

    例:
        "IF A > 10.00 THEN B ↑"
        ↓
        "b :- a_high."

    - 変数名は小文字化（ASPは小文字が慣例）
    - 閾値超えは "var_high"
    - 正の因果はそのまま "head :- body."
    - 負の因果（↓）は "head :- not body."
    - ルールの末尾はピリオド必須

    Parameters:
        rules: list of str, ブーリアンルールの文字列リスト

    Returns:
        asp_rules: list of str, ASPルール文字列リスト
    """

    asp_rules = []

    for rule in rules:
        try:
            # 前処理: IF ... THEN ... の構文に分割
            if not rule.startswith("IF ") or " THEN " not in rule:
                raise ValueError("Invalid rule format")

            condition_part, consequence_part = rule[3:].split(" THEN ")

            # 条件部のパース: 「変数 演算子 閾値」
            cond_pattern = r"^(.*)\s*(>=|<=|>|<|==|!=)\s*([0-9eE\.\-+]+)$"
            cond_match = re.match(cond_pattern, condition_part.strip())
            if not cond_match:
                raise ValueError(f"Invalid condition format: '{condition_part.strip()}'")

            var_name_raw = cond_match.group(1).strip()
            comp_op = cond_match.group(2)
            threshold = cond_match.group(3)

            # 結果部のパース: 「変数名 ↑ or ↓」
            cons_pattern = r"^(.*)\s*(↑|↓)$"
            cons_match = re.match(cons_pattern, consequence_part.strip())
            if not cons_match:
                raise ValueError(f"Invalid consequence format: '{consequence_part.strip()}'")

            tgt_var_raw = cons_match.group(1).strip()
            direction = cons_match.group(2)

            # サニタイズ: 論理変数名に適した形に変換
            def sanitize(name):
                return (
                    name.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("%", "")
                    .replace("$", "")
                    .replace(",", "")
                    .replace("__", "_")
                )

            var_name = sanitize(var_name_raw)
            tgt_var = sanitize(tgt_var_raw)

            # 条件と帰結のASP表現
            if direction == "↑":
                asp_rule = f"{tgt_var}_high :- {var_name}_high.".replace("__", "_")
            else:  # "↓"
                asp_rule = f"{tgt_var}_high :- not {var_name}_high.".replace("__", "_")

            asp_rules.append(asp_rule)

        except Exception as e:
            print(f"Rule parse error: '{rule}' -> {e}")

    return asp_rules
