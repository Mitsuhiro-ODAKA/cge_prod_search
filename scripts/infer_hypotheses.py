import clingo
import re

def run_clingo(*files):
    ctl = clingo.Control()
    for f in files:
        ctl.load(f)
    ctl.ground([("base", [])])
    
    results = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            results.append([str(atom) for atom in model.symbols(shown=True)])
    return results

def check_satisfiability(data_rules, background_rules, verbose=False):
    program = "\n".join(background_rules + data_rules)

    if verbose:
        print("=== ASP プログラム ===")
        print(program)
        print("======================")

    ctl = clingo.Control()
    ctl.add("base", [], program)
    ctl.ground([("base", [])])

    answer_sets = []

    with ctl.solve(yield_ = True) as handle:
        for model in handle:
            answer_sets.append(model.symbols(shown=True))

    if answer_sets:
        print("✅ 充足可能：少なくとも1つのAnswer Setが存在します。")
        if verbose:
            for i, ans in enumerate(answer_sets):
                print(f"Answer Set {i+1}: {[str(a) for a in ans]}")
        return True, answer_sets
    else:
        print("❌ 非充足：Answer Setが存在しません（ルールに矛盾あり）。")
        return False, []
        

def parse_rules_from_list(rule_list):
    """ASPルールのリストを (head, body) タプルのリストに変換"""
    rules = []
    for rule in rule_list:
        if not isinstance(rule, str):
            continue  # 念のためリスト要素が文字列であることを確認
        rule = rule.strip().rstrip('.')
        if not rule or rule.startswith('%'):
            continue
        if ":-" in rule:
            head, body = map(str.strip, rule.split(":-"))
            body_literals = [b.strip() for b in body.split(",")]
        else:
            head = rule
            body_literals = []
        rules.append((head, body_literals))
    return rules

def detect_conflicts(data_rules, background_rules):
    """観測ルールと背景知識ルールの間の矛盾を検出"""
    data_parsed = parse_rules_from_list(data_rules)
    bg_parsed = parse_rules_from_list(background_rules)

    conflicts = []
    for d_head, d_body in data_parsed:
        for b_head, b_body in bg_parsed:
            if d_head != b_head:
                continue

            # 否定付きリテラルの衝突を検出
            for d in d_body:
                for b in b_body:
                    if d.startswith("not ") and d[4:] == b:
                        conflicts.append((("data", d_head, d_body), ("background", b_head, b_body)))
                    elif b.startswith("not ") and b[4:] == d:
                        conflicts.append((("data", d_head, d_body), ("background", b_head, b_body)))
    return conflicts

def remove_conflicting_data_rules(data_rules, background_rules):
    """矛盾する観測ルールを検出して削除"""
    data_parsed = parse_rules_from_list(data_rules)
    bg_parsed = parse_rules_from_list(background_rules)

    conflicts = set()
    for i, (d_head, d_body) in enumerate(data_parsed):
        for b_head, b_body in bg_parsed:
            if d_head != b_head:
                continue
            for d in d_body:
                for b in b_body:
                    if d.startswith("not ") and d[4:] == b:
                        conflicts.add(i)
                    elif b.startswith("not ") and b[4:] == d:
                        conflicts.add(i)

    # 非衝突ルールだけ抽出
    cleaned_rules = [data_rules[i] for i in range(len(data_rules)) if i not in conflicts]
    removed_rules = [data_rules[i] for i in conflicts]
    
    return cleaned_rules, removed_rules
    
