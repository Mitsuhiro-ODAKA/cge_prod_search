import networkx as nx
import re
from typing import Optional, List
import ollama
from tqdm import trange

def clean_node_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s

def detect_high_low(text: str) -> Optional[str]:
    text = text.lower()
    if any(word in text for word in ["increase", "higher", "high", "rise", "growth", "more", "up", "↑"]):
        return "high"
    if any(word in text for word in ["decrease", "lower", "low", "drop", "decline", "less", "down", "↓"]):
        return "low"
    return None

def convert_if_then_to_asp(node_a: str, node_b: str, sentence: str) -> Optional[str]:
    m = re.match(r"IF\s+(.+?)\s+THEN\s+(.+)", sentence, re.I)
    if not m:
        return None
    if_part, then_part = m.group(1), m.group(2)

    a_node_clean = clean_node_name(node_a)
    b_node_clean = clean_node_name(node_b)

    a_state = detect_high_low(if_part) or "high"
    b_state = detect_high_low(then_part) or "high"

    a_literal = f"{a_node_clean}_{a_state}"
    b_literal = f"{b_node_clean}_{b_state}"

    return f"{b_literal} :- {a_literal}."

def extract_background_rules(graph: nx.Graph, model="mistral") -> List[str]:
    """
    因果グラフの全ノードペアから背景知識のIF THENルールを抽出し、ASP形式で返す。
    意味のない関係は無視。
    """
    nodes = list(graph.nodes)
    asp_rules = []

    with open("rules/background_rules.lp",mode="r",encoding="utf-8") as f:
        tq1 = []
        line = f.readline()[:-1]
        while line:
            te = line.split(" :- ")
            tq1.append([te[0],te[1][:-1]])
            line = f.readline()[:-1]
    with open("rules/background_rules_negative.lp",mode="r",encoding="utf-8") as f:
        tq2 = []
        line = f.readline()[:-1]
        while line:
            te = line.split(" :- ")
            tq2.append([te[0],te[1][:-1]])
            line = f.readline()[:-1]

    for i in trange(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            node_a = nodes[i]
            node_b = nodes[j]

            if [clean_node_name(node_a)+"_high",clean_node_name(node_b)+"_high"] not in tq1 or [clean_node_name(node_a)+"_high",clean_node_name(node_b)+"_high"] not in tq2:
                print("checking...")
                print([clean_node_name(node_a)+"_high",clean_node_name(node_b)+"_high"])

                # Ollamaに背景知識問い合わせ（英語で質問）
                prompt = f"""
                You are an expert in causal relations.
                Given two variables: A = "{node_a}", B = "{node_b}".
                Please answer ONLY if there is a meaningful causal or logical relationship between A and B in the form:
                IF A is high THEN B increases
                or
                IF A is high THEN B decreases
                If there is no meaningful relationship, answer "No meaningful relation".
                Answer ONLY in English and in the above IF THEN format or "No meaningful relation".
                """

                response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in economic causal knowledge."},
                    {"role": "user", "content": prompt}
                ]
                )

                answer = response['message']['content'].strip()
                if "no meaningful relation" in answer.lower():
                    continue

                asp_rule = convert_if_then_to_asp(node_a, node_b, answer)
                if asp_rule:
                    print("asp_rule exists: ",asp_rule)
                    tw = asp_rule.split(":-")[1]
                    if "low" in tw:
                        tw = tw.replace("_low", "_high")
                        asp_rule = asp_rule.split(":-")[0] + ":- not" + tw
                    else:
                        pass
                    asp_rules.append(asp_rule)
                else:
                    print("asp_rule does not exist.")
                    with open("rules/background_rules_negative.lp",mode="a",encoding="utf-8") as g:
                        g.write(clean_node_name(node_a)+"_high"+" :- "+clean_node_name(node_b)+"_high"+".\n")
                    

    return asp_rules


