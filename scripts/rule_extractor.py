import networkx as nx
import pandas as pd


def extract_boolean_rules(graph: nx.DiGraph, data: pd.DataFrame, threshold_mode="median") -> list:
    """
    因果グラフとデータフレームをもとにブーリアンルール（if-then形式）を生成

    Parameters:
    - graph: nx.DiGraph, 因果関係を持つ有向グラフ
    - data: pd.DataFrame, ノードに対応するデータ（閾値抽出に使用）
    - threshold_mode: "median" または "mean"

    Returns:
    - List of rules (as strings)
    """

    rules = []
    for src, tgt, attr in graph.edges(data=True):
        if src not in data.columns or tgt not in data.columns:
            continue

        if threshold_mode == "median":
            threshold = data[src].median()
        elif threshold_mode == "mean":
            threshold = data[src].mean()
        else:
            raise ValueError("threshold_mode must be 'median' or 'mean'")

        sign = "↑" if data[tgt].corr(data[src]) > 0 else "↓"
        rule = f"IF {src} > {threshold:.2f} THEN {tgt} {sign}"

        rules.append(rule)

    return rules
