import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from lingam import DirectLiNGAM
import re

def save_edge_weights_to_csv(G, filepath="graph_edges.csv"):
    """
    NetworkXグラフGのエッジと重みをCSVに保存する関数

    Parameters:
    - G: networkx.DiGraph または Graph
    - filepath: 保存先のCSVファイルパス
    """
    edge_data = []

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", None)  # weight属性がない場合None
        edge_data.append({"source": u, "target": v, "weight": weight})

    df = pd.DataFrame(edge_data)
    df.to_csv(filepath, index=False)
    print(f"エッジ情報を {filepath} に保存しました。")


def estimate_and_plot_causal_graph(config: dict):
    # データの読み込みと列の除去
    df = pd.read_csv(config["input_csv"], encoding="utf-8")
    df.drop(columns=config["drop_columns"], errors='ignore', inplace=True)

    # データの標準化（Zスコア）
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # モデルの適用
    model = DirectLiNGAM()
    model.fit(df_scaled.values)

    adj_matrix = model.adjacency_matrix_
    columns = list(df.columns)

    # 因果グラフの作成
    G = nx.DiGraph()
    G.add_nodes_from(columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            weight = adj_matrix[i, j]
            if abs(weight) > config["threshold"]:
                G.add_edge(columns[i], columns[j], weight=weight)

    # レイアウト（Graphviz優先、失敗時にspring）
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception as e:
        print("Graphviz layout failed (dot overflow?), falling back to spring layout.")
        pos = nx.spring_layout(G, seed=config["seed"])

    # 描画
    plt.figure(figsize=tuple(config["figsize"]))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        arrowsize=20,
        font_size=9,
        font_weight="bold",
        edge_color="gray"
    )
    # エッジラベルの描画（係数を小数第2位で表示）
    edge_labels = nx.get_edge_attributes(G, "weight")
    formatted_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels)

    plt.title("Estimated Causal Graph (DirectLiNGAM, standardized)")
    
    plt.show()
    
    save_edge_weights_to_csv(G, "data/causal_graph_edges.csv")

    return G, adj_matrix
    

def plot_causal_graph_from_rules(observed_rules):
    """
    観測ルールのリストから有向因果グラフを構築し、矢印付きで可視化する関数。
    
    Parameters:
        observed_rules (list of str): 観測ルール（例: "y :- x.", "y :- not x."）
    """
    G = nx.DiGraph()

    for rule in observed_rules:
        if ":-" not in rule:
            continue
        head, body = rule.split(":-")
        head = head.strip().replace('.', '')
        body = body.strip().replace('.', '').split(',')

        for cond in body:
            cond = cond.strip()
            is_negative = cond.startswith('not ')
            var = cond[4:].strip() if is_negative else cond

            G.add_edge(
                var,
                head,
                color='red' if is_negative else 'green',
                label='¬' if is_negative else ''
            )

    # グラフ描画
    pos = nx.spring_layout(G, seed=42)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=50,  # 矢印サイズを強調
        width=2,
        # connectionstyle='arc3,rad=0.1'  # 曲線で交差を避ける
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=9)

    plt.title("Directed Causal Graph from Observed Rules")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    save_edge_weights_to_csv(G, "data/causal_graph_edges_cleaned.csv")
