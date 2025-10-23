import re
from gamspy import Container, Variable, Parameter, Equation, Model, Set
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict
from collections import defaultdict
import pandas as pd

def to_gams_safe_name(name: str) -> str:
    """GAMSに使える形式に変換"""
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    safe_name = re.sub(r'_+', '_', safe_name).strip('_').lower()
    return safe_name

def build_custom_nested_cge_model(
    rules: List[str],
    df: pd.DataFrame,
    regression_targets: List[str],
    main_structure_rules: List[str],
    main_output: str = "gdp_current_us_high"
) -> Tuple[Container, Model]:
    m = Container()
    variables, equations = {}, []

    # 安全名変換マップを作成しデータフレームに適用
    name_map = {col: to_gams_safe_name(col) for col in df.columns}
    df = df.rename(columns=name_map)
    safe_main_output = to_gams_safe_name(main_output)
    safe_regression_targets = [to_gams_safe_name(t) for t in regression_targets]

    # ステップ1: 回帰対象の変数に対するルール抽出
    regressions = defaultdict(set)
    for rule in rules:
        rule = rule.strip().strip(".")
        if ":-" in rule:
            head, body = rule.split(":-")
            head = to_gams_safe_name(head.strip())
            if head in safe_regression_targets:
                body_vars = [to_gams_safe_name(b.strip().replace("not ", "")) for b in body.strip().split(",")]
                regressions[head].update(body_vars)

    # ステップ2: 回帰推定
    regression_results = {}
    for target, inputs in regressions.items():
        X, y = df[list(inputs)], df[target]
        model = LinearRegression().fit(X, y)
        regression_results[target] = (model.intercept_, dict(zip(inputs, model.coef_)))

    # ステップ3: メイン構造の抽出
    main_inputs = []
    for rule in main_structure_rules:
        rule = rule.strip().strip(".")
        if ":-" in rule:
            head, body = rule.split(":-")
            head = to_gams_safe_name(head.strip())
            body_vars = [to_gams_safe_name(b.strip()) for b in body.strip().split(",")]
            if head == safe_main_output:
                main_inputs.extend(body_vars)

    # ステップ4: セット定義
    t_set = Set(m, name="t", records=list(regressions.keys()))
    x_set = Set(m, name="x", records=list({v for s in regressions.values() for v in s}))

    # ステップ5: 変数登録
    all_vars = set([safe_main_output] + main_inputs + list(regressions.keys()) + list(x_set.records))
    for v in all_vars:
        var_type = "free" if v == safe_main_output else "positive"
        variables[v] = Variable(m, name=v, type=var_type)

    # ステップ6: パラメータ登録と回帰式定義
    beta0 = Parameter(m, name="beta0", domain=["t"])
    beta = Parameter(m, name="beta", domain=["t", "x"])
    for t, (intercept, coefs) in regression_results.items():
        beta0[t] = intercept
        for x, coef in coefs.items():
            beta[t, x] = coef

    for t in t_set.records:
        expr = beta0[t]
        for x in x_set.records:
            if (t, x) in beta.records:
                expr += beta[t, x] * variables[x]
        eq = Equation(m, name=f"eq_{t}")
        eq[...] = variables[t] == expr
        equations.append(eq)

    # ステップ7: Cobb-Douglas 関数の定義
    A = Parameter(m, name="A")
    A[...] = 1.0
    rhs_expr = A
    for v in main_inputs:
        alpha = Parameter(m, name=f"alpha_{v}")
        alpha[...] = round(1.0 / len(main_inputs), 2)
        rhs_expr *= variables[v] ** alpha
    eq_main = Equation(m, name=f"eq_{safe_main_output}")
    eq_main[...] = variables[safe_main_output] == rhs_expr
    equations.append(eq_main)

    # モデル定義
    model = Model(
        m,
        name="custom_nested_cge_model",
        equations=equations,
        problem="NLP",
        sense="max",
        objective=variables[safe_main_output]
    )
    
    print(model.solve())
    results = []
    for name, var in variables.items():
        if hasattr(var, "level"):
            results.append({"variable": name, "value": var.level})
        else:
            print(f"Skipping '{name}': not a Variable object.")
    print(pd.DataFrame(results))
    return m


from IPython.display import Markdown, display
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import pandas as pd

def generate_latex_equations_with_coefficients(
    rules,
    regression_targets,
    main_structure_rules,
    df,
    main_output="gdp_current_us_high"
):
    """
    cleaned_data_rules + データフレーム df に基づき、回帰式とCobb-Douglas式を
    Notebookにマークダウン表示。各目的変数に1本ずつ、回帰係数を埋め込み。
    """
    regressions = defaultdict(list)
    main_inputs = []

    # 回帰対象の説明変数を収集
    for rule in rules:
        if ":-" in rule:
            head, body = rule.split(":-")
            head = head.strip()
            if head in regression_targets:
                body_vars = [b.strip() for b in body.strip().strip(".").split(",")]
                regressions[head].extend(body_vars)

    # 各目的変数に対して一度だけ回帰（notを除いた列名で学習）
    regression_results = {}
    for target in regression_targets:
        inputs = list(set(regressions[target]))
        safe_inputs = [var[4:] if var.startswith("not ") else var for var in inputs]

        # 安全にサブセット抽出
        missing = [col for col in safe_inputs if col not in df.columns]
        if missing:
            print(f"Warning: Missing columns for target '{target}': {missing}")
            continue

        if target not in df.columns:
            print(f"Warning: Target column '{target}' not in DataFrame.")
            continue

        X = df[safe_inputs]
        y = df[target]

        if X.empty or y.empty:
            print(f"Skipping regression for {target} due to empty data.")
            continue

        model = LinearRegression().fit(X, y)
        regression_results[target] = (model.intercept_, dict(zip(inputs, model.coef_)))

    # Cobb-Douglas の構成要素
    for rule in main_structure_rules:
        if ":-" in rule:
            head, body = rule.split(":-")
            head = head.strip()
            if head == main_output:
                main_inputs = [b.strip() for b in body.strip().strip(".").split(",")]

    latex_lines = []

    # 回帰式（各目的変数ごとに1本）
    for target, (intercept, coefs) in regression_results.items():
        target_latex = target.replace("_", r"\_")
        terms = [f"{intercept:.2f}"]
        for i, (var, coef) in enumerate(coefs.items()):
            if var.startswith("not "):
                var_clean = var[4:].replace("_", r"\_")
                terms.append(fr"{coef:.2f} \cdot (1 - {var_clean})")
            else:
                var_clean = var.replace("_", r"\_")
                terms.append(fr"{coef:.2f} \cdot {var_clean}")
        eq = fr"$$\text{{{target_latex}}} = " + " + ".join(terms) + "$$"
        latex_lines.append(eq)

    # Cobb-Douglas式
    output_latex = main_output.replace("_", r"\_")
    A_term = "A"
    cd_terms = []
    for i, var in enumerate(main_inputs):
        var_clean = var.replace("_", r"\_")
        alpha = fr"\alpha_{{{i+1}}}"
        cd_terms.append(fr"(\text{{{var_clean}}})^{{{alpha}}}")
    rhs_cd = A_term + r" \cdot " + r" \cdot ".join(cd_terms)
    eq_cd = fr"$$\text{{{output_latex}}} = {rhs_cd}$$"
    latex_lines.append(eq_cd)

    display(Markdown("\n\n".join(latex_lines)))


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from gamspy import Container, Variable, Parameter, Equation, Model

def estimate_alpha_and_build_cd_model_constrained(df, output_var, input_vars):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from gamspy import Container, Variable, Parameter, Equation, Model

    # 対数変換と欠損除去
    log_X = np.log(df[input_vars].replace(0, np.nan))
    log_y = np.log(df[output_var].replace(0, np.nan))
    Xy = pd.concat([log_X, log_y], axis=1).dropna()
    log_X_clean = Xy[input_vars]
    log_y_clean = Xy[output_var]

    # 線形回帰で係数推定
    reg = LinearRegression().fit(log_X_clean, log_y_clean)
    raw_coefs = dict(zip(input_vars, reg.coef_))

    # ✅ 負の係数を0に、合計が0にならないようにε加算
    positive_coefs = {k: max(v, 0.0) for k, v in raw_coefs.items()}
    total = sum(positive_coefs.values()) + 1e-8  # ゼロ除算回避
    alpha_values = {k: v / total for k, v in positive_coefs.items()}

    intercept = np.exp(reg.intercept_)

    # GAMSpy モデル構築
    m = Container()
    variables = {}
    for var in input_vars + [output_var]:
        var_type = "free" if var == output_var else "positive"
        variables[var] = Variable(m, name=var, type=var_type)

    A = Parameter(m, name="A")
    A[...] = intercept

    alpha_params = {}
    for var in input_vars:
        alpha = Parameter(m, name=f"alpha_{var}")
        alpha[...] = alpha_values[var]
        alpha_params[var] = alpha

    epsilon = 1e-3
    rhs_expr = A
    for var in input_vars:
        rhs_expr *= (variables[var] + epsilon) ** alpha_params[var]

    eq = Equation(m, name="cd_eq")
    eq[...] = variables[output_var] == rhs_expr

    model = Model(
        m,
        equations=[eq],
        problem="NLP",
        sense="max",
        objective=variables[output_var]
    )

    return model, alpha_values
