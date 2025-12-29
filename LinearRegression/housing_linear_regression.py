import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Housing Linear Regression")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    # 日本語フォント設定（Mac用）
    plt.rcParams['font.family'] = 'Hiragino Sans'
    return (
        LinearRegression,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Housing データセットの線形回帰分析

    UCI Machine Learning Repository の Housing データセットを使用して線形回帰を行います。
    """)
    return


@app.cell
def _(pd):
    # データの読み込み
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    column_names = [
        "CRIM",     # 犯罪率
        "ZN",       # 住宅地の割合
        "INDUS",    # 非小売業の割合
        "CHAS",     # チャールズ川沿いかどうか
        "NOX",      # 窒素酸化物濃度
        "RM",       # 平均部屋数
        "AGE",      # 古い住宅の割合
        "DIS",      # 雇用センターまでの距離
        "RAD",      # 高速道路へのアクセス
        "TAX",      # 固定資産税率
        "PTRATIO",  # 生徒教師比率
        "B",        # 黒人比率
        "LSTAT",    # 低所得者割合
        "MEDV"      # 住宅価格（目的変数）
    ]

    df = pd.read_csv(url, sep=r"\s+", names=column_names)
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## データの概要

    - サンプル数: {len(df)}
    - 特徴量数: {len(df.columns) - 1}
    - 目的変数: MEDV（住宅価格の中央値、単位: $1000）
    """)
    return


@app.cell
def _(df):
    # 基本統計量
    df.describe()
    return


@app.cell
def _(df, mo):
    # 特徴量選択用のチェックボックス
    feature_options = [col for col in df.columns if col != "MEDV"]
    feature_selector = mo.ui.multiselect(
        options=feature_options,
        value=feature_options,  # デフォルトで全て選択
        label="使用する特徴量を選択"
    )
    feature_selector
    return (feature_selector,)


@app.cell
def _(mo):
    # テストデータの割合を選択
    test_size_slider = mo.ui.slider(
        start=0.1,
        stop=0.5,
        step=0.05,
        value=0.2,
        label="テストデータの割合"
    )
    test_size_slider
    return (test_size_slider,)


@app.cell
def _(
    LinearRegression,
    df,
    feature_selector,
    test_size_slider,
    train_test_split,
):
    # 選択された特徴量でモデルを学習
    selected_features = feature_selector.value if feature_selector.value else ["RM"]

    X = df[list(selected_features)]
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_slider.value, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, selected_features, y_pred, y_test


@app.cell
def _(mean_squared_error, mo, np, r2_score, y_pred, y_test):
    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mo.md(
        f"""
        ## モデルの評価結果

        | 指標 | 値 |
        |------|-----|
        | MSE (平均二乗誤差) | {mse:.4f} |
        | RMSE (二乗平均平方根誤差) | {rmse:.4f} |
        | R² スコア | {r2:.4f} |
        """
    )
    return


@app.cell
def _(mo, model, pd, selected_features):
    # 回帰係数の表示
    coef_df = pd.DataFrame({
        "特徴量": selected_features,
        "係数": model.coef_
    }).sort_values("係数", key=abs, ascending=False)

    mo.md(
        f"""
        ## 回帰係数

        切片: {model.intercept_:.4f}
        """
    )
    return (coef_df,)


@app.cell
def _(coef_df):
    coef_df
    return


@app.cell
def _(plt, y_pred, y_test):
    # 予測値 vs 実測値のプロット
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("実測値 (MEDV)")
    ax.set_ylabel("予測値")
    ax.set_title("予測値 vs 実測値")
    plt.tight_layout()
    fig
    return


@app.cell
def _(plt, y_pred, y_test):
    # 残差プロット
    residuals = y_test - y_pred

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("予測値")
    ax2.set_ylabel("残差")
    ax2.set_title("残差プロット")
    plt.tight_layout()
    fig2
    return


if __name__ == "__main__":
    app.run()
