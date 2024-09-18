import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def run_main_app():
    # 重回帰分析アプリケーション
    st.title("データ分析アプリ")

    # サイドバーでのモデル選択
    st.sidebar.header("モデルの選択")
    model_type = st.sidebar.selectbox("分析モデルを選択してください", ["重回帰分析", "ロジスティック回帰"])

    # 指標の説明
    st.sidebar.title("指標の意味")
    
    if model_type == "重回帰分析":
        st.sidebar.markdown("""
            ### 決定係数 (R²)
                従属変数（目的変数）の分散がどれだけ説明変数で説明できているかを示す指標です。値が1に近いほどモデルの説明力が高いことを示します。

            ### 平均二乗誤差 (MSE)
                予測値と実際の値の差の二乗平均を表します。値が小さいほど、モデルの予測精度が高いことを示します。

            ### 回帰係数
                各説明変数が従属変数にどの程度影響を与えるかを示します。正の値は正の影響、負の値は負の影響を示します。

            ### 切片 (Intercept)
                説明変数がすべてゼロのときの従属変数の値を示します。
        """)
    elif model_type == "ロジスティック回帰":
        st.sidebar.markdown("""
            ### 正解率 (Accuracy)
                予測結果が実際の値と一致する割合を示します。値が1に近いほどモデルの精度が高いことを示します。

            ### 回帰係数
                各説明変数が従属変数にどの程度影響を与えるかを示します。正の値は正の影響、負の値は負の影響を示します。

            ### 切片 (Intercept)
                説明変数がすべてゼロのときの従属変数の値を示します。
        """)

    # サイドバーでのデータファイルのアップロード
    st.sidebar.header("CSVファイルをアップロード")
    uploaded_file = st.sidebar.file_uploader("ファイルを選択してください", type=["csv"])

    # データがアップロードされた場合
    if uploaded_file is not None:
        # データの読み込み
        data = pd.read_csv(uploaded_file)

        st.write("アップロードされたデータ:")
        st.write(data.head())

        # 説明変数と目的変数の選択
        st.sidebar.header("説明変数と目的変数の選択")

        # 目的変数の選択
        target_column = st.sidebar.selectbox("目的変数（Y）を選択", data.columns)

        # 説明変数の選択（目的変数も含めることができる）
        feature_columns = st.sidebar.multiselect("説明変数（X）を選択", data.columns)

        if len(feature_columns) > 0 and target_column:
            # 説明変数と目的変数の設定
            X = data[feature_columns]

            # カテゴリ変数のダミー変換
            X = pd.get_dummies(X, drop_first=True)

            # 目的変数を再設定
            y = data[target_column]

            st.write("ダミー変換後のデータ:")
            st.write(X.head())

            # データをトレーニングとテストに分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "重回帰分析":
                # 重回帰モデルの作成
                model = LinearRegression()
                model.fit(X_train, y_train)

                # 予測と評価
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # 結果の表示
                st.write(f"決定係数 (R^2): {r2:.3f}")
                st.write(f"平均二乗誤差 (MSE): {mse:.3f}")

                # 回帰係数の表示
                st.write("回帰係数:")
                coef_df = pd.DataFrame({
                    '特徴量': X.columns,
                    '係数': model.coef_
                })
                st.write(coef_df)

            elif model_type == "ロジスティック回帰":
                # 目的変数がカテゴリ（0,1など）であることを確認
                y_train = pd.to_numeric(y_train, errors='coerce')
                y_test = pd.to_numeric(y_test, errors='coerce')

                # ロジスティック回帰モデルの作成
                model = LogisticRegression()
                model.fit(X_train, y_train)

                # 予測と評価
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # 結果の表示
                st.write(f"正解率 (Accuracy): {accuracy:.3f}")

                # 回帰係数の表示
                st.write("回帰係数:")
                coef_df = pd.DataFrame({
                    '特徴量': X.columns,
                    '係数': model.coef_[0]
                })
                st.write(coef_df)

            # ヒートマップ (全ての特徴量と目的変数の相関) の描画
            st.write("全ての特徴量と目的変数の相関ヒートマップ:")
            # 説明変数と目的変数を連結して相関行列を作成
            data_with_target = X.copy()
            data_with_target[target_column] = y
            correlation_matrix = data_with_target.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("全ての特徴量と目的変数の相関")
            st.pyplot(plt)
            plt.close()

            # 箱ひげ図を一つにまとめて描画
            st.write("箱ひげ図（全ての説明変数）:")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=X)
            plt.title("箱ひげ図：全ての説明変数")
            st.pyplot(plt)
            plt.close()

            # 新たなCSVファイルを読み込んで、目的変数を推定
            st.sidebar.header("テスト用のCSVファイルをアップロード")
            test_file = st.sidebar.file_uploader("テストデータファイルを選択してください", type=["csv"], key="test_data")

            if test_file is not None:
                test_data = pd.read_csv(test_file)
                st.write("アップロードされたテストデータ:")
                st.write(test_data.head())

                # 説明変数のみを使用するため、同じカラムをダミー変数化
                test_data_processed = pd.get_dummies(test_data, drop_first=True)

                # トレーニング時と同じ特徴量に揃える
                missing_cols = set(X.columns) - set(test_data_processed.columns)
                for col in missing_cols:
                    test_data_processed[col] = 0
                test_data_processed = test_data_processed[X.columns]

                # 目的変数の推定
                y_test_pred = model.predict(test_data_processed)

                # 推定結果を表示
                st.write("推定された目的変数:")
                st.write(pd.DataFrame({target_column: y_test_pred}))

        else:
            st.write("説明変数を選択してください。")
    else:
        st.write("左側のサイドバーからCSVファイルをアップロードしてください。")

if __name__ == '__main__':
    run_main_app()
