# - coding: utf-8 -

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# --- データ準備 ---
print("--- libsvmファイル読み込み ---")

x, y = load_svmlight_file('Data/svm_fmt1129.txt')

# データを訓練データとテストデータ(20%)に分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y,test_size=0.2, random_state=0)        # いずれシード値は指定なしで


# --- モデル作成 clf = classifier ---
print("--- モデル作成 ---")
# clfの生成はどちらか一方だけ実行する
clf = LogisticRegression(solver='lbfgs')        # solver 最適化手法
#clf = joblib.load("sklearn/classifier.pkl")     # モデル読み込み

# トレーニングデータでモデルの学習
clf.fit(x_train, y_train)

# 学習の精度
print("トレーニングデータの学習精度:" + str(clf.score(x_train, y_train)))

# テストデータで予測
predict = clf.predict(x_test)
# テストデータの精度
print("テストデータの予測精度:" + str(clf.score(x_test, y_test)))

# --- モデルの評価 ---
print("--- モデルの評価 (交差検証) ---")
scores = cross_val_score(clf, x, y)
# 各分割のスコア
print('Cross-Validation scores: {}'.format(scores))
# スコアの平均
print('Average score: {}'.format(np.mean(scores)))


# --- モデルの保存 ---
joblib.dump(clf, "Data/sklearn/classifier_lbfgs.pkl", compress=True)

# モデルの各パラメータ
print("バイアス:\n" + str(clf.intercept_))
print("重み\n" + str(clf.coef_))
np.savetxt("Data/sklearn/coefs_lbfgs.txt", clf.coef_, delimiter='\n', fmt='%.18f')

print("\n--- end of execution^^ ---")
