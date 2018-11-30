# - coding: utf-8 -

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


# シグモイド関数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# https://momonoki2017.blogspot.com/2018/02/scikit-learn.html
def LR_1():
    # 乳がんデータセットを読み込む
    dataset = datasets.load_breast_cancer()
    x, y = dataset.data, dataset.target

    # ロジスティック回帰
    clf = LogisticRegression()
    # Stratifired K-Fold CV　で性能を評価する
    skf = StratifiedKFold(shuffle=True)
    scoring = {
        'acc': 'accuracy',
        'auc': 'roc_auc',
    }
    scores = cross_validate(clf, x, y, cv=skf, scoring=scoring)

    print('Accuracy(mean):', scores['test_acc'].mean())
    print('AUC(mean):', scores['test_auc'].mean())


'''
コンソール実行
https://momonoki2017.blogspot.com/2018/02/scikit-learn.html
'''
def LR_2():
    # 乳がんデータセット
    dataset = datasets.load_breast_cancer()

    # datasetを訓練データとテストデータ(20%)に分割
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=0)

    # 分割後のデータ件数
    print('train =', len(X_train))
    print('test =', len(X_test))

    # モデル作成
    clf = LogisticRegression()

    # 訓練データで学習
    clf.fit(X_train, y_train)
    clf.score(X_train, y_train) # 学習精度

    # テストデータで予測
    predict = clf.predict(X_test)
    clf.score(X_test, y_test)

    # シグモイド関数プロット
    plt.xkcd()
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('sigmoid function')


    # 決定係数値
    X_test_value = clf.decision_function(X_test)
    # 決定間数値をシグモイド関数で確率に変換
    X_test_prob = sigmoid(X_test_value)

    # テストデータの予測結果(Pandasにデータを格納して確認)
    pd.set_option("display.max_rows", 114)
    cancer_test_df = pd.DataFrame(y_test, columns=['正解値'])
    cancer_test_df['予測値'] = pd.Series(predict)
    cancer_test_df['正誤'] = pd.Series(y_test==predict)
    cancer_test_df['決定間数値'] = pd.Series(X_test_value)
    cancer_test_df['確立(良性)'] = pd.Series(X_test_prob).round(4)
    cancer_test_df

# https://algorithm.joho.info/machine-learning/python-scikit-learn-logistic-regression/
def LR_3():
    # 学習用のデータを読み込み
    data = pd.read_csv("train.csv", sep=",")

    # 説明変数：x1, x2
    X = data.loc[:, ['x1', 'x2']].as_matrix()

    # 目的変数：x3
    y = data['x3'].as_matrix()

    # 学習（ロジスティック回帰）
    clf = LogisticRegression(random_state=0)
    clf.fit(X, y)

    # ロジスティック回帰の学習結果
    a = clf.coef_
    b = clf.intercept_
    print("回帰係数:", a)
    print("切片:", b)
    print("決定係数:", clf.score(X, y))

    # テスト用データの読み込み
    data = pd.read_csv("test.csv", sep=",")

    # 学習結果の検証（テスト用データx1, x2を入力）
    X_test = data.loc[:, ['x1', 'x2']].as_matrix()
    y_predict = clf.predict(X_test)

    # 検証結果の表示
    print("検証結果：", y_predict)

    # 学習結果を出力
    joblib.dump(clf, 'train.learn')


if __name__ == '__main__':
    LR_3()


sys.exit()