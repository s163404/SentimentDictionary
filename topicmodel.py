# - coding: utf-8 -

import csv
import emoji
import gensim
import matplotlib.pyplot as plt
import numpy as np
import os
import pyfpgrowth
import re
import sys
import tools
from gensim import corpora, models
from orangecontrib.associate.fpgrowth import *
from pandas.tools import plotting
from pprint import pprint
from pymongo import MongoClient
from sklearn.cluster import KMeans

os.environ['MALLET_HOME'] = "C:\\Users\\KMLAB-02\\mallet-2.0.8"


"""
テキストを単語のリストに変換
入力: テキスト集合
出力: 単語リストのリスト
- テキスト前処理
- 条件に合う名詞を取り出す
"""
def trans_texts(N:int):
    processed_texts = []
    stopwords = []
    with open("Data/Topicmodel/stopwords.txt", 'r', encoding="utf-8") as f:
        for word in f.read().split('\n'):
            if word: stopwords.append(word)

    print("テキスト読み込み")
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    """
    入出力先設定
    """
    col = db.Alldomain_exceptRingfA # 入力
    csv_path = "Data/Topicmodel/texts_Alldomain_exceptRingfA.csv"  # 出力先
    # csv_path = "Data/Topicmodel/texts_AirpodsPro.csv"  # 出力先
    # csv_path = "Data/Topicmodel/texts_AppleWatch5.csv"  # 出力先
    # csv_path = "Data/Topicmodel/texts_RingfA.csv"  # 出力先
    # csv_path = "Data/Topicmodel/texts_GoproHero8.csv"   # 出力先
    # csv_path = "Data/Topicmodel/texts_FitbitVersa2.csv"   # 出力先

    docs = col.find(projection={'_id': 0})
    raw_texts = []
    for doc in docs:
        if len(raw_texts) == N: break  # 文書数
        if doc["text"] is not None: raw_texts.append(doc["text"])

    print("テキストを単語のリストで表現")
    for text in raw_texts:
        words = []
        frequency = {}
        # 生テキストを前処理
        text = tools.pre_process(text)

        if not text: continue
        # 単語単位の前処理
        nodes = tools.mecab_tolist(text)
        for node in nodes:
            flag = False
            word = node[0]
            # ストップワードをスキップ
            for stopword in stopwords:
                if word == stopword:
                    flag = True
                    break
            if flag is True: continue
            if "iphone" in word: continue
            if len(word) == 1 and re.search(r'[\u30A1-\u30F4]', word): continue
            if len(word) == 1 and re.search(r'[あ-ん]', word): continue
            if len(word) == 1 and re.search(r'[a-z]', word): continue
            # 名詞を抜き出す
            if node[1] == "名詞" \
                    and node[2] != "接尾" \
                    and node[2] != "数" \
                    and node[2] != "サ変接続" \
                    and node[2] != "特殊" \
                    and node[2] != "引用文字列" \
                    and node[2] != "接続詞的" \
                    and node[2] != "形容動詞語幹" \
                    and node[2] != "固有名詞":
                words.append(word)

        if words: processed_texts.append(words)

            # 名詞を抜き出す
        #     if node[1] == "名詞" and node[2] != "接尾" and node[2] != "数" and node[2] != "サ変接続":
        #         if word not in frequency: frequency[word] = 1
        #         else: frequency[word] += 1
        #
        # for word, freq in frequency.items():
        #     if freq >= 1: words.append(word)
        # if words: processed_texts.append(words)

    # CSV化
    with open(csv_path, 'w', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(processed_texts)
        f.close()

    return processed_texts


"""
トピックモデルの実装
"""
def lda_test():
    T = 5   # トピック数
    W = 20  # トピックから取り出す単語の数
    C = 3   # クラスタ数
    M = 50000   # テキスト数
    NO_BELOW = 100    # 出現文書数が指定値以下の語を除外    目安300
    NO_ABOVE = 1.0  # 10割以上の文書に登場した語除外
    INPUT_PATH = "Data/Topicmodel/texts_iphone11.csv"

    print("処理済のテキスト集合(単語リスト)を呼び出し")
    texts = []
    with open(INPUT_PATH, 'r', encoding="utf-8") as f:
        for row in csv.reader(f): texts.append(row)

    print("テキストコーパス作成～LDA学習")

    # gensim辞書
    dictionary = corpora.Dictionary(texts)
    # フィルタリング
    dictionary.filter_extremes(no_below=NO_BELOW,
                               no_above=NO_ABOVE)
    # フィルタリング済のdictionaryから単語辞書(dict型)生成　
    word_id_dict = dictionary.token2id
    dictionary[0]
    id_w_dict = dictionary.id2token
    # コーパス生成
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("\n==============トピック抽出==============")
    # トピックを持つLDAモデルを作成
    lda_bayes = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                num_topics=T,
                                                id2word=dictionary,
                                                random_state=1)
    # for t in range(T): print(t, lda_bayes.get_topic_terms(t, W))
    pprint(lda_bayes.show_topics())

    print("トピック抽出結果をnumpy行列に加工")
    """
    term_wid_prob_list =
    [
        [(t1, id1_1, prob), (t1, id1-2, prob), ..., (t1, id1-W, prob)],
        [(t2, id2-1, prob), (t2, id2-2, prob), ..., (t2, id1-W, prob)],
        [(t3, id3-1, prob), (t3, id3-2, prob), ..., (t3, id1-W, prob)],
        ...
        [(tT, idT-1, prob), (tT, idT-2, prob), ..., (tT, idT-W, prob)]
    ]
    トピック数T×(トピック番号t, 単語id, 出現確率)の出現確率のリストをつくる    
    """
    topic_wid_prob_list = []
    for t in range(T):
        one_topic = []
        for tuple in lda_bayes.get_topic_terms(t, W):
            tuple = (t,) + tuple    # (トピック番号, 単語id, 出現確率)
            one_topic.append(tuple)
        topic_wid_prob_list.append(one_topic)

    #ヘッダ行を作る header ⇒[トピック番号, "単語id", "単語id", ...]
    header = []
    # header.append("topicnum")                                   # ====トピック番号を考慮する場合====
    for topic in topic_wid_prob_list:
        for tuple in topic:
            if tuple[1] not in header: header.append(tuple[1])  # 未登録の単語idを登録

    topic_array = np.empty((0, len(header)), float)  # K-meansに渡すnumpyparray
    for topic in topic_wid_prob_list:
        insert_list = [0] * len(header)
        t = topic[0]
        # insert_list[0] = t[0]                                   # ====トピック番号を考慮する場合====
        for tuple in topic:
            index = header.index(tuple[1])
            if index: insert_list[index] = tuple[2]
        topic_array = np.append(topic_array, np.array([insert_list]), axis=0)

    # np.set_printoptions(suppress=True, precision=5, linewidth=100, threshold=np.inf)  # ndarrayのprint表記
    # print(topic_array)

    print("\n=============トピッククラスタリング============")
    # クラスタ数の検討用
    distortions = []
    for i in range(1, C):  # 1~10クラスタまで一気に計算
        km = KMeans(n_clusters=i, random_state=1)
        km.fit(topic_array)  # クラスタリングの計算を実行
        distortions.append(km.inertia_)  # km.fitするとkm.inertia_が得られる inertia=重心との二乗誤差
    plt.plot(range(1, C), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

    kmeans_model = KMeans(n_clusters=C, random_state=1).fit(topic_array) # K-Meansクラスタリング
    labels = kmeans_model.labels_

    clus_topic = {}     # {cluster: [topicnums] } keyはクラスタラベル、valueはトピック番号リスト

    # for label, tarray in zip(labels, topic_array):
    #     if label not in clus_topic: clus_topic[label] = []
    #     topicnum = int(tarray[0])
    #     clus_topic[label].append(topicnum)
    # 上：====トピック番号を考慮する場合====
    # 下：====トピック番号を考慮しない場合====
    for label, topicnum in zip(labels, range(T)):
        if label not in clus_topic: clus_topic[label] = []
        clus_topic[label].append(topicnum)

    for cluster, topicnums in clus_topic.items():
        print("クラスタ: {0}".format(str(cluster)))
        for t in topicnums:
            print(t, lda_bayes.print_topic(t, W))
            print(lda_bayes.get_topic_terms(t, W))
        print("-" * 150)


    print("\n=============頻出パターン抽出(FP-Growth)=============")
    # 各クラスタに属する各トピックの単語のidを頻出パターン抽出
    for cluster, tnum_list in clus_topic.items():   # clus_topic = {クラスタ番号: トピック番号リスト }
        print("クラスタ: {0}".format(str(cluster)))
        transactions = []
        for topic in tnum_list:
            w_ids = []
            for tuple in lda_bayes.get_topic_terms(topic, W):
                w_ids.append(int(tuple[0]))
            transactions.append(w_ids)

        itemsets = frequent_itemsets(transactions, 2)   # 頻出アイテム集合
        for tuple in itemsets:
            print(tuple, end="")
            print(" ⇒ { ", end="")
            for id in tuple[0]:
                word = id_w_dict[id]
                print(word + ", ", end="")
            print("｝", end="")
            print(tuple[1])



    print("終了")


def demo():
    transactions = [
        [1, 2, 5],
        [2, 4],
        [2, 3],
        [1, 2, 4],
        [1, 3],
        [2, 3],
        [1, 3],
        [1, 2, 3, 5],
        [1, 2, 3]
    ]
    itemsets = frequent_itemsets(transactions, 2)
    print(list(itemsets))


def main():
    # trans_texts(50000)
    lda_test()
    # demo()


if __name__ == '__main__':
    main()
    sys.exit()
