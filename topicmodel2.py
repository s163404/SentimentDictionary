# - coding: utf-8 -

import csv
import gensim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from gensim import corpora, models
from orangecontrib.associate.fpgrowth import *
from pandas.tools import plotting
from pprint import pprint
from pymongo import MongoClient
from sklearn.cluster import KMeans

os.environ['MALLET_HOME'] = "C:\\Users\\KMLAB-02\\mallet-2.0.8"


"""
トピックモデルの実装
"""
def extract_aspects():
    W = 100  # トピックから取り出す単語の数
    C = 4   # クラスタ数
    F = 5   # FPM閾値
    # FPM閾値 >= 各商品のトピック数

    DATA = [
        # ファイルパス, 商品名, トピック数, NO_BELOW, NO_ABOVE
        ("Data/Topicmodel/texts_iphone11.csv", "iPhone 11", 4, 1000, 1.0),
        ("Data/Topicmodel/texts_AppleWatch5.csv", "AppleWatch 5", 4, 20, 1.0),
        # ("Data/Topicmodel/texts_FitbitVersa2.csv", "Fitbit Versa2", 4, 20, 1.0),
        ("Data/Topicmodel/texts_GoproHero8.csv", "GoPro Hero8", 4, 20, 1.0),
        # ("Data/Topicmodel/texts_RingfA.csv", "リングフィットアドベンチャー", 4, 20, 1.0),
        ("Data/Topicmodel/texts_AirpodsPro.csv", "AirpodsPro", 4, 20, 1.0)
        # ("Data/Topicmodel/texts_Alldomain_exceptRingfA.csv", "Alldomain", 5, 20, 1.0)   # 全商品
    ]
    print("処理済のテキスト集合を呼び出し")
    texts = []
    for item in DATA:
        path = item[0]
        a_domain_texts = []
        with open(path, 'r', encoding="utf-8") as f:
            for row in csv.reader(f): a_domain_texts.append(row)
        texts.append(a_domain_texts)
    del a_domain_texts, path, f, row, item

    # 商品ごとにトピック抽出⇒単語分布作成
    alldomain_topicid_topic_dict = {}
    topic_id = 0
    for text, data in zip(texts,DATA):
        PNAME = data[1]
        T = data[2]
        NO_BELOW = data[3]
        NO_ABOVE = data[4]

        # no_below =
        print("{0}のテキストコーパス作成～LDA学習".format(PNAME))
        # gensim辞書
        dictionary = corpora.Dictionary(text)
        # フィルタリング
        dictionary.filter_extremes(no_below=NO_BELOW,
                                   no_above=NO_ABOVE)
        # フィルタリング済のdictionaryから単語辞書(dict型)生成　
        dictionary[0]
        id_word_dict = dictionary.id2token
        # LDAに渡すコーパス生成
        corpus = [dictionary.doc2bow(terms) for terms in text]
        print("==============トピック抽出「{0}」========================================".format(PNAME))
        # LDAモデルを生成
        lda_bayes = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    num_topics=T,
                                                    id2word=dictionary,
                                                    random_state=1)
        pprint(lda_bayes.show_topics())

        """ 
        全商品全トピックの単語分布
        alldomain_topic_dict = 
        {
        topic_id: [ (word, prob), (word, prob), ... ],
        topic_id: [ (word, prob), (word, prob), ... ],
            ... 
        }
        """
        for t in range(T):
            distribution = []
            for id_prob in lda_bayes.get_topic_terms(t, W):  # １トピックの単語分布:
                word = id_word_dict[id_prob[0]]     # 抽出された単語
                prob = id_prob[1]                   # 出現確率

                distribution.append((word, prob))

            alldomain_topicid_topic_dict.setdefault(topic_id, distribution)
            topic_id += 1

    del t, id_word_dict, id_prob, distribution, word, prob, lda_bayes, dictionary
    print("="*14 + "トピック抽出終了" + "="*43)

    """
    全商品の全トピックにある単語の辞書を作る
    重複無し
    """
    alldomain_worddict = {}
    id = 1
    for topics in alldomain_topicid_topic_dict.values():
        for topic in topics:
            word = topic[0]
            if word in alldomain_worddict.values(): continue    # 単語の重複チェック
            alldomain_worddict.setdefault(id, word)
            id += 1
    del id, word, topics, topic

    print("K-Meansに渡すtopic_array作成")
    # トピックと商品名の対応
    topic_product_dict = {}
    topic_id = 0
    product_id = 1
    for item in DATA:
        pname = item[1]     # 商品名
        T = item[2]         # 指定したトピック数
        for t in range(T):
            # topic_pname_dict[topic_id] = pname
            topic_product_dict[topic_id] = (pname, product_id)
            topic_id += 1
        product_id += 1


    """
    numpy arrayの辞書版topic_arry_dict topic_arrayを作成
    """
    header = []  # ヘッダ header ⇒[単語id1, 単語id2, 単語id3, ...,]
    # header.append("product_id")
    for id in alldomain_worddict.keys():
        header.append(id)

    topic_array_dict = {}
    topic_id = 0
    for t_id, topics in alldomain_topicid_topic_dict.items():
        row = [0] * len(header)
        product = topic_product_dict[t_id]
        product_id = product[1]
        # row[0] = product_id
        for topic in topics:    # 各単語を照合
            word = topic[0]
            prob = topic[1]
            word_id = [id for id, dword in alldomain_worddict.items() if word == dword].pop(0)
            index = header.index(word_id)
            row[index] = prob
        topic_array_dict.setdefault(topic_id, row)
        topic_id += 1



    # topic_array_dictをK-meansに渡すndarrayに変換
    topic_array = np.empty((0, len(header)), float)
    for row in topic_array_dict.values():
        topic_array = np.append(topic_array, np.array([row]), axis=0)

    del id, topics, topic, word, prob, word_id, index, row, t, pname, item, text, header, topic_id,

    # np.set_printoptions(suppress=True, precision=5, linewidth=100, threshold=np.inf)  # ndarrayのprint表記
    # print(topic_array)

    print("\n=============トピッククラスタリング============")
    kmeans_model = KMeans(n_clusters=C, random_state=1).fit(topic_array)
    labels = kmeans_model.labels_

    # クラスタ-[トピックid]辞書
    cluster_topicids_dict = {}
    for label, topic_id in zip(labels, topic_array_dict.keys()):
        if label not in cluster_topicids_dict.keys(): cluster_topicids_dict[label] = []
        cluster_topicids_dict[label].append(topic_id)

    for cluster, topic_ids in cluster_topicids_dict.items():
        print("---------------------- クラスタ: {0}-----------------------".format(cluster))
        for topic_id in topic_ids:
            product = topic_product_dict[topic_id]
            pname = product[0]
            print(pname, alldomain_topicid_topic_dict[topic_id])


    print("\n" + "="*10 + "頻出パターン抽出(FP-Growth)" + "="*10)
    """
    クラスタごとに、属するトピックの単語idリストを作る  = transaction
    ex)
    クラスタAにはトピック1, 3, 5が属するとき、
    クラスタAのトランザクションは
    transactionA = [
        [トピック１の単語id1, １の単語id2, １の単語id3,...]
        [トピック２の単語id1, ２の単語id2, ２の単語id3,...]
        [トピック３の単語id1, ３の単語id2, ３の単語id3,...]
    ]    
    """
    for cluster, topic_ids in cluster_topicids_dict.items():
        transactions = []   # クラスタの単語idリストのリスト
        for topic_id in topic_ids:  # トピック番号[0, 1, 4]
            w_ids = []
            for item in alldomain_topicid_topic_dict[topic_id]: # 各トピック
                word = item[0]
                for key, value in alldomain_worddict.items():
                    if value == word:
                        w_ids.append(key)
                        break
            transactions.append(w_ids)

        print("-"*20 + "クラスタ: {0} ".format(cluster) + "-"*20)
        for topic_id in topic_ids:
            product = topic_product_dict[topic_id]
            print(product[0] + ", ", end="")
        print("\n")

        itemsets = frequent_itemsets(transactions, F)    # 頻出ワード集合
        for tuple in itemsets:
            # print(tuple, end="")
            print("  { ", end="")
            for id in tuple[0]:
                word = alldomain_worddict[id]
                print(word + ", ", end="")
            print("}", end="")
            print(tuple[1])

    print("終了")


def main():
    extract_aspects()


if __name__ == '__main__':
    main()
    sys.exit()
