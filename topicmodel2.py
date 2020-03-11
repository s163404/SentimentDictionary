# - coding: utf-8 -

import csv
import gensim
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import tools
from gensim import corpora, models
from gensim.models import HdpModel
from orangecontrib.associate import *
import pandas as pd
from pandas import plotting
from pprint import pprint
from pymongo import MongoClient
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from tools import mecab
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm

os.environ['MALLET_HOME'] = "C:\\Users\\KMLAB-02\\mallet-2.0.8"



"""
複数商品に共通する属性を抽出する
＊＊属性抽出における閾値を個別に自動設定する＊＊
閾値はトランザクション内の商品数
"""
def extract_aspects2(DIRECTORY:str):
    W = 10      # トピックから取り出す単語の数
    C = 6      # クラスタ数
    T_iphone = 10
    T_airpods = 10
    T_awatch = 10
    T_gopro = 10
    T_firehd = 10
    NB_param = 0
    NB_rate = 0.0
    DATA = [
        # ファイルパス, 商品名, トピック数, NO_BELOW, NO_ABOVE
        (DIRECTORY+"texts_iphone11.csv", "iPhone 11", T_iphone, NB_rate, 1.0),
        (DIRECTORY+"texts_AirpodsPro.csv", "AirpodsPro", T_airpods, NB_rate, 1.0),
        (DIRECTORY+"texts_AppleWatch5.csv", "AppleWatch 5", T_awatch, NB_rate, 1.0),
        (DIRECTORY+"texts_GoproHero8.csv", "GoPro Hero8", T_gopro, NB_rate, 1.0),
        (DIRECTORY+"texts_FireHD10.csv", "Fire HD 10", T_firehd, NB_rate, 1.0) #1+NB_param
    ]
    #FPM出力
    FPM_PATH = DIRECTORY+"FPM_kmeanssss.txt"

    # print("処理済のテキスト集合を呼び出し")
    texts = []
    texts_num = []
    for item in DATA:
        path = item[0]
        a_product_texts = []
        with open(path, 'r', encoding="utf-8") as f:
            for row in csv .reader(f): a_product_texts.append(row)
        texts_num.append(len(a_product_texts))      # 文書数
        texts.append(a_product_texts)
    del a_product_texts, path, f, row, item

    """
    商品ごとにトピック抽出⇒単語分布作成
    """
    alldomain_topicid_topic_dict = {}
    topic_id = 0
    for text, data, num in zip(texts,DATA, texts_num):
        PNAME = data[1]
        T = data[2]
        rate = num / sum(texts_num)     # 商品１つのテキスト数/全商品のテキスト数

        print("{0}のテキストコーパス作成～LDA学習".format(PNAME))
        # gensim辞書
        dictionary = corpora.Dictionary(text)
        # フィルタリング
        dictionary.filter_extremes(no_below=1+data[3]*len(text),
                                   no_above=data[4])
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
        # print(lda_bayes)
        # for t in range(T): print(lda_bayes.print_topic(t, W))

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

                distribution.append((word, prob*rate))    # 出現確率*商品のテキスト数比率
                # distribution.append((word, prob))  # 出現確率*商品のテキスト数比率


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
    # クラスタ数の検討
    c_range = 0
    for data in DATA: c_range += data[2]    # トピック数の合計⭢最大クラスタ数

    distortions = []
    for i in range(2, c_range-1):  # 1~10クラスタまで一気に計算
        km = KMeans(n_clusters=i, random_state=1)
        km.fit(topic_array)  # クラスタリングの計算を実行
        # SSE
        distortions.append(km.inertia_)  # km.fitするとkm.inertia_が得られる inertia=重心との二乗誤差

        # シルエット値（-1～1）の平均
        cluster_labels = km.fit_predict(topic_array)
        silhouette_avg = silhouette_score(topic_array, cluster_labels)
        print('For n_clusters =', i,
              'The average silhouette_score is :', silhouette_avg)

    plt.plot(range(2, c_range-1), distortions, marker='o')
    plt.xlim([0, c_range])
    plt.xticks(list(range(2, c_range-1, 2)))
    plt.xlabel('Number of clusters')

    # plt.ylim([0.00, 0.02])
    plt.ylabel('Distortion')
    plt.grid(True)
    plt.show()


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
    result_FPM = "[K-meansクラスタリング] クラスタ数: {0}".format(str(C))
    result_FPM += "="*10 + "頻出パターン抽出(FP-Growth)" + "="*10 + "\n"
    """
    transaction
    ex)
    クラスタAにはトピック1, 3, 5が属するとき、
    クラスタAのトランザクションは
    transactionA = [[トピック１の単語id1, １の単語id2, １の単語id3,...],
                    [トピック２の単語id1, ２の単語id2, ２の単語id3,...],
                    [トピック３の単語id1, ３の単語id2, ３の単語id3,...]]    
    """
    for cluster, topic_ids in cluster_topicids_dict.items():    # クラスタごと
        transactions = []           # クラスタの単語idリストのリスト　
        pre_topic_product = None    # トピックの商品情報
        temp = []
        for topic_id in topic_ids:                              # トピックごと
            w_ids = []
            for item in alldomain_topicid_topic_dict[topic_id]: # 単語ごと
                word = item[0]
                for key, value in alldomain_worddict.items():   # 単語辞書から単語を取り出す
                    if value == word:
                        w_ids.append(key)
                        break

            # このトピックと商品のチェック
            if pre_topic_product is None:                                   # 前トピック情報がない
                temp = w_ids
                pre_topic_product = topic_product_dict[topic_id]
            elif pre_topic_product == topic_product_dict[topic_id]:     # 前トピックと同じ商品
                temp.extend(w_ids)
                temp = list(set(temp))
            elif pre_topic_product != topic_product_dict[topic_id]:     # 前トピックと別の商品
                transactions.append(temp)
                temp = w_ids
                pre_topic_product = topic_product_dict[topic_id]

        transactions.append(temp)


        print("\n" + "-"*20 + "クラスタ: {0} ".format(cluster) + "-"*20)
        result_FPM += "\n" + "-"*20 + "クラスタ: {0} ".format(cluster) + "-"*20 + "\n"
        pname = ""
        for topic_id in topic_ids:
            product = topic_product_dict[topic_id]
            if pname != product[0]:
                pname = product[0]
                print(pname + ", ", end="")
                result_FPM += pname + ", "

        print("\n")
        result_FPM += "\n\n"

        if len(transactions) == 1:
            print("商品１つのみ")
            result_FPM += "商品１つのみ\n"
            for topic_id in topic_ids:
                print(alldomain_topicid_topic_dict[topic_id])
                result_FPM += str(alldomain_topicid_topic_dict[topic_id]) + "\n"
            continue
        itemsets = fpgrowth.frequent_itemsets(transactions, len(transactions))    # 頻出ワード集合
        if itemsets is None:
            for topic_id in topic_ids:
                print(alldomain_topicid_topic_dict[topic_id])
                result_FPM += str(alldomain_topicid_topic_dict[topic_id]) + "\n"

        for tuple in itemsets:
            if len(tuple[0]) != 1: continue
            # print(tuple, end="")
            print("  ( ", end="")
            result_FPM += "( "
            for id in tuple[0]:
                word = alldomain_worddict[id]
                print(word + ", ", end="")
                result_FPM += word + ", "
            print(")Ｘ ", end="")
            print(tuple[1])
            result_FPM += ")Ｘ {0}\n".format(str(tuple[1]))

    with open(FPM_PATH, 'w', encoding="utf-8") as f:
        f.write(result_FPM)

    print("終了")



"""
複数商品に共通する属性を抽出する(階層的クラスタリング)
＊＊属性抽出におけるFPM閾値を個別に自動設定する＊＊
閾値はトランザクション内の商品数
"""
def extract_aspects_hierarichal(DIRECTORY:str):
    W = 10      # トピックから取り出す単語の数
    T_iphone = 10
    T_airpods = 10
    T_awatch = 10
    T_gopro = 10
    T_firehd = 10
    NB_param = 0
    # 階層クラスタリングのアルゴリズム設定
    method = 'ward'
    metric = 'euclidean'  # ユークリッド距離
    color_threshold = 0.8       # ユークリッド距離の閾値?
    fcluster_params = [5, 'maxclust']   # クラスタリング実行のパラメータ
    # distance-> tはユークリッド距離の閾値, maxclust-> tはクラスタ数　

    DATA = [
        # ファイルパス, 商品名, トピック数, NO_BELOW, NO_ABOVE
        (DIRECTORY+"texts_iphone11.csv", "iPhone 11", T_iphone, 1+NB_param, 1.0),
        (DIRECTORY+"texts_AirpodsPro.csv", "AirpodsPro", T_airpods, 1+NB_param, 1.0),
        (DIRECTORY+"texts_AppleWatch5.csv", "AppleWatch 5", T_awatch, 1+NB_param, 1.0),
        (DIRECTORY+"texts_GoproHero8.csv", "GoPro Hero8", T_gopro, 1+NB_param, 1.0),
        (DIRECTORY+"texts_FireHD10.csv", "Fire HD 10", T_firehd, 1 + NB_param, 1.0)
    ]
    #FPM出力
    FPM_PATH = DIRECTORY+"FPM_hieraric.txt"

    print("処理済のテキスト集合を呼び出し")
    texts = []
    texts_num = []
    for item in DATA:
        path = item[0]
        a_product_texts = []
        with open(path, 'r', encoding="utf-8") as f:
            for row in csv .reader(f): a_product_texts.append(row)
        texts_num.append(len(a_product_texts))      # 文書数
        texts.append(a_product_texts)
    del a_product_texts, path, f, row, item

    """
    商品ごとにトピック抽出⇒単語分布作成
    """
    alldomain_topicid_topic_dict = {}
    topic_product_dict = {}
    topic_id = 0
    product_id = 1
    for text, data, num in zip(texts,DATA, texts_num):
        PNAME = data[1]
        T = data[2]
        NO_BELOW = data[3]
        NO_ABOVE = data[4]
        rate = num / sum(texts_num)

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
        # for t in range(T): print(lda_bayes.print_topic(t, W))

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

                distribution.append((word, prob*rate))    # 出現確率*商品のテキスト数比率

            alldomain_topicid_topic_dict.setdefault(topic_id, distribution)
            topic_product_dict.setdefault(topic_id, (PNAME, product_id))
            topic_id += 1
        product_id += 1

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

    print("階層的クラスタリングに渡すDataFrame作成")

    """
    numpy arrayの辞書版topic_arry_dict topic_arrayを作成
    """
    # header は単語idのリスト[単語id1, 単語id2, 単語id3, ...,]
    header = []
    for id in alldomain_worddict.keys(): header.append(id)

    topic_array_dict = {}
    topic_id = 0
    for t_id, topics in alldomain_topicid_topic_dict.items():
        row = [0] * len(header)
        for topic in topics:    # 各単語を照合
            word = topic[0]
            word_id = [id for id, dword in alldomain_worddict.items() if word == dword].pop(0)
            prob = topic[1]
            word_index = header.index(word_id)
            row[word_index] = prob
        topic_array_dict.setdefault(topic_id, row)
        topic_id += 1

    # topic_array_dictからndarrayを作成
    topic_array = np.empty((0, len(header)), float)
    for row in topic_array_dict.values():
        topic_array = np.append(topic_array, np.array([row]), axis=0)


    del id, topics, topic, word, prob, word_id, word_index, row, text, header, topic_id


    print("\n=============階層型クラスタリング============")
    dataflame_index = []
    for topic_id, product in topic_product_dict.items():
        dataflame_index.append(str(product[0]) + "_" + str(topic_id))

    topic_dataflame = pd.DataFrame(topic_array, index=dataflame_index)
    # plotting.scatter_matrix(topic_dataflame, figsize=(9, 9))              # 散布図
    # plt.show()

    # tools.clustering_cophenet(topic_dataflame)
    linkage_result = linkage(topic_dataflame, method=method, metric=metric)

    # 樹形図　クラスタリングの結果を図で
    plt.figure(num=None, figsize=(17, 17), dpi=200, facecolor='w', edgecolor='k')
    dendrogram(linkage_result, labels=topic_dataflame.index, color_threshold=color_threshold*max(linkage_result[:,2]))   # color_threshold ユークリッド距離の閾値
    plt.xlabel("Product Name_Topic Number")
    plt.ylabel(metric + " distance")
    plt.title(method)
    plt.show()

    fcluster_result = fcluster(Z=linkage_result, t=fcluster_params[0], criterion=fcluster_params[1])    # クラスタリングしたラベルを取得
    cluster_topicids_dict ={}
    for topic, c in enumerate(fcluster_result):
        if c in cluster_topicids_dict.keys():
            cluster_topicids_dict[c].append(topic)
            continue
        else:
            cluster_topicids_dict.setdefault(c, [topic])

    print("\n" + "="*10 + "頻出パターン抽出(FP-Growth)" + "="*10)
    result_FPM = "[階層クラスタリング] 樹形図閾値:{0}, fclusterパラメータ:{1}\n".format(str(color_threshold), str(fcluster_params))
    result_FPM += "=" * 10 + "頻出パターン抽出(FP-Growth)" + "=" * 10 + "\n"

    for cluster, topic_ids in cluster_topicids_dict.items():  # クラスタごと
        transactions = []  # クラスタの単語idリストのリスト　
        pre_topic_product = None  # 単一トピックに対応する商品
        temp = []

        #同じ商品のトピックをまとめる
        for topic_id in topic_ids:  # トピックごと
            w_ids = []
            for item in alldomain_topicid_topic_dict[topic_id]:  # 単語ごと
                word = item[0]
                for key, value in alldomain_worddict.items():  # 辞書を引いて単語idを取り出す
                    if value == word:
                        w_ids.append(key)
                        break

            # このトピックと商品のチェック
            if pre_topic_product is None:  # 前トピック情報がない
                temp = w_ids
                pre_topic_product = topic_product_dict[topic_id]
            elif pre_topic_product == topic_product_dict[topic_id]:  # 前トピックと同じ商品
                temp.extend(w_ids)
                temp = list(set(temp))
            elif pre_topic_product != topic_product_dict[topic_id]:  # 前トピックと別の商品
                transactions.append(temp)
                temp = w_ids
                pre_topic_product = topic_product_dict[topic_id]

        transactions.append(temp)

        print("\n" + "-" * 20 + "クラスタ: {0} ".format(cluster) + "-" * 20)
        result_FPM += "\n" + "-" * 20 + "クラスタ: {0} ".format(cluster) + "-" * 20 + "\n"
        pname = ""
        for topic_id in topic_ids:
            product = topic_product_dict[topic_id]
            if pname != product[0]:
                pname = product[0]
                print(pname + ", ", end="")     #商品名print
                result_FPM += pname + ", "

        print("\n")
        result_FPM += "\n\n"

        """
        １商品のみのクラスタ
        """
        if len(transactions) == 1:
            print("商品１つのみ")
            result_FPM += "商品１つのみ\n"
            for topic_id in topic_ids:
                print(alldomain_topicid_topic_dict[topic_id])
                result_FPM += str(alldomain_topicid_topic_dict[topic_id]) + "\n"
            continue
        """
        複数商品のクラスタ
        """
        itemsets = fpgrowth.frequent_itemsets(transactions, len(transactions))  # 頻出ワード集合
        if itemsets is None:
            for topic_id in topic_ids:
                print(alldomain_topicid_topic_dict[topic_id])
                result_FPM += str(alldomain_topicid_topic_dict[topic_id]) + "\n"

        for tuple in itemsets:
            if len(tuple[0]) != 1: continue     # 単語が複数のセットをスキップ
            print("  ( ", end="")
            result_FPM += "( "
            for id in tuple[0]:
                word = alldomain_worddict[id]
                print(word + ", ", end="")
                result_FPM += word + ", "
            print(")Ｘ ", end="")
            print(tuple[1])
            result_FPM += ")Ｘ {0}\n".format(str(tuple[1]))

    with open(FPM_PATH, 'w', encoding="utf-8") as f:
        f.write(result_FPM)

    print("終了")





"""
K-meansをせず、商品の組み合わせで複数商品に共通する属性を抽出

＊＊属性抽出における閾値を個別に自動設定する＊＊
閾値はトランザクション内の商品数
"""
def extract_aspects_comparison(DIRECTORY:str):
    W = 10  # トピックから取り出す単語の数
    T_iphone = 10
    T_airpods = 10
    T_awatch = 10
    T_gopro = 10
    T_firehd = 10
    NB_param = 0
    DATA = [
        # ファイルパス, 商品名, トピック数, NO_BELOW, NO_ABOVE
        (DIRECTORY+"texts_iphone11.csv", "iPhone 11", T_iphone, 1 + NB_param, 1.0),
        (DIRECTORY+"texts_AirpodsPro.csv", "AirpodsPro", T_airpods, 1 + NB_param, 1.0),
        (DIRECTORY+"texts_AppleWatch5.csv", "AppleWatch 5", T_awatch, 1+NB_param, 1.0),
        (DIRECTORY+"texts_GoproHero8.csv", "GoPro Hero8", T_gopro, 1 + NB_param, 1.0),
        (DIRECTORY+"texts_FireHD10.csv", "Fire HD 10", T_firehd, 1 + NB_param, 1.0)
    ]
    FPM_PATH = DIRECTORY+"FPM_souatari.txt"

    C = len(DATA)   #組み合わせで選ぶ数


    print("処理済のテキスト集合を呼び出し")
    texts = []
    for item in DATA:
        path = item[0]
        a_product_texts = []
        with open(path, 'r', encoding="utf-8") as f:
            for row in csv.reader(f): a_product_texts.append(row)
        texts.append(a_product_texts)
    del a_product_texts, path, f, row, item

    # 商品ごとにトピック抽出⇒単語分布作成
    alldomain_topicid_topic_dict = {}
    topic_id = 0
    for text, data in zip(texts, DATA):
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
        for t in range(T): print(lda_bayes.print_topic(t, W))

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
                word = id_word_dict[id_prob[0]]  # 抽出された単語
                prob = id_prob[1]  # 出現確率

                distribution.append((word, prob))

            alldomain_topicid_topic_dict.setdefault(topic_id, distribution)
            topic_id += 1

    del t, id_word_dict, id_prob, distribution, word, prob, lda_bayes, dictionary
    print("=" * 14 + "トピック抽出終了" + "=" * 43)

    """
    全商品の全トピックにある単語の辞書を作る
    重複無し
    """
    alldomain_worddict = {}
    id = 1
    for topics in alldomain_topicid_topic_dict.values():
        for topic in topics:
            word = topic[0]
            if word in alldomain_worddict.values(): continue  # 単語の重複チェック
            alldomain_worddict.setdefault(id, word)
            id += 1
    del id, word, topics, topic

    print("K-Meansに渡すtopic_array作成")
    # トピックと商品名の対応
    topic_product_dict = {}
    l = []

    topic_id = 0
    product_id = 1
    for item in DATA:
        pname = item[1]  # 商品名
        l.append(pname)
        T = item[2]  # 指定したトピック数
        for t in range(T):
            # topic_pname_dict[topic_id] = pname
            topic_product_dict[topic_id] = (pname, product_id)
            topic_id += 1
        product_id += 1



    """
    topic_arry_dictの辞書版 topic_wid_dictを作成
    """
    header = []  # ヘッダ header ⇒[単語id1, 単語id2, 単語id3, ...,]
    # header.append("product_id")
    for id in alldomain_worddict.keys():
        header.append(id)


    topic_wid_dict = {}
    for t_id, topics in alldomain_topicid_topic_dict.items():
        row2 = []
        for topic in topics:  # 各単語を照合
            word = topic[0]
            word_id = [id for id, dword in alldomain_worddict.items() if word == dword].pop(0)
            index = header.index(word_id)
            row2.append(word_id)
        topic_wid_dict.setdefault(t_id, row2)


    # topic_array_dictをK-meansに渡すndarrayに変換
    topic_array = np.empty((0, len(header)), float)

    print("\n" + "=" * 10 + "[トピック総当たり手法]頻出パターン抽出(FP-Growth)" + "=" * 10)
    result_FPM = "トピック総当たり手法\n" + "=" * 10 + "頻出パターン抽出(FP-Growth)" + "=" * 10 + "\n"
    for n in range(2, C+1):
        for combi in itertools.combinations(l, n):
            print("\n{0}の共通属性".format(str(combi)))
            result_FPM += "\n{0}の共通属性\n".format(str(combi))
            transactions = []
            for product in combi:
                product_w_ids = []
                product_topics = []
                for key, item in topic_product_dict.items():
                    if product == item[0]: product_topics.append(key)

                for topic in product_topics:
                    product_w_ids.extend(topic_wid_dict[topic])
                product_w_ids = list(set(product_w_ids))
                transactions.append(product_w_ids)

            itemsets = fpgrowth.frequent_itemsets(transactions, len(transactions))  # 頻出ワード集合
            for tuple in itemsets:
                if len(tuple[0]) != 1: continue  # 単語が複数のセットをスキップ
                print("  ( ", end="")
                result_FPM += "( "
                for id in tuple[0]:
                    word = alldomain_worddict[id]
                    print(word + ", ", end="")
                    result_FPM += word + ", "
                print(") Ｘ ", end="")
                print(tuple[1])
                result_FPM += ")Ｘ {0}\n".format(str(tuple[1]))

    with open(FPM_PATH, 'w', encoding="utf-8") as f:
        f.write(result_FPM)


    print("終了")




"""
ツイートから、属性語を評価する評価語(極性語)のペアを取り出す
"""
def ext_aspw_pair(pnum:int, zokusei:str):
    """
    入力先設定
    """
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    if pnum == 1:
        col = db.unique_tweets  # iphone11          スマートフォン
        pname = "iPhone 11"
    elif pnum == 2:
        col = db.AirpodsPro     # AirpodsPro        ワイヤレスイヤホン
        pname = "Airpods Pro"
    elif pnum == 3:
        col = db.AppleWatch5    # Apple Watch5      スマートウォッチ
        pname = "Apple Watch 5"
    elif pnum == 4:
        col = db.GoproHero8     # GoPro HERO 8      アクションカメラ
        pname = "GoPro HERO 8"
    elif pnum == 5:
        col = db.FireHD10       # Amazon Fire HD 10 タブレット
        pname = "Fire HD 10"

    print("\n{0} の属性\"{1}\"と評価語のペア抽出".format(pname, zokusei))

    texts = []
    for doc in col.find(): texts.append(doc["text"])

    polword_dict = {}
    # 評価語辞書読み込み
    db = client.word_dict

    # 乾辞書   活用形網羅するために、スペースを含む単語がある
    # col = db.pn_wago
    # docs = col.find(projection={'_id': 0})
    # for doc in docs:
    #     word = doc["word"].replace(" ", "+") # スペースを含む単語は、スペースを+に置換して登録
    #     polword_dict.setdefault(word, doc["param"])
    # if '' in polword_dict.keys(): polword_dict.pop('')
    # # 高村辞書　用言＋助動詞（"ない""ぬ"）
    # col = db.pn_takamura
    # docs = col.find(projection={'_id': 0})
    # for doc in docs: polword_dict.setdefault(doc["word"], doc["param"])

    # 統合辞書(重複無し)
    col = db.pn_combi
    docs = col.find(projection={'_id': 0})
    for doc in docs:
        word = doc["word"].replace(" ", "+")  # スペースを含む単語は、スペースを+に置換して登録
        polword_dict.setdefault(word, doc["param"])
    if '' in polword_dict.keys(): polword_dict.pop('')

    del docs


    """
    名詞に評価語を割り当てる処理
    - 1文ずつ読み込む⇒評価語辞書で探索
    
    辞書result　属性に対応する評価語を集めた
    Posi_count 全テキストから探したポジティブ評価回数
    """
    result = {}
    Posi_count = 0
    Nega_count = 0
    for ttext in texts:
        if zokusei not in ttext: continue

        sentences = re.sub(r'!|！|\?|？|\n', '。', ttext).split("。")     #1文ずつ区切る

        for text in sentences:
            if zokusei not in text: continue

            # テキスト分かち書き
            wakati_text = mecab(text, "-Owakati")
            for word in polword_dict.keys():          # スペース含む評価語への対応
                word = word.replace("+", " ")
                if word in wakati_text:
                    wakati_text = wakati_text.replace(word, word.replace(" ", "+"))

            # 分かち書きテキストを辞書化
            wakati_dict = {}
            position = 1
            for keitaiso in wakati_text.split(" "):
                wakati_dict.setdefault(keitaiso, position)
                position += 1
            del position

            # 属性語の位置を特定
            aspect = ()  # 語, 位置
            for word, position in wakati_dict.items():
                if zokusei == word:
                    aspect = (zokusei, position)
            if not aspect: continue

            # 評価語を探索
            pol_posi_dist = []    # 評価語、位置、 属性語との距離
            for word, position in wakati_dict.items():
                for p_word in polword_dict.keys():
                    # word2 = mecab(word, "-Owakati").replace("\n", "")         # 乾辞書のスペース有り単語への対応
                    if p_word != word: continue
                    pol_posi_dist.append([p_word, position, None])
                    break
            if not pol_posi_dist: continue
            for item in pol_posi_dist:
                    item[2] = abs(aspect[1] - item[1])

            # 属性語に最短距離の評価語を割り当てる
            pol_posi_dist = sorted(pol_posi_dist, key=lambda x: x[2])  # 距離の昇順ソート
            closest = pol_posi_dist[0]
            pol_word = closest[0]
            if polword_dict[pol_word] > 0:
                pair = "{0} - {1} {2}".format(zokusei, pol_word, "P")     # ポジティブ評価を割り当てる
                Posi_count += 1
            else:
                pair = "{0} - {1} {2}".format(zokusei, pol_word, "N")     # ネガティブ評価を割り当てる
                Nega_count += 1

            if pair not in result.keys():
                result.setdefault(pair, 1)    # 属性語-評価語のペアと、ペアの出現回数
            else:
                result[pair] += 1

    result = sorted(result.items(), key=lambda x:x[1], reverse=True)
    pprint(result)
    print("{0} ⇒ポジティブ: {1}回, ネガティブ: {2}回".format(zokusei, str(Posi_count), str(Nega_count)))


    print("終了")


def main():
    DIRECTORY_PATH = "Data/Topicmodel/1217/"

    extract_aspects2(DIRECTORY_PATH)
    # extract_aspects_hierarichal(DIRECTORY_PATH)
    # extract_aspects_comparison(DIRECTORY_PATH)
    # for i in range(3, 6): ext_aspw_pair(i, "値段")   # 引数　(対象商品の番号, 評価したい属性)


if __name__ == '__main__':
    main()
    sys.exit()
