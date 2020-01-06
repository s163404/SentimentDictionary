# - coding: utf-8 -

import csv
import emoji
import gensim
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import tools
from tqdm import tqdm
from gensim import corpora
from gensim.models.ldamodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import HdpModel
from orangecontrib.associate.fpgrowth import *
from pandas.tools import plotting
from pprint import pprint
from pymongo import MongoClient
from sklearn.cluster import KMeans
from statistics import mean

os.environ['MALLET_HOME'] = "C:\\Users\\KMLAB-02\\mallet-2.0.8"

"""
テキストを単語のリストに変換
入力: テキスト集合
出力: 単語リストのリスト
- テキスト前処理
- 条件に合う名詞を取り出す
"""
def trans_texts(DIRECTORY_PATH:str, pnum:int, ctrlHenkan:bool):
    STOPWORD_PATH = DIRECTORY_PATH+"stopwords_1217.txt"
    DOUGIGO_PATH = DIRECTORY_PATH+"dougigo_1217.csv"
    # ストップワード読み込み
    stopwords = []
    with open(STOPWORD_PATH, 'r', encoding="utf-8") as f:
        for word in f.read().split('\n'):
            if word: stopwords.append(word)
    # 同義語読み込み
    dougigo = {}
    with open(DOUGIGO_PATH, 'r', encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row: continue
            if len(row) != 2: continue
            dougigo.setdefault(row[0], row[1])

    print("テキスト読み込み")
    client = MongoClient('localhost', 27017)
    db = client.tweet_db

    # 入出力先設定
    if pnum == 1:
        col = db.unique_tweets                                  # 入力元
        csv_path = DIRECTORY_PATH+"texts_iphone11.csv"    # 出力先
    elif pnum == 2:
        col = db.AirpodsPro
        csv_path = DIRECTORY_PATH+"texts_AirpodsPro.csv"
    elif pnum == 3:
        col = db.AppleWatch5
        csv_path = DIRECTORY_PATH+"texts_Applewatch5.csv"
    elif pnum == 4:
        col = db.GoproHero8
        csv_path = DIRECTORY_PATH+"texts_Goprohero8.csv"
    elif pnum == 5:
        col = db.FireHD10
        csv_path = DIRECTORY_PATH+"texts_Firehd10.csv"

    # unique_tweets-iphone11
    # AirpodsPro-AirpodsPro
    # AppleWatch5-AppleWatch5
    # GoproHero8-GoproHero8
    # FireHD10-FireHD10

    raw_texts = []
    for doc in col.find(projection={'_id': 0}):
        if doc["text"] is not None: raw_texts.append(doc["text"])


    print("テキストを単語群で表現")
    words_of_texts = []
    for text in raw_texts:
        words = []
        # URL付きツイートはスキップ
        if re.search(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', text): continue
        # テキスト前処理
        if ctrlHenkan == True:
            """
            True
            分かち書きしてから同義語変換
            """
            text = tools.pre_process(text, DOUGIGO_PATH, False)   # テキスト前処理で同義語変換しない

            wakati_text = tools.mecab(text, "-Owakati").split(" ")
            for keitaiso in wakati_text:
                for origin, henkan in dougigo.items():
                    if keitaiso == origin: keitaiso = henkan    # 変換できる語は置換
            text = ''.join(wakati_text)

        else:
            """
            False
            分かち書きせずに同義語変換
            pre_preprocessメソッド内で変換
            """
            text = tools.pre_process(text, DOUGIGO_PATH, True)

        if not text: continue


        # 形態素単位前処理
        nodes = tools.mecab_tolist(text)
        for node in nodes:
            word = node[0]
            flag = False
            # ストップワードをスキップ
            for stopword in stopwords:
                if word == stopword:
                    flag = True
                    break
            if flag is True: continue

            # 商品名をスキップ
            if "iphone" in word: continue
            if "airpods" in word: continue
            if "applewatch" in word:continue
            if "apple watch" in word: continue
            if "gopro" in word:continue
            if "fire" in word: continue

            if len(word) == 1 and re.search(r'[\u30A1-\u30F4]', word): continue     # 1文字カタカナ
            if len(word) == 1 and re.search(r'[あ-ん]', word): continue              # 1文字ひらがな
            if len(word) == 1 and re.search(r'[a-z]', word): continue               # 1文字アルファベット
            # 名詞を抜き出す
            if node[1] == "名詞" and node[2] == "一般":
                words.append(word)  # 名詞-一般

        if words: words_of_texts.append(words)

    # CSV化
    print(csv_path + "に出力")
    with open(csv_path, 'w', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(words_of_texts)
        f.close()


"""
トピックモデルの実装
"""
def lda():
    T = 10  # トピック数
    W = 10 # トピックから取り出す単語の数
    C = T   # クラスタ数
    NO_BELOW = 1
    NO_ABOVE = 1.0  # 10割以上の文書に登場した語除外
    INPUT_PATH = "Data/Topicmodel/1217/texts_FireHD10.csv"
    FPM_PATH = "Data/Topicmodel/1217/FPM_FireHD10.txt"

    print("\nINPUTPATH: " + INPUT_PATH)
    print("T= " + str(T))
    print("W= " + str(W))
    print("C= " + str(C))
    print("NO_BELOW= " + str(NO_BELOW))
    print("NO_ABOVE= " + str(NO_ABOVE))

    print("処理済テキスト(単語集合)を呼び出し")
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
    lda_bayes = LdaModel(corpus=corpus,
                        num_topics=T,
                        id2word=dictionary,
                        random_state=1)

    for t in range(T): print(lda_bayes.print_topic(t, W))
    # pprint(lda_bayes.show_topics())

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

    #ヘッダ行を作る header ⇒["単語id", "単語id", ...]
    header = []
    for topic in topic_wid_prob_list:
        for tuple in topic:
            if tuple[1] not in header: header.append(tuple[1])  # 未登録の単語idを登録

    topic_array = np.empty((0, len(header)), float)  # K-meansに渡すnumpyparray
    for topic in topic_wid_prob_list:
        insert_list = [0] * len(header)
        for tuple in topic:
            index = header.index(tuple[1])
            if index: insert_list[index] = tuple[2]
        topic_array = np.append(topic_array, np.array([insert_list]), axis=0)

    # np.set_printoptions(suppress=True, precision=5, linewidth=100, threshold=np.inf)  # ndarrayのprint表記
    # print(topic_array)

    print("\n=============トピッククラスタリング============")
    # # クラスタ数の検討用
    # distortions = []
    # for i in range(1, T):  # 1~10クラスタまで一気に計算
    #     km = KMeans(n_clusters=i, random_state=1)
    #     km.fit(topic_array)  # クラスタリングの計算を実行
    #     distortions.append(km.inertia_)  # km.fitするとkm.inertia_が得られる inertia=重心との二乗誤差
    # plt.plot(range(1, T), distortions, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Distortion')
    # plt.show()

    kmeans_model = KMeans(n_clusters=C, random_state=1).fit(topic_array) # K-Meansクラスタリング
    labels = kmeans_model.labels_

    clus_topic = {}     # {cluster: [topicnums] } keyはクラスタラベル、valueはトピック番号リスト
    for label, topicnum in zip(labels, range(T)):
        if label not in clus_topic: clus_topic[label] = []
        clus_topic[label].append(topicnum)

    for cluster, topicnums in clus_topic.items():
        print("クラスタ: {0}".format(str(cluster)))
        for t in topicnums:
            print(t, lda_bayes.print_topic(t, W))
            # print(lda_bayes.get_topic_terms(t, W))
        print("-" * 150)


    result_FPM = ""
    print("\n=============頻出する属性語抽出(FP-Growth)=============")
    # 各クラスタに属する各トピックの単語のidを頻出パターン抽出

    """
    全トピックにFPMをかける
    最適なとトピック数の検討用
    ココカラ
    """
    transactions2 = []
    for t in range(T):
        w_ids = []
        for tuple in lda_bayes.get_topic_terms(t, W):
            w_ids.append(int(tuple[0]))
        transactions2.append(w_ids)

    itemsets = frequent_itemsets(transactions2, T)  # 頻出アイテム集合
    print("len(transactions): {0}".format(len(transactions2)))
    result_FPM += "len(transactions): {0}\n".format(len(transactions2))
    for tuple in itemsets:
        # print(tuple, end="")
        print("( ", end="")
        result_FPM += "( "
        for id in tuple[0]:
            word = id_w_dict[id]
            print(word + ", ", end="")
            result_FPM += word + ", "
        print(")Ｘ ", end="")
        result_FPM += ")Ｘ "
        print(tuple[1])
        result_FPM += "{0}\n".format(str(tuple[1]))
    """
    ココマデ
    """
    # for cluster, tnum_list in clus_topic.items():   # clus_topic = {クラスタ番号: トピック番号リスト }
    #     print("\nクラスタ: {0}".format(str(cluster)))
    #     result_FPM += "\nクラスタ: {0}\n".format(str(cluster))
    #     transactions = []
    #     topic_count = 0
    #     for topic in tnum_list:
    #         w_ids = []
    #         for tuple in lda_bayes.get_topic_terms(topic, W):
    #             w_ids.append(int(tuple[0]))
    #         transactions.append(w_ids)
    #         topic_count += 1
    #
    #     if len(transactions) == 1:
    #         print("トランザクションの要素数1")
    #         result_FPM += "トランザクションの要素数1\n"
    #     else:
    #         itemsets = frequent_itemsets(transactions, 2)  # 頻出アイテム集合
    #         print("len(transactions): {0}".format(len(transactions)))
    #         result_FPM += "len(transactions): {0}\n".format(len(transactions))
    #         for tuple in itemsets:
    #             # print(tuple, end="")
    #             print("( ", end="")
    #             result_FPM += "( "
    #             for id in tuple[0]:
    #                 word = id_w_dict[id]
    #                 print(word + ", ", end="")
    #                 result_FPM += word + ", "
    #             print(")Ｘ ", end="")
    #             result_FPM += ")Ｘ "
    #             print(tuple[1])
    #             result_FPM += "{0}\n".format(str(tuple[1]))
    #
    # with open(FPM_PATH, 'w', encoding="utf-8") as f:
    #     f.write(result_FPM)


    print("終了")

"""
トピック数とαパラメータの検討
- alpha: LDAのパラメータα
- mean: 

https://fits.hatenablog.com/entry/2018/03/13/214609
"""
def exam_lda_topic_alpha():
    NO_BELOW = 1
    NO_ABOVE = 1.0  # 10割以上の文書に登場した語除外
    INPUT_PATH = "Data/Topicmodel/1217/texts_Firehd10.csv"
    print(INPUT_PATH + "について")

    print("処理済テキスト(単語集合)を呼び出し")
    texts = []
    with open(INPUT_PATH, 'r', encoding="utf-8") as f:
        for row in csv.reader(f): texts.append(row)

    print("テキストコーパス作成～LDA学習")

    # gensim辞書
    dictionary = corpora.Dictionary(texts)
    # フィルタリング
    dictionary.filter_extremes(no_below=NO_BELOW,
                               no_above=NO_ABOVE)

    # コーパス生成
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("\n==============トピック抽出==============")
    # トピックを持つLDAモデルを作成
    # lda = LdaModel(corpus=corpus,
    #                num_topics=T,
    #                id2word=dictionary,
    #                random_state=1)

    # for t in range(T): print(lda.print_topic(t, W))
    # pprint(lda_bayes.show_topics())

    for t in range(2, 30, 1):
        for n in range(1, 10, 4):
            a = n / 100

            lda = LdaModel(corpus=corpus,
                           id2word=dictionary,
                           num_topics=t,
                           alpha=a,
                           random_state=1)

            # 文書に割り当てられたトピック数をの平均算出
            r = mean([len(lda[c]) for c in corpus])

            print(f"num_topics = {t}, alpha = {a}, mean = {r}")






def hdp():
    INPUT_PATH = "Data/Topicmodel/texts2_AirpodsPro.csv"

    T = 10  # トピック数
    W = 10  # トピックから取り出す単語の数
    C = T  # クラスタ数
    NO_BELOW = 1
    NO_ABOVE = 1.0  # 10割以上の文書に登場した語除外

    save_csv_file = "Data/Topicmodel/hdp/result_hdp.csv"
    FPM_PATH = "Data/Topicmodel/hdp/result_FPM.txt"

    print("処理済のテキスト集合(単語リスト)を呼び出し")
    texts = []
    with open(INPUT_PATH, 'r', encoding="utf-8") as f:
        for row in csv.reader(f): texts.append(row)

    print("テキストコーパス作成～LDA学習")

    # gensim辞書
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=NO_BELOW,
                               no_above=NO_ABOVE)

    # dictionaryから単語辞書(dict型)生成　
    word_id_dict = dictionary.token2id
    dictionary[0]
    id_w_dict = dictionary.id2token

    # コーパス生成
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("\n==============トピック抽出==============")
    # トピックを持つLDAモデルを作成
    hdp = HdpModel(corpus=corpus,
                   id2word=dictionary,
                   random_state=1
                   )

    alpha = hdp.hdp_to_lda()[0] #LDA対応のαパラメータ
    pprint(hdp.show_topics(num_topics=150))
    hdp.show_topics(num_topics=-1)
    # print(hdp.get_topics())
    # for i in range(150): print(hdp.print_topics(i, 10))
    topic_weights = topic_prob_extractor(hdp)



    # 参照 http://kamonohashiperry.com/archives/373
    # 文書ごとに割り当てられたトピックの確率をCSVで出力
    mixture = [dict(hdp[x]) for x in corpus]
    dataflame = pd.DataFrame(mixture)
    dataflame.to_csv("Data/Topicmodel/hdp/topic_for_corpus.csv")
    # トピックごとの上位10語をCSVで出力
    topicdata = hdp.print_topics(num_topics=100, num_words=10)
    pd.DataFrame(topicdata).to_csv("Data/Topicmodel/hdp/topic_detail.csv")


    # 参照http://seesaakyoto.seesaa.net/article/457025212.html
    # shown_topics = hdp.show_topics(num_topics=-1, formatted=False)
    # topics_nos = [x[0] for x in shown_topics]
    # weights = [sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos]
    # data_frame = pd.DataFrame({'topic_id': topics_nos, 'weight': weights})
    # if save_csv_file is not None:
    #     data_frame.to_csv(save_csv_file)
    # return data_frame




def topic_prob_extractor(gensim_hdp, t=-1, w=25, isSorted=True):
    """
    Input the gensim model to get the rough topics' probabilities
    """
    shown_topics = gensim_hdp.show_topics(num_topics=t, num_words=w, formatted=False)
    topics_nos = [x[0] for x in shown_topics]
    weights = [sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos ]
    if (isSorted):
        return pd.DataFrame({'topic_id': topics_nos, 'weight': weights}).sort_values(by="weight", ascending=False);
    else:
        return pd.DataFrame({'topic_id': topics_nos, 'weight': weights});



"""
コヒーレンスとパープレキシィ
"""
def lda_check():
    NO_BELOW = 1
    NO_ABOVE = 1.0  # 10割以上の文書に登場した語除外
    T = 15 # トピック数レンジ
    INPUT_PATH = "Data/Topicmodel/texts2_iphone11.csv"


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

    for i in range(1, T):
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                              num_topics=i,
                                              id2word=dictionary,
                                              alpha=0.01,
                                              random_state=1)

        cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()

        perwordbound = lda.log_perplexity(corpus)
        perplexity = np.exp2(-perwordbound)

        print(f"num_topics = {i}, coherence = {coherence}, perplexity = {perplexity}")



def demo():
    str = "かきくあああけこ"

    if re.search(r'あああ', str):
        print("有り")

def main():
    DIRECTORY = "Data/Topicmodel/1217_2/"

    for n in range(1, 5):
     trans_texts(DIRECTORY, n, False)      # 同義語変換 True ->分かち書きしてから変換 False ->分かち書きせず変換
    # lda()
    # hdp()
    # exam_lda_topic_alpha()
    # demo()



if __name__ == '__main__':
    main()
    sys.exit()
