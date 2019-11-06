# - coding: utf-8 -

import csv
import emoji
import gensim
import numpy as np
import os
import re
import sys
import tools
from sklearn.cluster import KMeans
from pprint import pprint
from pymongo import MongoClient
from gensim import corpora, models

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
    with open("Data/Twitter/stopwords.txt", 'r', encoding="utf-8") as f:
        rows = f.read()
        for word in rows.split('\n'):
            if word: stopwords.append(word)

    print("テキスト読み込み")
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    col = db.unique_tweets
    docs = col.find(projection={'_id': 0})
    texts = []
    for doc in docs:
        if len(texts) == N: break  # 文書数
        if doc["text"] is not None:
            texts.append(doc["text"])

    print("テキストを単語のリストで表現")
    for text in texts:
        words = []
        frequency = {}
        # 生テキストを前処理
        text = tools.pre_process(text)

        # 単語分割と品詞特定
        nodes = tools.mecab_tolist(text)
        for node in nodes:
            word = node[0]
            # ストップワードをスキップ
            if word in stopwords: continue
            if "iphone" in word: continue
            if len(word) == 1 and re.search(r'[あ-ん]', word): continue
            # 名詞を抜き出す
            if node[1] == "名詞" and node[2] != "接尾" and node[2] != "数" and node[2] != "サ変接続":
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1

        for word, freq in frequency.items():
            if freq >= 1: words.append(word)
        if words: processed_texts.append(words)

    with open("Data/Topicmodel/texts{0}.csv".format(N), 'w', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(processed_texts)
        f.close()

    return processed_texts


"""
トピックモデルの実装
"""
def lda_test():
    print("テキスト集合(単語リスト)を呼び出し")
    texts = []
    with open("Data/Topicmodel/texts100.csv", 'r', encoding="utf-8") as f:
        for row in csv.reader(f): texts.append(row)

    print("テキストコーパス作成～LDA学習")
    N = 15  # トピック数

    # 辞書
    dictionary = corpora.Dictionary(texts)
    w_id_dict = dictionary.token2id
    # フィルタリング
    dictionary.filter_extremes(no_below=1)
    # コーパス生成
    corpus = [dictionary.doc2bow(text) for text in texts]
    # トピックを持つLDAモデルを作成
    lda_bayes = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                num_topics=N,
                                                id2word=dictionary)

    print("トピック(変分ベイズ)")
    pprint(lda_bayes.show_topics(num_topics=N))

    topics = np.empty((0, 2), int)
    print("トピックをクラスタリング")
    for n in range(N):
        for t in lda_bayes.get_topic_terms(n, 10):
            id = t[0]
            topics = np.append(topics, np.array([[n, id]]), axis=0)    # [トピック番号, 単語id]

    kmeans_model = KMeans(n_clusters=5).fit(topics)
    labels = kmeans_model.labels_
    for label, topic in zip(labels, topics):
        id = topic[1]
        word = [key for key, value in w_id_dict.items() if value == id]
        print(label, topic, word)



    # """
    # LDA+Gibbs sampling
    # """
    # corpora.malletcorpus.MalletCorpus.serialize("./corpus.mallet", corpus)
    # mallet_corpus = corpora.malletcorpus.MalletCorpus("./corpus.mallet")
    # mallet_path = "C:/Users/KMLAB-02/mallet-2.0.8/bin/mallet"
    # lda_gibbs = models.wrappers.LdaMallet(mallet_path, mallet_corpus, num_topics=5, id2word=dictionary)
    # print("トピック(ギブスサンプリング)")
    # pprint(lda_gibbs.show_topics())

def main():
    lda_test()
    # trans_texts(1000)
    print("終了")


if __name__ == '__main__':
    main()
    sys.exit()
