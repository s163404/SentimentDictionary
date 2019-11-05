# - coding: utf-8 -

import csv
import emoji
import gensim
import re
import sys
import tools
from pprint import pprint
from pymongo import MongoClient
from gensim import corpora

def lda_test():
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
        if doc["text"] is None: continue
        texts.append(doc["text"])

    print("テキストを単語のリストで表現")
    for text in texts:
        words = []
        frequency = {}

        # 生テキストの前処理
        text = tools.pre_process(text)
        # chasen単語分割
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

    f = open("Data/Topicmodel/texts{0}.csv".format(N), 'w', encoding="utf-8")
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(processed_texts)
    f.close()
    return processed_texts


def lda_test():
    print("テキスト集合(単語リスト)を呼び出し")
    texts = []
    with open("Data/Topicmodel/texts100.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader: texts.append(row)

    print("コーパス作成～LDA学習")
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

    for n in range(N):
        print("トピック：{0}".format(str(n)))
        pprint(lda_bayes.get_topic_terms(n, 10))


def main():
    lda_test()
    # trans_texts(1000)
    print("終了")


if __name__ == '__main__':
    main()
    sys.exit()
