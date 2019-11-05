# - coding: utf-8 -

import emoji
import gensim
import re
import sys
import tools
from pprint import pprint
from pymongo import MongoClient
from gensim import corpora

def lda_test():
    processed_texts = []
    stopwords = []
    with open("Data/Twitter/stopwords.txt", 'r', encoding="utf-8") as f:
        rows = f.read()
        for word in rows.split('\n'):
            if word: stopwords.append(word)


    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    col = db.unique_tweets
    # テキスト読み込み
    texts = []
    docs = col.find(projection={'_id':0})
    for doc in docs:
        if len(texts) is 500: break
        if doc["text"] is None: continue
        texts.append(doc["text"])

    for text in texts:
        words_of_text = []
        # 生テキスト前処理
        text = tools.pre_process(text)
        # chasen単語分割
        nodes = tools.mecab_tolist(text)
        frequency = {}
        for node in nodes:
            word = node[0]
            # ストップワードをスキップ
            if word in stopwords: continue
            # 名詞を抜き出し出現回数をカウント
            if node[1] == "名詞" \
                    and node[2] != "接尾" \
                    and node[2] != "数" \
                    and node[2] != "サ変接続":
                if word not in frequency: frequency[word] = 1
                else: frequency[word] += 1
            
        for word, freq in frequency.items():
            if freq >= 1: words_of_text.append(word)
        if words_of_text: processed_texts.append(words_of_text)
    
    # dictionary
    dictionary = corpora.Dictionary(processed_texts)
    # フィルタリング no_below出現文書数, no_above文書出現率
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    # BoW形式コーパス作成
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    # トピックを持つLDAモデルを作成
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)
    print("トピック")
    pprint(lda.show_topics())
    gensim.models.wrappers.ldamallet



def main():
    lda_test()
    print("終了")

if __name__ == '__main__':
    main()
    sys.exit()