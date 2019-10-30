# - coding: utf-8 -

"""
本機能を補助するツール
"""

import MeCab
import mojimoji
import re
import sys

from pymongo import MongoClient

"""
MeCab形態素解析
-Ochasenの出力フォーマット　\tで区切り　\nで改行
"""
def mecab(text:str, format:str):
    if not str: format = "-Owakati"
    m = MeCab.Tagger(format)
    return m.parse(text)

"""
形態素解析結果をリスト出力する
入力str:テキスト
出力list[tuple]:[(表層形, 品詞細分類1, 品詞細分類2, 品詞細分類3, 原型または"", 位置), (), ...]
"""
def mecab_tolist(text:str):
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse('')
    node = tagger.parseToNode(text)    # 解析結果
    word_class = []
    position = 1
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        if wclass[0] != u'BOS/EOS':
            if wclass[6] == None:
                word_class.append((word, wclass[0], wclass[1], wclass[2], "", position))
            else:
                word_class.append((word, wclass[0], wclass[1], wclass[2], wclass[6], position))
        node = node.next
        if word_class: position += 1

    return word_class

"""
極性辞書をDBにインポートする
value-> word|posi|nega|param
type -> str |bool|bool|float
"""
def import_dict():
    print("極性辞書をDBにインポート")
    # MongoDB呼び出し
    client = MongoClient('localhost', 27017)
    db = client.worddict
    collection = db.log10_over05

    # 極性語を抜き出す
    with open("Data/polarity_2_log10.txt", 'r', encoding="utf-8") as f:
        for row in f.readlines():
            pair = row.replace('\n', '').split('\t')
            word = pair[0]
            param = float(pair[1])
            post = {
                "word": word,
                "param": param
            }
            collection.insert_one(post)

"""
複数のコレクションを統合する
"""
def comb_collections():
    combined_list = []
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    collection = db.collection1
    for doc in collection.find(projection={'_id':0, 'text':1, 'id':1, 'created_at':1}): combined_list.append(doc)

    collection = db.collection2
    for doc in collection.find(projection={'_id':0, 'text':1, 'id':1, 'created_at':1}): combined_list.append(doc)

    collection = db.collection3
    for doc in collection.find(projection={'_id':0, 'text':1, 'id':1, 'created_at':1}): combined_list.append(doc)

    collection = db.combined
    # collection.insert_many(combined_list)

"""
テキストを成形する
-記号とアルファベットを半角化
-URL除去
-(リプの場合)宛先表現を除去
"""
def tweet_former(text:str):
    text = mojimoji.zen_to_han(text, kana=False)
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    text = re.sub(r'^(@.{1,15} (?=)){1,2}', '', text)
    return text

"""
DBのデータ移し替え
"""
def move_documents():
    client = MongoClient('localhost', 27017)
    out_db = client.tweet_db
    out_col = out_db.EVs_over05

    in_db = client.tweet_db
    in_col = in_db.test_collection

    docs = out_col.find(projection={'_id':0}).limit(3000)
    in_col.remove()
    in_col.insert_many(docs)



def test():
    print("1から10までカウント")
    for i in range(10):
        print(i)
    print("ほい")

if  __name__ == '__main__':
    # import_dict()
    tweet_former("ボタンを押して9/26の実況者チームを応援した5名様に10000円分のAmazonギフト券を贈呈！ さらに、応援したチームが優勝した1名様にiPhone 11を贈呈！ 9月26日 20時、実況者による天地対決が開戦！ #荒野行動 #荒野天地対決 #でぃふぇあチーム応援")
    sys.exit