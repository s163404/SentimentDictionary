# - coding: utf-8 -

"""
本機能を補助するツール
"""

import csv
import datetime
import emoji
import MeCab
import mojimoji
import neologdn
import numpy as np
import pandas as pd
import pandas
import pytz
import re
import sys
import time
import topicmodel
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
import matplotlib.pyplot as plt


from pymongo import MongoClient

"""
生テキストの前処理
入力：str
出力：str
"""

def pre_process(text:str, DOUGIGO_PATH:str, isdougigo:bool):
    # リプ宛先除去
    text = re.sub(r'@.{1,15} ', '', text)
    text = re.sub(r'@.{1,15}さんから', '', text)
    # text = re.sub(r'^(@.{1,15} (?=)){1,2}', '', text)
    # 正規化
    # -アルファベット：全→半
    # -カタカナ：半→全
    # -伸ばし棒統一
    # ただし、全角記号は例外
    text = neologdn.normalize(text)
    # アルファベット：小文字化
    text = text.lower()
    # URL除去
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    # 数字の","除去
    text = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text)
    # 絵文字  -----文字自体消せず。品詞が「名詞」→「記号」になるのみ
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
    text = re.sub(r'[\U0001F300-\U0001F5FF]+', '', text)
    # 表現フィルター
    if "モイ!iphoneからキャス配信中" in text:
        text = ""


    """
    同義語変換
    textを分かち書きせず、単語を変換する
    """
    if isdougigo == True:
        dougigo = {}
        with open(DOUGIGO_PATH, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                if len(row) != 2: continue
                dougigo.setdefault(row[0], row[1])

        for k, v in dougigo.items():
            text = text.replace(k, v)

    return text



"""
MeCab形態素解析
-Ochasenの出力フォーマット　\tで区切り　\nで改行
"""
def mecab(text:str, format:str):
    if not str: format = "-Owakati"
    m = MeCab.Tagger(format)
    parsed = m.parse(text)
    return parsed

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
値-> word|posi|nega|param
型 -> str|bool|bool|float
"""
def import_dict():
    print("極性辞書をDBにインポート")
    # MongoDB呼び出し
    client = MongoClient('localhost', 27017)
    db = client.word_dict
    collection = db.pn_takamura

    # # 極性語を抜き出す
    # with open("Data/wago_pn.txt", 'r', encoding="utf-8") as f:
    #     for row in f.readlines():
    #         pair = row.replace('\n', '').split('\t')
    #         word = pair[1]
    #         if "ポジ" in pair[0]: param = 1
    #         else: param = -1
    #         # if abs(param) < 0.5: continue
    #         post = {
    #             "word": word,
    #             "param": param
    #         }
    #         collection.insert_one(post)

    with open("Data/pn_ja.txt", 'r', encoding="utf-8") as f:
        for row in f.readlines():
            items = row.split(':')
            if items[2] == "名詞" or items[2] == "副詞": continue

            word = items[0]
            if float(items[3]) > 0: param = 1
            else: param = -1
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

    collection = db.RingFA      # 統合したいコレクション１
    for doc in collection.find(projection={'_id':0, 'text':1, 'id':1, 'created_at':1}):
        combined_list.append(doc)

    collection = db.AirpodsPro  # 統合したいコレクション２
    for doc in collection.find(projection={'_id':0, 'text':1, 'id':1, 'created_at':1}):
        combined_list.append(doc)

    collection = db.GoProHero8  # 統合したいコレクション３
    for doc in collection.find(projection={'_id':0, 'text':1, 'id':1, 'created_at':1}):
        combined_list.append(doc)

    collection = db.combined1113
    collection.insert_many(combined_list)



"""
DBのデータ移し替え
"""
def move_documents():
    client = MongoClient('localhost', 27017)
    # エクスポート側
    ex_db = client.dbaname
    ex_col = ex_db.colname
    # インポート側
    in_db = client.tweet_db
    in_col = in_db.test_collection

    docs = ex_col.find(projection={'_id':0})

    in_col.insert_many(docs)


"""
CSVデータをMongoDBにインポート
"""
def importcsvtoDB():
    client = MongoClient('localhost', 27017)
    # インポート先
    db = client.tweet_db
    col = db.FireHD10

    id_dict = {}
    count = 0

    write_list = []



    with open("Data/firehd10_origin.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # ヘッダ行
            if row[0] == "text" and row[1] == "id" and row[2] == "created_at":
                continue
            # 要素の数が３でない
            if len(row) != 3:
                print("1行の要素が3つでない")
                break
            # 重複チェック
            if row[1] in id_dict.keys(): continue

            id_dict.setdefault(row[1], 1)
            count += 1

            post = {"text": row[0],
                    "id": row[1],
                    "created_at": change_time(row[2])
                    }
            col.insert_one(post)
            write_list.append([row[0], row[1], row[2]])


    with open("Data/firehd10.csv", 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in write_list:
            writer.writerow(row)


"""
リストをサブリストに分割する
:param l: リスト
:param n: サブリストの要素数
:return: 
"""
def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


"""
MongoDBのコレクションをcsv出力
"""
def DBtotcsv():
    client = MongoClient('localhost', 27017)
    db = client.dbname
    col = db.colname
    docs = col.find(projection={'_id':0})
    count = 0
    with open("Data/csvfilename.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for doc in docs:
            text = doc["text"]
            text = text.replace("\"", "")
            writer.writerow([text, doc["id"], doc["created_at"]])


"""
UTC時間をJST時間に補正
"""
def change_time(created_at):
    st = time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')  # time.struct_timeに変換
    utc_time = datetime.datetime(st.tm_year, st.tm_mon, st.tm_mday, \
                                 st.tm_hour, st.tm_min, st.tm_sec,
                                 tzinfo=datetime.timezone.utc)  # datetimeに変換(timezoneを付与)
    jst_time = utc_time.astimezone(pytz.timezone("Asia/Tokyo"))  # 日本時間に変換
    str_time = jst_time.strftime("%Y-%m-%d %H:%M:%S")  # 文字列で返す
    return str_time


def update_collection():
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    col = db.www
    docs = col.find()
    for doc in docs:
        # doc = col.find_one()
        date = doc["created_at"]
        if re.search(r'^[A-Za-z].+', date):
            doc["created_at"] = change_time(date)
            col.save(doc)


"""
階層的クラスタリングのコーフェン相関係数を検証
"""
def clustering_cophenet(dataflame):
    # アルゴリズム
    METHOD = ["single", "complete", "average", "weighted",
               "centroid", "median", "ward"]

    #距離指標
    metric = 'euclidean'  # ユークリッド距離

    for method in METHOD:
        linkage_result = linkage(dataflame, method=method, metric=metric)
        c, d = cophenet(linkage_result, dataflame)

        print("{0} {1}".format(method, c))

        # plt.figure(num=None, figsize=(17, 17), dpi=200, facecolor='w', edgecolor='k')
        # dendrogram(linkage_result, labels=dataflame.index)
        # plt.xlabel("Product Name_Topic Number")
        # plt.ylabel(metric + " distance")
        # plt.title(method)
        # plt.show()



def tes():

    nodes = mecab_tolist("バッテリに新型と新モデルは、\n予約確認済みで、受け取り予定。受取と受取り")
    for node in nodes:
        print(node)

    print(mecab("バッテリに新型と新モデルは、apple watch series 5", "-Owakati"))

    # text = "アップルウォッチシリーズ 5"
    # print(text.replace("アップルウォッチシリーズ 5", "apple watch series 5"))


if __name__ == '__main__':
    importcsvtoDB()
    # comb_collections()
    # update_collection()
    # tes()




    sys.exit