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
import pytz
import re
import sys
import time

from pymongo import MongoClient

"""
生テキストの前処理
入力：str
出力：str
"""
def pre_process(text:str):
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
    preprocess 内にいれるか、
    transtext単語単位の処理内にいれるか
    """
    dougigo = {}
    with open("Data/dougigo_test.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
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


"""
CSVデータをMongoDBにインポート
"""
def importcsvtoDB():
    client = MongoClient('localhost', 27017)
    # インポート先
    db = client.tweet_db
    col = db.AirpodsPro

    with open("Data/airpodspro.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            post = {"text": row[0],
                    "id": row[1],
                    "created_at": row[2]
            }
            col.insert_one(post)


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


def tes():
    list = ["A", "B", "C"]
    str = "B個"
    for l in list:
        print(l in str)

    sentence = "iosとipadとengadget"
    print(mecab(sentence, "-Ochasen"))


if  __name__ == '__main__':
    # importcsvtoDB()
    # comb_collections()

    sys.exit