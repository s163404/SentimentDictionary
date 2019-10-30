# - coding: utf-8 -

"""
EVツイートからの特徴語抽出
"""

import csv
import re
import sys
import MeCab
import mojimoji
import tweepy
from pymongo import MongoClient

"""
Constants
"""
CK_LIST = []
CS_LIST = []
AT_LIST = []
AS_LIST = []

with open("C:/Users/s1634/Twitter_Keys/TwitterAPIkeys.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            CK_LIST.append(row[0])
            CS_LIST.append(row[1])
            AT_LIST.append(row[2])
            AS_LIST.append(row[3])

# """
# APIインスタンス生成
# """
# def create_API(key_id):
#     if key_id >= len(CK_LIST)-1: key_id = 0
#     auth = tweepy.OAuthHandler(CK_LIST[key_id], CS_LIST[key_id])
#     auth.set_access_token(AT_LIST[key_id], AS_LIST[key_id])
#     return tweepy.API(auth), key_id
# 
# # status id検索
# def get_bystatus(id: str):
#     api, key_id = create_API(0)
#     return api.get_status(id)



"""
モジュール1. 極性語を含むツイートを”商品評価ツイート(EVツイート)”として抽出
"""
def ext_evtweets():
    unique_ids = []
    unique_docs = []
    pol_words = []
    ev_tweets = []

    # 極性語を抜き出す
    with open("Data/polarity_2_log10.txt", 'r', encoding="utf-8") as f:
        for row in f.readlines():
            word = re.sub('\t-?[0-9]{1,3}\.?[0-9]*\n?', '', str(row))
            pol_words.append(word)

    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    collection = db.combined
    docs = db.combined.find()
    for doc in docs:
        # 重複を除く
        if doc["id"] in unique_ids: continue
        unique_ids.append(doc["id"])

        unique_doc = {"text": doc["text"],
                      "id": doc["id"],
                      "created_at": doc["created_at"]}
        unique_docs.append(unique_doc)
    db.unique_tweets.insert_many(unique_docs)

    # EVツイートを抽出
    for doc in unique_docs:
        for word in pol_words:
            if word in doc["text"]:
                ev_tweets.append(doc)
                break
    db.EVtweets.insert_many(ev_tweets)

"""
モジュール2 EVツイートのノイズ除去 
"""
def rem_evtext(text:str):
    obj = "type:obj"
    # TODO objはDBから取得する
    text = obj.text

    trans_list = ['!', '?', '・']
    remove_list = [
        "「", "」", "[", "]", "（", "）", "(", ")", "【", "】", "『", "』"
    ]
    # ➀の対応
    if re.search(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', text): return None
    print(text)
    # ➁の対応
    text = re.sub(r"#.+", "", text)


    text = mojimoji.zen_to_han(text, kana=False) # 記号を半角変換
    # ➂の対応
    # (1)
    for mark in trans_list: text = text.replace(mark, "。")
    # (2)
    for mark in remove_list: text = text.replace(mark, "")
    # (3)
    text = re.sub(r"。+", "。", text)

    print(text)



    # 置換後に"。"をまとめる




# csv(のn列目)→リスト
def load_csvtolist(filename, n):
    list = []
    with open(filename, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[5] is "1": list.append(row[n])

    return list


def main():
    # mecab()
    # ext_evtweets()
    # rem_evtext("あいうカキク123４５６ABCDEF!！?？#＃。.・")
    rem_evtext("「あいうえお！！!」 #ハッシュタグ")
    # tweets_crawler.get_bystatus("1184117522070564864")


if __name__ == '__main__':
    main()

sys.exit()
