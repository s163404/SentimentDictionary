# - coding: utf-8 -
"""
クエリ検索によるツイート収集

MongoDBにinsertする
"""

import csv
from pymongo import MongoClient
import re
import sys
import time, locale, calendar
import json
import tweepy

"""
Constants
"""
CK_LIST = []
CS_LIST = []
AT_LIST = []
AS_LIST = []

with open("C:/Users/KMLAB-02/Twitter_Keys/TwitterAPIkeys.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            CK_LIST.append(row[0])
            CS_LIST.append(row[1])
            AT_LIST.append(row[2])
            AS_LIST.append(row[3])

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
APIインスタンス生成
渡されたkey idが超過していた時の対応処理
return : apiインスタンス、現在のkey id
"""
def create_API(key_id):
    if key_id >= len(CK_LIST)-1: key_id = 0
    auth = tweepy.OAuthHandler(CK_LIST[key_id], CS_LIST[key_id])
    auth.set_access_token(AT_LIST[key_id], AS_LIST[key_id])
    return tweepy.API(auth), key_id

"""
ワードからツイート(countの数だけ)を取得
"""
def get_tweetobj(query:str):
    api, key_id = create_API(0)
    results = api.search(query,lang='ja', count=100)
    for result in results:
        if result.retweeted is True:
            print("RTされたツイート: " + result.id_str)
            continue
        print(result.id_str + " " + result.text)
        
"""
単一idからツイートをreturnするだけ
"""
def get_bystatus(id: str):
    api, key_id = create_API(0)
    obj = api.get_status(id)
    return obj

"""
単一idで取得し、entitiesを取り出す
"""
def get_entities(id:str):
    obj = get_bystatus(id)
    return obj.entities     # 辞書型

"""
collectionから取りたいdataを抜き出してリストで返す
"""
def get_datatolist(collection, data:str):
    id_list = []
    for doc in collection.find(): id_list.append(doc[data])

    print("{0}の数: {1}".format(data, str(len(id_list))))
    return id_list

"""
idリストでツイートを取得し、DBに格納
"""
def get_tweetsbyids():
    # DB呼び出し
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    collection = db.unique_tweets

    id_list = get_datatolist(collection, "id")
    id_list = list(split_list(id_list, 100))

    for ids in id_list:


        print("")





"""
単一idでツイートを取得しハッシュタグワードを返す
"""
def get_tagwords(id:str):
    tag_words = []
    obj = get_bystatus(id)
    entities = obj.entities # entities は辞書型
    if not entities: print("entities empty")
    hashtags = entities["hashtags"]
    if not hashtags:
        # TODO hashtagがNoneのときの対応
        print("------ハッシュタグなし------")
        return ""
    else:
        for item in hashtags: tag_words.append(item["text"])
        print(tag_words)
        return tag_words

"""
レートリミットの検証と対応　api:twitterapiインスタンス,method:チェックしたいメソッド名
"""
def proc_ratelimit(api, key_id):
    count = 0
    while True:
        try: results = api.search("ツイッター", lang='ja', count=1)
        except tweepy.TweepError as e:
            # レートリミット
            if "Rate limit exceeded" in str(e):
                print("\n----------  レートリミット  ----------\n")
                count += 1
                if count == 2:
                    print("===================\nsleep...\n===================")
                    time.sleep(60 * 15)
                    count = 0
                    break
                api, key_id = create_API(key_id + 1)
        else: break

    return api, key_id

"""
ワード検索 -> MongoDBへ
"""
def get_tweets_byquery(word:str, max_id):
    texts = []  # 取得したテキストのリスト
    api, key_id = create_API(0)     # 初期APIインスタンス
    limit = 40

    # MongoDB呼び出し
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    collection = db.goprohero8

    # レートリミット検証
    api, key_id = proc_ratelimit(api, key_id)

    # 1回目取得
    if max_id:
        results = api.search(word, lang='ja', count=100, max_id=str(max_id), result_type="recent")
    else:
        results = api.search(word, lang='ja', count=100, result_type="recent")

    while True:
        # 取得結果の確認　-> max_id更新 -> DB格納
        if len(results) is 0:
            print("results なし")
            break

        current_max_id = str(results[-1].id - 1)      # 一番古いstatus id -1
        for result in results:
            print(result)
            if re.match(r'^RT @.*', result.text): continue  # 公式RTはcontinue
            # text = text_former(result.text)     # テキスト成形
            post = {"text": result.text,
                    "id": result.id_str,
                    "created_at": result.created_at,
                    "entities": result.entities
                    }
            collection.insert_one(post)

        if collection.count() >= limit: break
        # if len(texts) >= get_limit: break

        # レートリミット検証と対応処理
        api, key_id = proc_ratelimit(api, key_id)

        # 2回目以降の取得
        results = api.search(word, lang='ja', count=100, max_id=current_max_id, result_type="recent")
        if len(results) is 0:
            print("results なし API再生成")
            api, key_id = create_API(key_id+1)
            results = api.search(word, lang='ja', count=100, max_id=current_max_id, result_type="recent")


    """
    取得終了 -> ファイルwrite
    取得できたところまでwriteする
    """
    # with open("Data/Twitter/iPhone11/texts_iphone11_10000.txt", 'w', encoding="utf-8") as f:
    #     f.write('\n'.join(texts))





"""
メイン
"""
def main():
    get_tweets_byquery("\"GoPro HERO8\"|\"gopro hero8\"|\"goprohero8\"", None)
    # get_tweetobj("\"虫さん\"")
    # print(get_tagwords("1184477648501669888"))
    # get_datatolist("id")
    # get_tweetsbyids()



if __name__ == '__main__':
    main()

sys.exit()
