# - coding: utf-8 -

import csv
import sys
import time
import json
import tweepy

# Constants
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



# APIインスタンス生成
# インスタンスと今居るkey idを返す
def create_API(key_id):
    if key_id > len(CK_LIST): key_id = 0
    auth = tweepy.OAuthHandler(CK_LIST[key_id], CS_LIST[key_id])
    auth.set_access_token(AT_LIST[key_id], AS_LIST[key_id])
    return tweepy.API(auth), key_id


# ワード検索
def get_tweets_byquery(word:str):
    api, key_id = create_API(0)
    tweets = api.search(q=word, count=10)
    for tweet in tweets:
        text = text_former(tweet.text)
        print(tweet.text + "\n")

# status id検索
def get_bystatus(id: str):
    api, key_id = create_API(0)
    tweet = api.get_status(id)
    print(tweet)


# ツイート文章を成形する　作成中
# 1. 改行を削除
# 2
def text_former(text: str):
    if '\n' in text: text = text.replace('\n', '')
    return text


# リスト→csv
def write_listtocsv(filename, list):
    with open(filename, 'w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list)


# csv(のn列目)→リスト
def load_csvtolist(filename, n):
    list = []
    with open(filename, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[5] is "1": list.append(row[n])

    return list


# status idからツイートを取得
def get_tweets_bystatusids():
    key_index = 0  # Consumer, Accessキーリストのインデックス
    statusid_list = []  # 集めたいツイートのidのリスト
    ignore_list = []    # 取得対象外のツイートid　プロテクト、削除済、アカウント削除等
    unableid_list = []  # 取得できなかったツイートid
    output = ""

    # 集めるツイートのidをcsvからload
    with open("Data/Twitter/tweets_open.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(statusid_list) >= 500: break  # 500件のツイート
            statusid_list.append(row[2])

    # ツイート取得

    api, key_id = create_API(0)  # Twitterインスタンス
    for status_id in statusid_list:
        if status_id in ignore_list:            # 既知の対象外idを
            print(status_id + " スキップ")
            continue
        if key_id is len(CK_LIST) - 1: key_index = 0     # APIキーを回す

        try:
            index = statusid_list.index(status_id)
            text = api.get_status(status_id)
            print(text)
            output += str(status_id) + ',' + text + '\n'

        except tweepy.TweepError as e:
            if "Rate limit exceeded" in str(e):
                # レートリミットの処理
                print("-" * 20 + "レートリミット" + "-" * 20)
                key_index += 1
                api = create_API(key_index)
                statusid_list.insert(index + 1, status_id)  # 次のループで再取得
                continue
            else:
                # 他の例外
                print(str(e))
                unableid_list.append([status_id])
                continue

    with open("Data/Twitter/tweets_0405.csv", 'w', encoding="utf-8", newline='') as f:
        f.writelines(output)

    if len(unableid_list) is not 0:
        with open("Data/Twitter/unableids_1_1.csv", 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(unableid_list)


# 作成中 status idのリストからツイートを集める
# ==============================================
# status id100個ずつのリストの多次元リスト
# statusid_list = [
#   [1, 2, 3, ...], [101, 102, 103, ...], [201, 202, 203, ...]
# ]
# ==============================================
def get_bystatusid_list():
    idset = []  # id100個が入る
    texts = []  # 収集できたテキストのリスト
    count = 0   # レートリミット

    api, key_id = create_API(0)     # api=Twitterインスタンス, key idでアクセスキーを管理する
    allids = load_csvtolist("Data/Twitter/tweets_open.csv", 2)     # csvからstatus idをロード

    print("status idから100個ずつツイート")
    for id in allids:
        idset.append(id)
        if len(idset) is not 100: continue  # id100個たまってから次へ

        # レートリミットの検証
        # tryしてレートリミットでなければwhileループを抜け、後ろのstatus objの取得処理へすすむ
        # 例外があった場合、、
        #   レートリミットでない⇒ログを残してbreak
        #   レートリミット     ⇒key idを変えてインスタンスを生成しなおす
        #   レートリミットが２回目なら、15分待ってからリトライ
        while True:
            try: obj = api.user_timeline(screen_name="TwitterJP")
            except tweepy.TweepError as e:
                if "Rate limit exceeded" in str(e):     # レートリミット
                    print("----------レートリミット----------")
                    count += 1
                    if count == 2:
                        print("sleep...")
                        time.sleep(60 * 15)
                        count == 0

                    key_id += 1
                    api, key_id = create_API(key_id)
                else:       # レートリミット以外の例外処理
                    with open("Data/Twitter/exceptionlog.txt", 'a', encoding="utf-8") as f:
                        f.write(str(e) + "\n")
                    break
            else: break

        # 100個のstatus objを取得
        while True:
            try:
                objects = api.statuses_lookup(idset, include_entities="False", trim_user="True")
                if len(objects) is not 0: print("objects is exsits")
                for object in objects:
                    text = object.text
                    if text is not "":
                        s = 1
                    print(text)
                    texts.append(''.join(object.text.splitlines()))
                idset = []
                break

            except tweepy.TweepError as e:
                if "Rate limit exceeded" in str(e):
                    # レートリミット
                    print("----------レートリミットレートリミット----------")
                    if count == 2:
                        time.sleep(60 * 15)
                        count == 0

                    key_id += 1
                    api, key_id = create_API(key_id)
                    count += 1
                    continue    # 取得をやり直す
                else:
                    with open("Data/Twitter/exceptionlog.txt", 'a', encoding="utf-8") as f:
                        f.write(str(e) + "\n")
                    break
            else: break


    # 収集が終了、ファイルへのテキスト書き込み
    # write_listtocsv("Data/Twitter/collected_texts20.csv", texts)
    with open("Data/Twitter/collected_texts_nega.txt", 'w', encoding="utf-8") as f:
        f.write('\n'.join(texts))


# 作成中 status idリストでツイートを集め、メタ情報とともにCSVに出力
# ==============================================
# status id100個ずつのリストの多次元リスト
# statusid_list = [
#   [1, 2, 3, ...], [101, 102, 103, ...], [201, 202, 203, ...]
# ]
# ==============================================
def get_tweets_tocsv():
    idset = []      # id 100個
    texts = []      # 収集できたテキスト
    count = 0       # レートリミット

    api, key_id = create_API(0)     # Twitterインスタンス生成

    allids = []         # status idのリスト
    # 1. CSVからpをラベル付けされたstatus idをloadする
    with open("Data/Twitter/tweets_open.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)       # ヘッダー行
        for row in reader:
            if row[5] is "1": allids.append(row[2])
            # 3行目row[4]: ポジティブのフラグ
            # 4行目row[5]: ネガティブのフラグ


    # 2. status idから100個ずつツイート
    print("status idでツイート収集")
    for id in allids:
        if id is None: continue
        idset.append(id)
        if len(idset) is not 100: continue      # ここまでで100個のセットが出来上がる

        # レートリミットの検証
        #   レートリミットでない⇒ログを残してbreak
        #   レートリミット     ⇒key idを変えてインスタンスを生成しなおす
        #   レートリミットが２回目なら、15分待ってからリトライ
        while True:
            try:
                obj = api.user_timeline(screen_name="TwitterJP")
            except tweepy.TweepError as e:
                if "Rate limit exceeded" in str(e):  # レートリミット
                    print("----------レートリミット----------")
                    count += 1
                    if count == 2:
                        print("sleep...")
                        time.sleep(60 * 15)
                        count == 0

                    key_id += 1
                    api, key_id = create_API(key_id)
                else:
                    with open("Data/Twitter/exceptionlog.txt", 'a', encoding="utf-8") as f:
                        f.write(str(e) + "\n")
                    break
            else: break

        # statusオブジェクト100個を取得
        while True:
            try:
                objects = api.statuses_lookup(idset, include_entities="False", trim_user="True")
                if len(objects) is not 0: print("objects exsists")
                print(objects)
                for object in objects:      # オブジェクト１個ずつ
                    text = object.text
                    if text is "": continue
                    print(text)
                    texts.append(''.join(text.splitlines()))

                idset.clear()
                break

            except tweepy.TweepError as e:
                if "Rate limit exceeded" in str(e):
                    # レートリミット
                    print("----------レートリミットレートリミット----------")
                    count += 1
                    if count == 2:
                        print("sleep...")
                        time.sleep(60 * 15)
                        count == 0

                    key_id += 1
                    api, key_id = create_API(key_id)
                    continue  # 取得をやり直す
                else:
                    with open("Data/Twitter/exceptionlog.txt", 'a', encoding="utf-8") as f:
                        f.write(str(e) + "\n")
                    break
            else: break


    # 収集が終了、ファイルへのテキスト書き込み
    write_listtocsv("Data/Twitter/collected_texts10.csv", texts)
    with open("Data/Twitter/collected_texts08.txt", 'w', encoding="utf-8") as f:
        f.write('\n'.join(texts))




if __name__ == '__main__':
    # get_tweets_bystatusids()
    get_bystatusid_list()
    # get_tweets_tocsv()
    # get_tweets_byquery("バイリー")


sys.exit()
