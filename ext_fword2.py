# - coding: utf-8 -
"""
賛否の度合いを考慮した特徴語の抽出
"""

import MeCab
import mojimoji
import re
import sys
import math
import tools
from tools import mecab
from tools import tweet_former
from tools import mecab_tolist
from pymongo import MongoClient


"""
極性辞書読み込み
-極性値の絶対値に閾値を設定
-1文字の極性語をはじく
-別コレクションに登録
"""
def load_worddict(thres:float):
    worddict = {}  # word: param
    client = MongoClient('localhost', 27017)
    db = client.worddict
    col = db.log10
    for doc in col.find():
        if abs(doc["param"]) < thres: continue  # 極性値の閾値
        if len(doc["word"]) == 1: continue
        worddict[doc["word"]] = doc["param"]

    col = db.log10_over05
    col.remove({})
    for word, param in worddict.items():
        try:
            col.insert_one({"word": word, "param": param})
        except Exception as e:
            print(word + " " + str(param) + " " + str(e))

    return worddict


"""
モジュール1. 評価表現文の抽出
"""
def ext_evalsentence():
    ev_sentences = []
    worddict = load_worddict(0.5)

    marks = ['!', '?', '。', '・・・']

    # ツイートを文単位に分割処理
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    col = db.unique_tweets
    for doc in col.find():
        # 不要部除去-> !,?,。,・・・を"\n"に統一-> 連続する"\n"をまとめる
        text = tweet_former(doc["text"])
        for mark in marks: text = text.replace(mark, "\n")
        text = re.sub(r'\n{1,}', '\n', text)
        # \nで文単位分割
        sentences = [a for a in text.split('\n') if a != '']
        print("\n[文分割]\n" + '\n'.join(map(str, sentences)))
        # 評価表現文抽出
        print("[評価表現文]")
        for sentence in sentences:
            keitaiso = mecab(sentence, "-Owakati").split(" ")
            for w in keitaiso:
                if w in worddict.keys():
                    print(sentence)
                    ev_sentences.append({"EVsentence": sentence})
                    break

    # 評価表現文sentencesを登録
    db = client.tweet_db
    db.EVs_over05.remove({})
    db.EVs_over05.insert_many(ev_sentences)

"""
モジュール2. 特徴語候補の抽出と極性判定
入力:評価表現文リスト[str]
出力:特徴語候補の情報{word, [肯定評価数, 否定評価数, 出現する評価表現文の数, 肯定評価率(null), 特徴度(null)]}

-評価表現文から評価属性候補を取り出す
-属性候補の極性を判定する

"""
def det_polarity():
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    ev_sentences = [doc["EVsentence"] for doc in db.EVs_over05.find()]
    worddict = load_worddict(0.5)

    result = [] # [{"word": word, "P": p, "N": n, "sf": sf}, ...]
    for ev_sentence in ev_sentences:
        ca_lists = []
        po_tuples = []
        for node in mecab_tolist(ev_sentence):  # (word, class1, class2, class3, origin(or""), position)
            # 特徴語候補抽出
            if node[1] == "名詞":
                if "iPhone" in node[0] or "iphone" in node[0]: continue
                if node[2] != "非自立" and node[2] != "形容動詞語幹" \
                        and node[2] != "代名詞" and node[2] != "副詞可能" \
                        and node[2] != "接尾":
                    ca_lists.append([node[0], node[-1], 1000])  # [word, position, temp_dist]
            # 極性語抽出
            if node[1] != "名詞" and node[1] != "助詞" and node[0] in worddict.keys():
                po_tuples.append((node[0], node[-1], worddict[node[0]]))  # [word, position, param]

        if not ca_lists: continue
        # 極性語ごとに
        for pol in sorted(po_tuples, key=lambda x: x[1], reverse=True): # 位置の降順ソート:
            for cand in ca_lists: cand[2] = abs(cand[2] - pol[1])  # 距離を算出値に更新

            ca_lists = sorted(ca_lists, key=lambda x: x[2])  # 距離の昇順ソート
            closest = ca_lists[0]  # 最も近い候補語を取り出す

            feature_word = [closest[0], closest[1], closest[2], pol[2]]  # [word, position, dist, param]

            # DB登録
            # リストが空のとき
            if not result:
                result.append({"word": feature_word[0],
                               "P": 1 if feature_word[3] > 0.0 else 0,
                               "N": 1 if feature_word[3] < 0.0 else 0,
                               "sf": 1,
                               "P_rate": None,
                               "T": None
                               })
            else:
                words = []
                for dict in result: words.append(dict["word"])
                # 未登録
                if feature_word[0] not in words:
                    result.append({"word": feature_word[0],
                                   "P": 1 if feature_word[3] > 0.0 else 0,
                                   "N": 1 if feature_word[3] < 0.0 else 0,
                                   "sf": 1,
                                   "P_rate": None,
                                   "T": None
                                   })
                # 登録済
                else:
                    for dict in result:
                        if dict["word"] == feature_word[0]:
                            dict["P"] = dict["P"] + 1 if feature_word[3] > 0.0 else dict["P"]
                            dict["N"] = dict["N"] + 1 if feature_word[3] < 0.0 else dict["N"]
                            dict["sf"] = dict["sf"] + 1

    for dict in result:
        dict["P_rate"] = dict["P"] / dict["sf"]

    db = client.tweet_db
    db.ca_featureword.remove({})
    db.ca_featureword.insert_many(result)

"""
モジュール3. 特徴度計算
-特徴語候補の特徴度計算
-特徴度上位の特徴語を抽出
入力：特徴語候補のリスト[{"word": word1, "P": p1, "N": n1, "sf": sf1},...]
出力：
"""
def ext_featurewords():
    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    docs = db.ca_featureword.find(projection={'_id':0})

    for doc in docs:
        P_rate = doc["P_rate"]
        sf = doc["sf"]
        d = 1.64 * math.sqrt(P_rate*(1-P_rate)/sf)
        P_R = abs(P_rate+d)
        P_L = abs(P_rate-d)
        D_R = abs(0.5-P_R)
        D_L = abs(0.5-P_L)
        cal = 1 - (D_R + D_L) / 2

        db.ca_featureword.update_one({"word": doc["word"]}, {'$set': {"T": cal}})



""""
各形態素について
-名詞-> 評価属性候補に
-それ以外-> 極性辞書で検索
"""
def tesst():
    worddict = load_worddict(0.5)
    ev_sentence = "なので、俺は11000円余分に払えば、すぐにもうハード料金だけ払えばiPhone 11 proに変更可能。でも11000円もったいないからオンラインショップにした。"

    #↓　単一の評価表現文についての処理
    result = [] # word: [P, N, sf]
    ca_lists = []
    po_tuples = []
    for node in mecab_tolist(ev_sentence): # (word, class1, class2, class3, origin(or""), position)
        # 特徴語候補抽出
        if node[1] == "名詞":
            if node[2] != "非自立" and node[2] != "形容動詞語幹" \
                    and node[2] != "代名詞" and node[2] != "副詞可能" \
                    and node[2] != "接尾":
                ca_lists.append([node[0], node[-1], 1000])   # [word, position, temp_dist]
        # 極性語抽出
        if node[1] != "名詞" and node[0] in worddict.keys():
            po_tuples.append((node[0], node[-1], worddict[node[0]])) # [word, position, param]

    po_tuples = sorted(po_tuples, key=lambda x: x[1], reverse=True)   # 位置の降順ソート
    # 極性語ごとに
    for pol in po_tuples:
        flag = False
        for cand in ca_lists: cand[2] = abs(cand[2] - pol[1])   # 距離を算出値に更新

        ca_lists = sorted(ca_lists, key=lambda x: x[2])   # 距離の昇順ソート
        closest = ca_lists[0]   # 最も近い候補語

        feature_word = [closest[0], closest[1], closest[2], pol[2]] # [word, position, dist, param]

        # 出力
        if not result:
            result.append({"word": feature_word[0],
                           "P": 1 if feature_word[3] > 0.0 else 0,
                           "N": 1 if feature_word[3] < 0.0 else 0,
                           "sf": 1
                           })
            continue
        for dict in result:
            if feature_word[0] not in dict:
                result.append({feature_word[0]: [1 if feature_word[3] > 0.0 else 0,
                                                 1 if feature_word[3] < 0.0 else 0,
                                                 1]
                               })
            else:
                before = dict[feature_word[0]]
                result[feature_word[0]] = [before[0]+1 if feature_word[3] > 0.0 else before[0],
                                           before[1]+1 if feature_word[3] < 0.0 else before[1],
                                           before[2]+1]

    client = MongoClient('localhost', 27017)
    db = client.tweet_db
    db.ca_featureword.remove({})
    db.ca_featureword.insert_many(result)



def ttt():
    post = {"word": "う", "num": 3, "check": 3}
    str = "iPhone 8をヤフオクで売却できたので、"
    list = mecab(str, "-Owakati").split(" ")
    i = 1
    for w in list:
        print("{0}\n{1}".format(i, mecab_tolist(w)))
        i += 1


def main():
    # ext_evalsentence()
    # det_polarity()
    # ext_featurewords()
    # test()
    # print(mecab("iPhone 11 Pro欲しい", "-Ochasen"))
    ttt()


if __name__ == '__main__':
    main()

    sys.exit()