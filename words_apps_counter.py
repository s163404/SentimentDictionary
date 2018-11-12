# - coding: utf-8 -

import sys
import re

review_path = "Data/rakuten_reviews_wakati.txt"
original_review_path = "Data/rakuten_reviews.txt"
dictionary_path = "Data/dictionary_test1108.txt"


# 入力した極性値以上の単語を取り出し、出現回数をカウントする
def word_counter():
    pdict_to_count = {}  # <word>, <count>
    ndict_to_count = {}
    # -- 極性値の閾値 --
    pbased_value = 0.7      # ⇒0.7以上のポジ単語
    nbased_value = -0.6     # ⇒-0.6以下のネガ単語

    print("valueの閾値で処理した単語-出現頻度辞書")
    # 辞書読み込み
    with open(dictionary_path, 'r', encoding="utf-8") as file:
        dict = {}
        for item in file.readlines():
            if item is None: continue
            item = re.sub(r'(.+) $', r'\1', item).split(" ")
            if len(item) is not 2: continue
            dict[item[0]] = float(item[1].replace('\n', ''))

    # ポジ/ネガ別に極性値でフィルタをかけた辞書(単語, 出現回数)を作る
    for w, v in dict.items():
        if v >= pbased_value: pdict_to_count[w] = 0
        elif v <= nbased_value: ndict_to_count[w] = 0
        else: continue

    # レビュー文を検索しカウント()
    with open(review_path, 'r', encoding="utf-8") as file:
        reviews = file.read().split('\n')
    for review in reviews:
        review = re.sub(r'.+ __ SEP __ (.+)$', r'\1', review)   # セパレータと評価の数字をreplace
        if review is None: continue
        for pword, count in pdict_to_count.items():             # ポジ辞書で検索
            #if pword in review: pdict_to_count[pword] += 1
            pdict_to_count[pword] += review.count(pword)
        for nword, count in ndict_to_count.items():             # ネガ辞書で検索
            #if nword in review: ndict_to_count[nword] += 1
            ndict_to_count[nword] += review.count(nword)

    with open("Data/dictionary_to_check.txt", 'w', encoding="utf-8") as file:
        for pword, count in pdict_to_count.items(): file.write("{0} {1}\n".format(pword, str(count)))
        for nword, count in ndict_to_count.items(): file.write("{0} {1}\n".format(nword, str(count)))
    print("辞書出力完了")


# 検索単語を含むレビュー文を抽出
def review_search(search_word):
    result = ""
    result_path = "Data/KensakuReviews/kensaku_reviews_{0}.txt".format(search_word)
    with open(original_review_path, 'r', encoding="utf-8") as file:
        for review in file.readlines():
            if search_word in review: result += "{0}".format(review)

    with open(result_path, 'w', encoding="utf-8") as out_file:
        out_file.write(result)
    print(" {0} で検索したレビューの出力完了".format(search_word))



if __name__ == '__main__':

    #word_counter()

    review_search("後輩")

sys.exit()
