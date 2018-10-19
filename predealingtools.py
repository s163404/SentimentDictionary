# - coding: utf-8 -

import sys
import math

# Step.1 単語のindex付け
#フォーマットは <星の数> _SEP_ <分かち書き済レビュー文>
def term_indexer():
    term_index = {}
    term_index_str = ""
    index_num = 1
    with open("Data/rakuten_reviews_wakati_test.txt", 'r', encoding="utf-8") as file:
        for l in file.readlines():
            if l is None: continue
            ents = l.split(" __ SEP __ ") #ents[0] 星の数、 ents[1] レビュー文
            if len(ents) < 2: continue
            terms = ents[1] #分かち書き済レビュー文

            terms = terms.split(" ")    # 単語のlist
            # term_indexのkeyに単語が登録されていなければ、valueにindexを振っていく
            for term in terms:
                if term in term_index.keys(): continue
                term_index.setdefault(term, str(index_num))
                term_index_str += "{0} {1}\n".format(term, str(index_num))
                index_num += 1
                # <term1> <index1><term2> <index2><term3> <index3>...

    with open("Data/term_index_test1019.txt", 'w', encoding="utf-8") as out_file:
        out_file.write(term_index_str)

    print("単語のid付け完了")
    return term_index       # term, index の単語辞書

# Step.2 フォーマット化
# 高次元データの場合、スパースで大規模なものになりやすく、この場合、Pythonなどのラッパー経由だと正しく処理できないことがあります。そのため、libsvm形式と呼ばれる形式に変換して扱います。
# 直接、バイナリに投入した方が早いので、以下の形式に変換します。
# 1 1:0.12 2:0.4 3:0.1 ...
# 0 2:0.59 4:0.1 5:0.01 ...


if __name__ == '__main__':
    dict_term_id = term_indexer()


sys.exit()
