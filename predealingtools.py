# - coding: utf-8 -

import sys
import math

# Step.1 単語のindex付け
# フォーマットは <星の数> _SEP_ <分かち書き済レビュー文>
def term_indexer(review_path, term_index_path):
    term_index = {}
    term_index_str = ""
    index_num = 1
    with open(review_path, 'r', encoding="utf-8") as file:
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

    with open(term_index_path, 'w', encoding="utf-8") as out_file:
        out_file.write(term_index_str)
    print("単語のid付け完了")


# Step.2 フォーマット化
# 入力： <term1> <index1><term2> <index2><term3> <index3>
# 出力： libsvmフォーマット
# <label> <index1>:<value1> <index2>:<value2> ...
# 1 1:0.12 2:0.4 3:0.1 ...
# 0 2:0.59 4:0.1 5:0.01 ...
def svm_indexer(review_path, term_index_path):
    term_id = {}
    # <term1> <index1>\n<term2> <index2>\n ... -> {<term1>:<index1>, <term2>:<index2>, ...}
    with open(term_index_path, 'r', encoding="utf-8") as term_index_file:
        term_index = term_index_file.read()
        for i in term_index.split('\n'):
            term_index_pair = i.split(" ")
            if len(term_index_pair) < 2: continue
            term_id.setdefault(term_index_pair[0], int(term_index_pair[1]))

    # <star> __ SEP __ <review sentence> ->
    print("レビュー文中の単語の出現頻度と単語idを照合し、libsvm形式になおす")
    with open(review_path, 'r', encoding="utf-8") as review_wakati_file:
        output = ""
        for line in review_wakati_file.readlines():
            if line is None: continue
            ents = line.split(" __ SEP __ ")
            if len(ents) < 2: continue
            star = float(ents[0].replace("\ufeff", "").replace(" ", ""))
            terms = ents[1]

            # レビュー文中の単語の出現頻度　{<term1>:<freq1>, <term2>:<freq2>, ...}
            term_freq = {}
            for term in terms.split(" "):
                if term is '\n': continue
                if not term in term_freq.keys(): term_freq[term] = 0.0
                else: term_freq[term] += 1.0

            # 単語のインデックスと頻度(log重み)　
            id_freq = {}
            for term in term_freq:
                id_freq[term_id[term]] = round(math.log(term_freq[term] + 1.0), 3)
            # {<id1>:<log_freq1>, <id2>:<log_freq2>, ...} idでソート
            id_freq = dict(sorted(id_freq.items()))


            # 星の数ラベル、単語インデックス、頻度重みをlibsvmフォーマットに
            if star > 4.0: ans = str(1)
            else : ans = str(0)
            #else : ans = str(-1)
            for id, freq in id_freq.items():
                if id is not None:
                    ans += " {0}:{1}".format(str(id), str(freq))
                    # ans: <class> <key1>:<value1> <key2>:<value2> ...
            output += ans + '\n'
            print(ans)

    with open("Data/svm_fmt_test1026.txt", 'w', encoding="utf-8") as out_file:
        out_file.write(output)


if __name__ == '__main__':
    # レビュー文と単語idデータのパス
    review_path = "Data/rakuten_reviews_wakati_test.txt"    # レビュー文
    term_index_path = "Data/term_index_test1026.txt"        # 単語―id データ
    model_path = "Data/svm_fmt_test1026.txt.model"          # 機械学習モデル


    term_indexer(review_path, term_index_path)

    svm_indexer(review_path, term_index_path)


sys.exit()
