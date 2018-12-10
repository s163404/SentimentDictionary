# - coding: utf-8 -

import sys
import math
import re

# Step.1 単語のindex付け
# 入力フォーマットは<星の数> _SEP_ <分かち書き済レビュー文>
# 出力フォーマットは<term1> <index1>\n<term2> <index2>\n...　
def term_indexer(review_path, term_index_path):
    term_index = {}
    term_index_str = ""
    index_num = 1
    sentences = []
    print("単語のid付け")
    with open(review_path, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            if line is None: continue
            ents = line.split(" __ SEP __ ")  # ents[0] 星の数、 ents[1] レビュー文
            if len(ents) is not 2: continue
            sentences.append(ents[1])  # レビュー文

    # レビューごとにループ、さらにネストでレビュー内の単語でループを回す
    for sentence in sentences:
        for term in sentence.split(" "):
            if term in term_index.keys() or term is None or term is "\n": continue
            # term_indexのkey(単語)に重複が無ければ、valueにindexを振っていく
            term_index.setdefault(term, str(index_num))
            term_index_str += "{0} {1}\n".format(term, str(index_num))
            index_num += 1

    with open(term_index_path, 'w', encoding="utf-8") as out_file:
        out_file.write(term_index_str)
    print("単語のid付け完了")


# Step.2 フォーマット化
# 入力： <term1> <index1><term2> <index2><term3> <index3>
# 出力： libsvmフォーマット
# <label> <index1>:<value1> <index2>:<value2> ...
# 1 1:0.12 2:0.4 3:0.1 ...
# 0 2:0.59 4:0.1 5:0.01 ...
def svm_indexer(review_path, term_index_path, output_path):
    term_id = {}
    # <term1> <index1>\n<term2> <index2>\n ... -> {<term1>:<index1>, <term2>:<index2>, ...}
    with open(term_index_path, 'r', encoding="utf-8") as term_index_file:
        term_index = term_index_file.read()
        for i in term_index.split('\n'):
            term_index_pair = i.split(" ")
            if len(term_index_pair) < 2: continue
            term_id.setdefault(term_index_pair[0], int(term_index_pair[1]))

    # <star> __ SEP __ <review sentence> ->
    print("レビュー文中の単語の出現頻度を単語idと照合し、libsvm形式になおす")
    with open(review_path, 'r', encoding="utf-8") as review_wakati_file:
        output = ""
        for line in review_wakati_file.readlines():     # レビュー文ごと
            if line is None: continue
            ents = line.split(" __ SEP __ ")
            if len(ents) < 2: continue
            if re.match("^[0-9]{1}.?[0-9]{0,2}$", ents[0].replace("\ufeff", "").replace(" ", "")) is None: continue
            star = float(ents[0].replace("\ufeff", "").replace(" ", ""))
            terms = ents[1]

            # １つのレビュー文について、単語の出現回数を数える　{<term1>:<freq1>, <term2>:<freq2>, ...}
            term_freq = {}
            for term in terms.split(" "):       # 1レビュー中の単語ごと
                if term is '\n': continue
                if not term in term_freq.keys(): term_freq[term] = 0.0
                else: term_freq[term] += 1.0
            # 数えた単語のidと頻度(log重み)を対応させる
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
                if id is None: continue
                ans += " {0}:{1}".format(str(id), str(freq))
                # ans: <class> <key1>:<value1> <key2>:<value2> ...
            output += ans + '\n'
            # print(ans)

    with open(output_path, 'w', encoding="utf-8") as out_file:
        out_file.write(output)
    print("libsvm出力完了")

# Step.3 機械学習
# pythonバインディングができず保留

# Step.4 学習結果と単語idを衝突させる
def weight_checker(term_index_path, model_path):
    print("学習結果と単語idを照合し、極性辞書を出力する")
    term_id = {}
    # <term1> <index1>\n<term2> <index2>\n ... -> {<term1>:<index1>, <term2>:<index2>, ...}
    with open(term_index_path, 'r', encoding="utf-8") as term_index_file:
        term_index = term_index_file.read()
        for i in term_index.split('\n'):
            term_index_pair = i.split(" ")
            if len(term_index_pair) < 2: continue
            term_id.setdefault(term_index_pair[0], int(term_index_pair[1]))

    weight_id = {}
    i = 1
    # modelファイルを読み、重みと単語idを対応づける
    with open(model_path, 'r', encoding="utf-8") as model_file:
        model = model_file.read().split('\n')
        for line in model[6:]:  # modelファイルの6行目以降から
            weight_id.setdefault(line, i)
            i += 1
        weight_id = dict(sorted(weight_id.items()))
        # {<weight1>:<index1>, <weight2>:<index2>, ...}

    dictionary_str = ""
    for weight, w_id in weight_id.items():
        for term, t_id in term_id.items():
            if w_id is t_id: dictionary_str += "{0} {1}\n".format(term, weight)

    with open("Data/dictionary_test1108.txt", 'w', encoding="utf-8") as out_file:
        out_file.write(dictionary_str)
    print("辞書出力完了")





if __name__ == '__main__':
    # レビュー文と単語idデータのパス
    review_path = "Data/rakuten_reviews_wakati.txt"         # レビュー文
    term_index_path = "Data/term_index_1108.txt"            # 単語―id データ
    libsvm_data_path = "Data/svm_1108.fmt"  # libsvmフォーマットのデータ
    model_path = "Data/svm_1108.fmt.model"          # 機械学習モデル


    #term_indexer(review_path, term_index_path)

    #svm_indexer(review_path, term_index_path, libsvm_data_path)

    weight_checker(term_index_path, model_path)


sys.exit()
