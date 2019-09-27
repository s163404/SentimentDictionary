# - coding: utf-8 -

import sys
import math
import MeCab
import re

# Step.1 単語のindex付け
# 入力フォーマットは<星の数> _SEP_ <分かち書き済レビュー文>
# 出力フォーマットは<term1> <index1>\n<term2> <index2>\n...　
def term_indexer(review_path, term_index_path, flag):
    term_index = {}
    term_index_str = ""
    index_num = 1
    sentences = []
    print("単語のid付け")
    with open(review_path, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            if line is None: continue
            if flag is True: ents = line.split(" __SEP__ ")
            else:            ents = line.split(" __ SEP __ ")  # ents[0] 星の数、 ents[1] レビュー文
            if len(ents) is not 2: continue
            sentences.append(ents[1])  # レビュー文

    # レビューごとにループ、さらにネストでレビュー内の単語でループを回す
    for sentence in sentences:
        if flag is True:  # flag=trueなら分かち書き
            m = MeCab.Tagger("-Owakati")
            sentence = m.parse(sentence)

        for term in sentence.split(" "):
            if term in term_index.keys() or term is "" or term is "\n": continue
            # term_indexのkey(単語)に重複が無ければ、valueにindexを振っていく
            term_index.setdefault(term, str(index_num))
            term_index_str += "{0} {1}\n".format(term, str(index_num))
            index_num += 1

    with open(term_index_path, 'w', encoding="utf-8") as out_file:
        out_file.write(term_index_str)
    print("単語のid付け完了")


# Step.2 フォーマット
# レビュー(評価の数、テキスト)をlibsvmの形式に直す
# libsvm形式↓↓↓↓
# <label> <id1>:<value1> <id2>:<value2> ...
# 1 1:0.12 2:0.4 3:0.1 ...
# 0 2:0.59 4:0.1 5:0.01 ...
def svm_indexer(review_path, term_index_path, output_path):
    term_id = {}

    # 1. 単語-id 辞書をロード
    with open(term_index_path, 'r', encoding="utf-8") as term_index_file:
        term_index = term_index_file.read()
    for i in term_index.split('\n'):
        term_index_pair = i.split(" ")
        if len(term_index_pair) < 2: continue
        term_id.setdefault(term_index_pair[0], int(term_index_pair[1]))

    # 2. 評価の数は1または0へ、テキストはidとlog出現回数重みへと変換
    # レビュー一個ずつ見ていく
    # 変換したレビューは1つの変数outputにまとめて
    output = ""
    # <star> __ SEP __ <text>
    print("レビュー文中の単語の出現頻度を単語idと照合し、libsvm形式になおす")
    with open(review_path, 'r', encoding="utf-8") as review_wakati_file:
        reviews = review_wakati_file.read()
    count = 0
    for line in reviews.split('\n'):     # レビュー文ごと
        if line is None: continue
        count += 1
        line = re.sub(" $|　$", "", line)  # 文末にあるスペースを削除してスペーススプリットのエラーに対応
        ents = line.split(" __ SEP __ ")
        ents[0] = ents[0].replace("\ufeff", "").replace(" ", "")    # 余計なコード,スペースを削除

        # 不正な形式をはじく
        if len(ents) is not 2:
            print("line{0} len(ents)≠2 {1}".format(count, str(ents)))
            continue
        if ents[0] is "":
            print("line{0} 評価無し {1}".format(count, str(ents)))
            continue
        if re.match("^[0-9]{1}$|^[0-9]{1}\.[0-9]{0,2}$", ents[0]) is "":    # 1桁の数字または小数点以下2の小数にマッチ
            print("line{0} N.NNまたはN にマッチしない {1}".format(count, str(ents)))
            continue
        try: i = float(ents[0])
        except ValueError:
            print("line{0} float型変換できない {1}".format(count, str(ents)))
            continue
        if ents[1] is "":
            print("line{0} テキスト無し {1}".format(count, str(ents)))
            continue

        star = float(ents[0])    # float
        terms = ents[1]          # レビュー文

        # １つのレビュー文における単語の出現回数を数える　{<id1>:<freq1>, <id2>:<freq2>, ...}
        id_freq = {}
        for term in terms.split(" "):
            id = term_id[term]
            if not id in id_freq.keys(): id_freq[id] = 1.0
            else: id_freq[id] += 1.0

        for id in id_freq.keys():
            id_freq[id] = round(math.log(id_freq[id]+1.0, 10), 3)     # 出現回数のlogをとって辞書valueを上書き
        id_freq = dict(sorted(id_freq.items()))  # idソート

        # 星の数でクラスをラベル付
        # TODO 星の数の閾値調整
        if star >= 3.5: ans = "1"
        elif star < 3.5: ans = "0"      # 3.5未満⇒0, 3.5以上⇒1

        # 単語indexと頻度の重みを文字列結合
        for id, freq in id_freq.items():
            if id is None:
                print("単語のidが振られていない" + str(ents))
                continue
            ans += " {0}:{1}".format(str(id), str(freq))
            # ansの形式↓
            # <class> <key1>:<value1> <key2>:<value2> ...
        output += ans + '\n'

    # 3. ファイル出力
    with open(output_path, 'w', encoding="utf-8") as out_file:
        out_file.write(output)
    print("libsvm出力完了")

# Step.3 機械学習
# pythonバインディングができず保留

# Step.4 学習結果と単語idを衝突させる
def weight_checker(term_index_path, model_path, dictionary_path):
    print("学習結果と単語idを照合し、極性辞書を出力する")
    id_term = {}
    # term_indexファイルをロード
    with open(term_index_path, 'r', encoding="utf-8") as term_index_file:
        term_index = term_index_file.read()
    for i in term_index.split('\n'):
        term_index_pair = i.split(" ")
        if len(term_index_pair) < 2: continue
        id_term.setdefault(int(term_index_pair[1]), term_index_pair[0], )
        # {<index1>:<term1>, <index2>:<term2>,...}

    # 辞書
    # {<id1>:<param1>,
    #  <id2>:<param2>, ...}
    id_param = {}
    # modelから単語の重みを読み込み、idと対応づける
    with open(model_path, 'r', encoding="utf-8") as model_file:
        model = model_file.read().split('\n')
    i = 1  # id
    for param in model[6:]:  # modelの6行目以降から
        if param is "": continue
        try:
            id_param.setdefault(i, float(param.replace(' ', '')))
        except ValueError as e: print("ValueError:{0}line{1} 「{2}」".format(e, i+6, param))
        i += 1

    id_param = dict(sorted(id_param.items(), key=lambda x: x[1]))       # paramでソート

    # 対応するidを検索し、単語とparamを対応付ける
    #   辞書id_term idと単語の対応
    #   辞書id_param idと極性値の対応
    #   一致するidからterm と paramを対応づける
    dictionary_str = ""
    for w_id, param in id_param.items():
        if w_id in id_term.keys(): dictionary_str += "{0}\t{1}\n".format(id_term[w_id], param)

    with open(dictionary_path, 'w', encoding="utf-8") as out_file:
        out_file.write(dictionary_str)
    print("辞書出力完了")





if __name__ == '__main__':
    # レビュー文と単語idデータのパス
    review_path = "Data/Twitter/collected_texts0823_wakachi.txt"         # レビュー文
    term_index_path = "Data/term_index_0823.txt"            # 単語―id データ
    libsvm_data_path = "Data/svm_2_log10.fmt"  # libsvmフォーマットのデータ
    model_path = "Data/svm_2_log10.fmt.model"          # 機械学習モデル
    dictionary_path = "Data/polarity_2_log10.txt"



    term_indexer(review_path, term_index_path, False)
    # svm_indexer(review_path, term_index_path, libsvm_data_path)
    # weight_checker(term_index_path, model_path, dictionary_path)


sys.exit()
