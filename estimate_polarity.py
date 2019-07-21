# - coding: utf-8 -
'''

感情（極性）推定


'''

import sys
import MeCab
import re


# テキストを分かち書きし、出力する
# セパレータは改行と仮定
#
# 職場の義理チョコです。 パッケージが落ち着いた感じなので 年配の方にもいいと思います。
# ⇒職場 の 義理チョコ です 。 パッケージ が 落ち着い た 感じ な ので 年配 の 方 に も いい と 思い ます 。
def wakachi(text_file, output):
    print("生のテキストを分かち書きして出力")
    m = MeCab.Tagger("-Owakati")    # parserの設定 分かち書き

    wakachi_texts = ""
    with open(text_file, 'r', encoding="utf-8") as f:
        texts = f.read()
    for text in texts.split('\n'):      # セパレータは改行
        wakachi_texts += m.parse(text)

    with open(output, 'w', encoding="utf-8") as f:
        f.write(wakachi_texts)
    print("完了")


# 基本的な感情(極性)推定
# テキスト中の単語の感情値の平均をとる
# 辞書はpredealingtools.pyで構築したものを使う
# 入力は分かち書き済みのテキスト
def estimate_simple(dic_path, w_path, output_path):
    print(" " + w_path + " 中のテキストの極性を算出")
    print("⇒ " + output_path + "  に出力")
    # 1. 単語-polarity 辞書をロード
    term_pol = {}    # 単語:感情値
    with open(dic_path, 'r', encoding="utf-8") as dic_f:
        term_polarity = dic_f.read()
    for i in term_polarity.split('\n'):
        term_pol_pair = i.split(" ")
        if len(term_pol_pair) < 2: continue
        term_pol.setdefault(term_pol_pair[0], float(term_pol_pair[1]))

    # 2. 辞書を使ってテキストの平均感情値を算出
    output = ""
    with open(w_path, 'r', encoding="utf-8") as f:
        texts = f.read()
    # テキストごと
    for text in texts.split('\n'):
        if text is "": continue
        value = 0.0     # テキストの感情(極性)値
        pol = []        # 単語の値リスト
        for term in text.split(' '):            # 単語１こずつ見ていく
            if term is "": continue
            if term not in term_pol.keys(): continue
            else: pol.append(term_pol[term])    # 極性値

        if len(pol) is 0: continue
        value = sum(pol) / len(pol)    # 単語数で感情値の平均をとる
        sumpol = sum(pol)
        output += "{0} {1}\n".format(str(round(sumpol, 2)), text)

    # 3. ファイル出力
    with open(output_path, 'w', encoding="utf-8") as out_file:
        out_file.write(output)
    print("出力完了")


# TODO 出現位置を考慮した極性推定
# 単語の出現位置を考慮した感情(極性)推定
# 辞書はpredealingtools.pyで構築したものを使う
# 入力は分かち書き済みのテキスト
def estimate_position(dic_path, w_path, output_path):
    m = MeCab.Tagger("Chasen")  # parserの設定 Chasen
    print(" " + w_path + " 中のテキストの極性を算出")
    print("⇒ " + output_path + "  に出力")

    # 1. 単語-polarity 辞書をロード
    term_pol = {}    # 単語:感情値
    with open(dic_path, 'r', encoding="utf-8") as dic_f:
        term_polarity = dic_f.read()
    for i in term_polarity.split('\n'):
        term_pol_pair = i.split(" ")
        if len(term_pol_pair) < 2: continue
        term_pol.setdefault(term_pol_pair[0], float(term_pol_pair[1]))


    # 2. 辞書を使ってテキストの感情値を算出
    output = ""
    with open(w_path, 'r', encoding="utf-8") as f: texts = f.read()
    # テキスト毎
    for text in texts.split('\n'):
        if text is "": continue
        if text == "Xperia 感度 良 すぎ だ な ":
            print("a")

        # 2-1. テキストからすべての単語(記号以外)と品詞情報を取り出す
        kaiseki = re.findall(".+\t.{1,3}詞", m.parse(text))   # 解析結果から「単語<tab>〇〇詞」だけ取り出す
        k_term = []     # 単語
        k_cate = []     # 品詞
        k_posi = []     # 出現位置
        ct = 0
        for i in kaiseki:
            ct += 1
            pair = i.split('\t')
            k_term.append(pair[0])
            k_cate.append(pair[1])
            k_posi.append(ct)

        # 2-2. 極性値計算
        # 単語―極性値辞書にある、且つ極性値非0の単語について
        #

        value = 0.0     # テキストの極性値
        pol = []        # 単語の極性値
        a = 0  # 出現位置パラメータ
        length = len(k_term)  # text中の総単語数

        for search_term in text.split(' '):
            if search_term is "": continue
            if search_term not in k_term: continue              # 解析結果にない単語
            if search_term not in term_pol.keys(): continue     # 辞書にない単語
            if term_pol[search_term] is 0.0: continue           # TODO 極性値0の単語への対応

            # 極性値0以外の単語が解析結果に在るとき
            id = k_term.index(search_term)     # インデックス取得

            if re.match("助詞", k_cate[id]):       # 助詞を除外
                k_term.pop(id)
                k_cate.pop(id)
                k_posi.pop(id)
                continue

            # beta_p = term_pol[search_term] ** k_posi[id]
            # beta_lp = term_pol[search_term] ** (length - k_posi[id])
            beta_p = term_pol[search_term] * k_posi[id]
            beta_lp = term_pol[search_term] * (length - k_posi[id])

            # 辞書極性値が負かつ偶数乗なら符号を反転
            # if term_pol[search_term] < 0:
            #     if k_posi[id] % 2 == 0: beta_p = -beta_p
            #     if (length - k_posi[id]) % 2 == 0: beta_lp = -beta_lp

            # 単語の重み付き極性値を算出
            c = a * beta_p + (1 - a) * beta_lp   # 重み付き極性値
            pol.append(c)

            # 重複単語への対応
            k_term.pop(id)
            k_cate.pop(id)
            k_posi.pop(id)

            st = "あ"

        if len(pol) is 0:
            output += "\n"
            continue
        mean = sum(pol) / len(pol)    # 感情値の平均
        sumpol = sum(pol)              # 感情値の合計
        output += "{0} {1}\n".format(str(round(mean, 2)), text)

    # 3. ファイル出力
    with open(output_path, 'w', encoding="utf-8") as out_file:
        out_file.write(output)
    print("出力完了")


def wakachi_test():
    m = MeCab.Tagger("Chasen")    # parserの設定 分かち書き
    text = "# Xperia Z 2 に し て から 、 スケジュール ストリート に 登録 する 時 に いちいち 自分 の Google アカウント を 選択 し て 登録 し なきゃ いけ なかっ た ん だ が 、 設定 を 弄っ たら アカウント を デフォ で 選択 できる 様 に なっ た 。 地味 に 嬉しい 。 "

    k_term = []
    k_cate = []

    # 形態素解析結果から不要な部分を削除
    kaiseki = re.sub(".+\t記号.+\n|EOS$", "", m.parse(text))         # 解析結果から記号の行とEOSを消去
    kaiseki = re.findall(".+\t.{1,3}詞", kaiseki)                    # 「単語<tab>〇〇詞」だけ抜き取ってリスト化
    for i in kaiseki:
        pair = i.split('\t')
        k_term.append(pair[0])
        k_cate.append(pair[1])

    kensaku = "ストリート"
    index = k_term.index(kensaku)
    print("{0} ＝ {1}".format(k_term[index], k_cate[index]))
    print(str(index+1) + "番目")
    print("終了")


if __name__ == '__main__':
    dictionary_path = "Data/polarity_0712_log10.txt"
    text_path = "Data/Twitter/collected_texts08.txt"      # 生のテキストファイル　改行でセパレート
    w_text_path = re.sub(".txt", "_wakachi.txt", text_path)
    # w_text_path = "Data/Twitter/rakuten_reviews_wakachi_fixed2_texts.txt"
    estimated_path = re.sub(".txt", "_estimated4.txt", w_text_path)


    #wakachi(text_path, w_text_path)
    #estimate_simple(dictionary_path, w_text_path, estimated_path)
    estimate_position(dictionary_path, w_text_path, estimated_path)

    #wakachi_test()


sys.exit()
