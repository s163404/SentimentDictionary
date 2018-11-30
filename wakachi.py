# - coding: utf-8 -
'''

生のレビュー文章を分かち書きした文章を出力する
セパレータは改行と仮定

職場の義理チョコです。 パッケージが落ち着いた感じなので 年配の方にもいいと思います。
⇒職場 の 義理チョコ です 。 パッケージ が 落ち着い た 感じ な ので 年配 の 方 に も いい と 思い ます 。

'''

import sys
import MeCab

review_path = ""        # レビューのパス　txt形式??
output_path = ""        # 出力する分かち書きレビューのパス

def wakachi():
    print("生のレビュー文を分かち書きして出力")
    m = MeCab.Tagger("-Owakati")    # parserの設定 分かち書き
    wakachi_reviews = ""            # 分かち書きしたレビュー

    with open(review_path, 'r', encoding="utf-8") as file:
        for text in file.readlines():
            if text is None: continue
            elif text.isnumeric(): continue
            wakachi_reviews += m.parse(text) + '\n'

        # reviews = file.read().split("\n")
    # for review in reviews:
    #     wakachi_reviews += m.parse(review) + '\n'

    with open(output_path, 'w', encoding="utf-8") as file:
        file.write(wakachi_reviews)


if __name__ == '__main__':

    wakachi()


sys.exit()
