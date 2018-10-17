# - coding: utf-8 -


# Step.1 単語のindex付け
#フォーマットは <星の数> _SEP_ <分かち書き済レビュー文>
def term_Indexer():
    term_index = {}
    index_num = 0
    with open("data/rakuten_reviews_wakati_test.txt", 'r', encoding="utf-8") as file:
        for l in file.readlines():
            if(l is None): break
            ents = l.split(" __ SEP __ ") #ents[0] 星の数、 ents[1] レビュー文
            if(len(ents) < 2): continue
            terms = ents[1] #分かち書き済レビュー文

            terms = terms.split(" ")    # termsは単語のlist
            # term_indexのkeyに単語が登録されていなければ、valueにindexを振っていく
            for term in terms:
                term_index.setdefault(term, index_num + 1)

    return term_index

# Step.2 フォーマット化
# 高次元データの場合、スパースで大規模なものになりやすく、この場合、Pythonなどのラッパー経由だと正しく処理できないことがあります。そのため、libsvm形式と呼ばれる形式に変換して扱います。
# 直接、バイナリに投入した方が早いので、以下の形式に変換します。
# 1 1:0.12 2:0.4 3:0.1 ...
# 0 2:0.59 4:0.1 5:0.01 ...

print(term_Indexer())
