from keyword_extract import KeywordExtract


input_list = [
    "自然语言处理是人工智能领域中的一个重要方向。它研究人与计算机之间如何使用自然语言进行有效沟通。"
]
key_extract = KeywordExtract(type="TF-IDF")
# 基于TF-IDF进行关键词的抽取
print(key_extract.infer(input_list))


key_extract = KeywordExtract(type="TextRank")
# 基于TF-IDF进行关键词的抽取
print(key_extract.infer(input_list))
