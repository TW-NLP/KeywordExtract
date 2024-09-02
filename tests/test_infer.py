from keyword_extract import KeywordExtract

input_tf_idf = [
    "自然语言处理是人工智能领域中的一个重要方向。", '人工智能领域中的一个重要方向。'
]
key_extract = KeywordExtract(type="TF-IDF")
print(key_extract.infer(input_tf_idf))
