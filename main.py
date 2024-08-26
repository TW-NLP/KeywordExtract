from app.key_extract import KeywordExtract

if __name__ == '__main__':
    input_list = ["this is another example example example"]
    key_extract = KeywordExtract()
    # 基于TF-IDF进行关键词的抽取
    print(key_extract.infer(input_list))
    # 基于TextRank 进行关键词的抽取
    print(key_extract.infer(input_list, type="TextRank"))
