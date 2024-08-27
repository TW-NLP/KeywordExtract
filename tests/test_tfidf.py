from keyword_extract.statistical_model.tf_idf import TFIDF

if __name__ == '__main__':
    input_list = ["自然语言处理是人工智能领域中的一个重要方向。它研究人与计算机之间如何使用自然语言进行有效沟通。"]
    tfidf = TFIDF()
    # 基于TF-IDF进行关键词的抽取
    print(tfidf.infer(input_list))
    # 基于TextRank 进行关键词的抽取
    print(tfidf.infer(input_list))
