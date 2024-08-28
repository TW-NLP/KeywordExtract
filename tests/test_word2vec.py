from keyword_extract.word2vec_cluster.word2vec import Word2VecCluster

if __name__ == '__main__':
    input_list = ["自然语言处理是人工智能领域中的一个重要方向。它研究人与计算机之间如何使用自然语言进行有效沟通。"]
    word2vec_model = Word2VecCluster()
    # 基于TextRank 进行关键词的抽取
    print(word2vec_model.infer(input_list))
