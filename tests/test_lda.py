from keyword_extract.lda_model.lda import LDA

if __name__ == '__main__':
    input_list = ["自然语言处理是人工智能领域中的一个重要方向。它研究人与计算机之间如何使用自然语言进行有效沟通。"]
    lda_model = LDA()
    # 基于LDA 进行关键词的抽取,topic_num是主题的个数
    print(lda_model.infer(input_list, topic_num=3))
