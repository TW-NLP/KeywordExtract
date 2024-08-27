from keyword_extract.graph_model.text_rank import TextRank

if __name__ == '__main__':
    input_list = ["自然语言处理是人工智能领域中的一个重要方向。它研究人与计算机之间如何使用自然语言进行有效沟通。"]
    text_rank = TextRank()
    # 基于TextRank 进行关键词的抽取
    print(text_rank.infer(input_list))
