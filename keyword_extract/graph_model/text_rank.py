import jieba
import networkx as nx
from itertools import combinations

from keyword_extract.utils import stop_load, TextRankConfig
from keyword_extract.base.base_keyword_extract import BaseKeyWordExtract


class TextRank(BaseKeyWordExtract):
    def preprocess_text(self, text):
        # 使用 jieba 分词
        words = jieba.lcut(text)
        # 去除停用词和无意义的标点符号
        stopwords = set(stop_load())
        candidate_keywords = [word for word in words if word not in stopwords and len(word) > 1]
        if candidate_keywords == []:
            # 采用2-gram 进行词的分割
            for i in range(len(words) - 1):
                candidate_keywords.append(''.join(words[i:i + 2]))
        return candidate_keywords

    def build_word_graph(self, words, window_size=TextRankConfig.WINDOW_SIZE):
        # 构建词语共现图
        graph = nx.Graph()
        word_pairs = list(combinations(range(len(words)), 2))

        for i, j in word_pairs:
            if abs(i - j) <= window_size:
                w1, w2 = words[i], words[j]
                if graph.has_edge(w1, w2):
                    graph[w1][w2]["weight"] += 1.0
                else:
                    graph.add_edge(w1, w2, weight=1.0)

        return graph

    def infer(self, text_list):
        """

        :param text_list:
        :return:
        """
        result = []
        for text_i in text_list:
            # 文本预处理
            words = self.preprocess_text(text_i)

            # 构建词语共现图
            graph = self.build_word_graph(words)

            # 使用 PageRank 计算节点权重
            pagerank = nx.pagerank(graph, weight="weight")

            # 根据权重排序
            sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            result.append(sorted_pagerank)

        return result


if __name__ == "__main__":
    # 示例文本
    text = [
        "有暖女吗？"
    ]
    text_rank = TextRank()
    # 提取关键词及其权重
    keywords_with_weights = text_rank.infer(text)

    print("提取的关键词及其权重：", keywords_with_weights)
