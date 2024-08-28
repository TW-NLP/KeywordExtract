import jieba
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
import numpy as np

from keyword_extract.config import Word2vecConfig
from keyword_extract.base.base_keyword_extract import BaseKeyWordExtract
from keyword_extract.utils import stop_load


class Word2VecCluster(BaseKeyWordExtract):

    # 文本预处理函数
    def preprocess_text(self, text):
        """
        对输入的文本进行预处理：分句、分词、去除停用词。
        返回分词后的句子列表。
        """

        stopwords = set(stop_load())
        processed_sentences = []
        words = jieba.lcut(text)  # 使用jieba进行分词
        words = [word for word in words if word not in stopwords and word.strip()]  # 去除停用词和空白词
        processed_sentences.append(words)
        return processed_sentences

    def train_model(self, split_text):
        """

        :param split_text: 结巴分词后的list
        :return:
        """
        word2vec_model = Word2Vec(sentences=split_text, vector_size=Word2vecConfig.W2V_VECTOR_SIZE,
                                  window=Word2vecConfig.W2V_WINDOW, min_count=Word2vecConfig.W2V_MIN_COUNT)
        return word2vec_model

    def get_embedding(self, model, words):
        """

        :param model: word2vec model
        :param words: split text
        :return:
        """

        word_vectors = []
        word_list = []
        for word in words:
            if word in model.wv:  # 检查词是否在Word2Vec词汇表中
                word_vectors.append(model.wv[word])  # 添加词向量到列表
                word_list.append(word)  # 记录对应的词
        return word_list, np.array(word_vectors)  # 返回词列表和词向量矩阵

    def extract_keywords_by_dbscan(self, word_vectors, word_list):
        """
        使用DBSCAN对词向量进行聚类，从每个聚类中提取代表性关键词。
        :param word_vectors:
        :param word_list:
        :return:
        """
        dbscan = DBSCAN(eps=Word2vecConfig.DB_EPS, min_samples=Word2vecConfig.DB_MIN_SAMPLES,
                        metric=Word2vecConfig.DB_METRIC)  # 使用余弦距离进行DBSCAN聚类
        labels = dbscan.fit_predict(word_vectors)  # 对词向量进行聚类，并获取每个词的聚类标签

        unique_labels = set(labels)  # 获取唯一的聚类标签
        keywords = []
        for label in unique_labels:
            if label == -1:  # 忽略噪声点
                continue

            cluster_indices = np.where(labels == label)[0]  # 找到属于当前聚类的词语索引
            cluster_words = [word_list[index] for index in cluster_indices]  # 获取聚类中的词语列表
            cluster_vectors = word_vectors[cluster_indices]  # 获取聚类中的词向量

            # 计算每个词与聚类中心的平均向量的距离，选择距离最近的词作为关键词
            cluster_center = np.mean(cluster_vectors, axis=0)  # 计算聚类中心的平均向量
            distances = np.linalg.norm(cluster_vectors - cluster_center, axis=1)  # 计算每个词到聚类中心的距离
            closest_index = np.argmin(distances)  # 找到距离最近的词语索引
            keywords.append(cluster_words[closest_index])  # 添加关键词

        return keywords

    def infer(self, input_text):
        result = []
        for test_i in input_text:
            # 文本预处理
            processed_sentences = self.preprocess_text(test_i)
            model = self.train_model(processed_sentences)
            word_list, word_vectors = self.get_embedding(model, processed_sentences[0])
            keywords = self.extract_keywords_by_dbscan(word_vectors, word_list)
            result.append(keywords)
        return result


# 示例文本
if __name__ == '__main__':
    text = ["人工智能是计算机科学的一个分支，它试图理解智能的本质，并生产出一种新的能以人类智能相似方式做出反应的智能机器。"]

    w2v_model = Word2VecCluster()
    print(w2v_model.infer(text))
