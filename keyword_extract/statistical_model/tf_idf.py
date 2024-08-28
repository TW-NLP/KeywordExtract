from collections import Counter
import math
import jieba

from keyword_extract.utils import stop_load
from keyword_extract.base.base_keyword_extract import BaseKeyWordExtract


class TFIDF(BaseKeyWordExtract):
    """
    TF-IDF算法
    """

    def __init__(self):
        pass

    def data_split(self, input_data):
        """
        分词 然后计算词频
        :param input_data:
        :return:
        """
        split_list = []
        counter_list = []
        stopwords = set(stop_load())
        for data_i in input_data:
            words = jieba.lcut(data_i)

            candidate_keywords = [word for word in words if word not in stopwords and len(word) > 1]
            if candidate_keywords == []:
                # 采用2-gram 进行词的分割
                for i in range(len(words) - 1):
                    candidate_keywords.append(''.join(words[i:i + 2]))
            split_list.append(candidate_keywords)
            counter_list.append(Counter(candidate_keywords))

        return split_list, counter_list

    def compute_tfidf(self, tf_dict, idf_dict):
        """

        :param tf_dict:
        :param idf_dict:
        :return:
        """
        tf_idf_dict = {}
        for word, tf_value in tf_dict.items():
            tf_idf_dict[word] = tf_value * idf_dict[word]
        # 根据tf-idf值的大小进行排序
        tf_idf_dict = sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True)
        return tf_idf_dict

    def compute_tf(self, word_dict, doc_words):
        """

        :param word_dict: 字符的统计个数
        :param doc_words: 文档中的字符集合
        :return:
        """
        tf_dict = {}
        words_len = len(doc_words)
        for word_i, count_i in word_dict.items():
            tf_dict[word_i] = count_i / words_len
        return tf_dict

    def compute_idf(self, doc_list):
        """

        :param doc_list: 文档的集合
        :return:
        """

        sum_list = list(set([word_i for doc_i in doc_list for word_i in doc_i]))

        idf_dict = {word_i: 0 for word_i in sum_list}

        for word_j in sum_list:
            for doc_j in doc_list:
                if word_j in doc_j:
                    idf_dict[word_j] += 1
        return {k: math.log(len(doc_list) / (v + 1)) for k, v in idf_dict.items()}

    def infer(self, input_list):
        """

        :param input_list: 用户输入的list,每个元素为一个文档
        :return:
        """
        doc_words, word_dict = self.data_split(input_list)

        tf_list = []
        for word_i, doc_j in zip(word_dict, doc_words):
            tf_list.append(self.compute_tf(word_i, doc_j))
        # 计算整个文档集合的IDF
        idf = self.compute_idf(doc_words)

        result_list = []
        for tf_i in tf_list:
            result_list.append(self.compute_tfidf(tf_i, idf))
        return result_list


if __name__ == "__main__":
    # # 示例文档
    #
    input_list = ["有暖女吗？"]
    #
    tfidf = TFIDF()
    print(tfidf.infer(input_list))
