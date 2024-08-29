import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from keyword_extract.config import LDAConfig
from keyword_extract.base.base_keyword_extract import BaseKeyWordExtract
from keyword_extract.utils import stop_load


class LDA(BaseKeyWordExtract):
    def __init__(self):
        pass

    def preprocess_texts(self, documents):
        """
        数据处理
        :param documents:
        :return:
        """

        stopwords = set(stop_load())
        # 分词
        texts = [" ".join(jieba.lcut(doc)) for doc in documents]
        # 去除停用词
        texts = [" ".join([word for word in doc.split() if word not in stopwords]) for doc in texts]

        return texts

    def build_bow_model(self, texts):
        """
        向量的转化
        :param texts:
        :return:
        """
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        return X, vectorizer

    def train_lda(self, X, n_topics):
        """
        LDA的模型训练
        :param X:
        :param n_topics:
        :return:
        """

        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda_model.fit(X)
        return lda_model

    def infer(self, input_text, topic_num=LDAConfig.TOP_NUM):
        """

        :param input_text:
        :param topic_num:
        :return:
        """

        texts = self.preprocess_texts(input_text)
        X_, vectorizer_ = self.build_bow_model(texts)
        lda_model_infer = self.train_lda(X_, topic_num)
        feature_names = vectorizer_.get_feature_names_out()
        top_list = []
        for topic_idx, topic in enumerate(lda_model_infer.components_):
            top_list.append([feature_names[i] for i in topic.argsort()])
        return top_list


if __name__ == "__main__":
    document_str = [
        "机器学习是人工智能的一个分支", "人工智能的一个分支"
    ]
    lda_model = LDA()
    print(lda_model.infer(document_str))
