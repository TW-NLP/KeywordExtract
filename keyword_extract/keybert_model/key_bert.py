import jieba
from transformers import BertTokenizer, BertModel
import numpy as np

from keyword_extract.base.base_keyword_extract import BaseKeyWordExtract
from keyword_extract.config import DEVICE, KeyBERTConfig
from keyword_extract.utils import stop_load


class KeyBERT(BaseKeyWordExtract):
    def __init__(self):
        # 1. 加载 BERT 模型和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(KeyBERTConfig.BERT_MODEL
                                                       )
        self.model = BertModel.from_pretrained(
            KeyBERTConfig.BERT_MODEL)
        self.model.to(DEVICE)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(DEVICE)
        outputs = self.model(**inputs)
        # 使用 [CLS] token 的向量作为文本的表示
        return outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()[0]

    def cos_cal(self, x, y):
        """
        进行余弦相似度的计算
        :param x: 文本的嵌入向量
        :param y: 候选关键词的嵌入向量
        :return: 余弦相似度数组
        """
        # 计算向量 x 的范数（长度）
        x_norm = np.linalg.norm(x)

        # 计算 y 中每个向量的范数
        y_norms = np.linalg.norm(y, axis=1)

        # 计算点积
        dot_products = np.dot(y, x.T)

        # 计算余弦相似度
        cosine_similarities = dot_products / (y_norms * x_norm)

        return cosine_similarities

    def sim_cal(self, text_embed, keyword_embed):
        """
        计算文本嵌入与关键词嵌入之间的相似度
        :param text_embed: 文本嵌入向量
        :param keyword_embed: 候选关键词嵌入向量
        :return: 相似度数组
        """
        return self.cos_cal(text_embed, keyword_embed)

    def infer(self, input_text):
        result_list = []
        for text_i in input_text:
            words = jieba.lcut(text_i)
            # 去除停用词和无意义的标点符号
            stopwords = set(stop_load())
            candidate_keywords = [word for word in words if word not in stopwords and len(word) > 1]

            # 生成嵌入向量
            text_embedding = self.get_embeddings(text_i)
            keyword_embeddings = np.array([self.get_embeddings(keyword) for keyword in candidate_keywords])

            # 计算相似度
            similarities = self.sim_cal(text_embedding, keyword_embeddings)

            # 排序
            top_keywords = [(candidate_keywords[i], similarities[i]) for i in similarities.argsort()[::-1]]
            result_list.append(top_keywords)

        return result_list


if __name__ == '__main__':
    key_bert = KeyBERT()
    key_bert.infer(['好好学习'])
