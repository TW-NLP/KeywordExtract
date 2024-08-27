"""KeywordExtract infer 的高层API"""

from typing import Literal

from keyword_extract.graph_model.text_rank import TextRank
from keyword_extract.keybert_model.key_bert import KeyBERT
from keyword_extract.statistical_model.tf_idf import TFIDF


class KeywordExtract:
    def __init__(self, type: Literal["TF-IDF", "TextRank"] = "TF-IDF"):
        self.keyword_extract = None

        if type == "TF-IDF":
            self.keyword_extract = TFIDF()
        elif type == "TextRank":
            self.keyword_extract = TextRank()
        elif type == "":
            self.keyword_extract = KeyBERT()

    def infer(self, infer_list: list):
        """
        :param infer_list:
        :param type:
        :return:
        """
        result = self.keyword_extract.infer(infer_list)
        return result
