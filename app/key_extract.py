from model.graph_model.text_rank import TextRank
from model.statistical_model.tf_idf import TFIDF


class KeywordExtract(object):
    def __init__(self):
        self.tf_idf = TFIDF()
        self.text_rank = TextRank()

    def infer(self, infer_list, type='TF-IDF'):
        """

        :param infer_list:
        :param type:
        :return:
        """
        result = []
        if type == "TF-IDF":
            result = self.tf_idf.infer(infer_list)
        if type == "TextRank":
            result = self.text_rank.infer(infer_list)
        return result
