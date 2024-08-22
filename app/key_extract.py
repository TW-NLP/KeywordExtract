from model.tf_idf import TFIDF


class KeywordExtract(object):
    def __init__(self):
        self.tf_idf = TFIDF()

    def infer(self, infer_list, type='TF-IDF'):
        if type == "TF-IDF":
            result = self.tf_idf.infer(infer_list)
            return result


