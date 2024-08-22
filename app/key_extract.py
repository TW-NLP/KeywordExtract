from model.tf_idf import TFIDF


class KeywordExtract(object):
    def __init__(self):
        self.tf_idf = TFIDF()

    def infer(self, infer_list, type='TF-IDF'):
        if type == "TF-IDF":
            result = self.tf_idf.infer(infer_list)
            return result


input_list = ["this is a sample",
              "this is another example example example",
              "this is a different example example"]
key_extract = KeywordExtract()

print(key_extract.infer(input_list))
