from app.key_extract import KeywordExtract

if __name__ == '__main__':

    input_list = ["this is a sample",
                  "this is another example example example",
                  "this is a different example example"]
    key_extract = KeywordExtract()

    print(key_extract.infer(input_list))
