#  中英文关键词抽取
欢迎使用关键词抽取，关键词抽取支持多种关键词抽取算法，支持一键调用进行关键词抽取




## 介绍

关键词抽取支持多种算法，算法如下：
- [1.TF-IDF](#1TF-IDF)
- [2.其他](#2其他)




---
## API



### 1.TF-IDF


```python
from app.key_extract import KeywordExtract

    
input_list = ["this is a sample",
              "this is another example example example",
              "this is a different example example"]
key_extract = KeywordExtract()

print(key_extract.infer(input_list))



## 路线

* [X] 支持TF-IDF关键词抽取算法
* [] 支持BM25关键词抽取算法
* [] 其他

