》》 Multi-Document Summarization of Evaluative Text
#GENERAL
- [PDF](http://www.anthology.aclweb.org/E/E06/E06-1039.pdf)
- [SLIDE ppt](https://www.google.co.jp/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwiNxNrYsLPXAhWIe7wKHfJ5CwgQFghAMAI&url=https%3A%2F%2Fwww.cs.ubc.ca%2F%7Erng%2Ftalkdepository%2Feacl2006.ppt&usg=AOvVaw1razPu9cN3_tzoV2VrQZvt)
- EACL 2006 ([Rank A in NLP conference](http://www.conferenceranks.com/#data))
- [182 citations](https://scholar.google.com/citations?user=HNNL22kAAAAJ&hl=ja)
- University of British Columbia Vancouver, Canada
- First Author: [**Giuseppe Carenini**](https://scholar.google.com/citations?user=HNNL22kAAAAJ&hl=ja), Associate Professor of Computer Science, University of British Columbia
- **Work series on same topic of Giuseppe Carenini**
    - [》》](https://qiita.com/badrabbit/items/6eeee8a5cc34043f5b65)[MULTI-DOCUMENT SUMMARIZATION OF EVALUATIVE TEXT, journal version](https://10.1111/j.1467-8640.2012.00417.x), Journal of Computational Intelligence 2013
    - [》》](https://qiita.com/badrabbit/items/66dc93b6a0fc7bc20735)[Extractive vs. NLG-based abstractive summarization of evaluative text: The effect of corpus controversiality](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.186.3640&rep=rep1&type=pdf), INLG 2008 ([rank B/B2](http://www.conferenceranks.com/#data)) 
    - [》》](https://qiita.com/badrabbit/items/22b3d5a271aa8c3b0cb9)[Multi-Document Summarization of Evaluative Text](#), EACL 2006 (rank A/A2)
    - [》》](https://qiita.com/badrabbit/items/f96c684d33b5064c1221)[Extracting knowledge from evaluative text](https://www.cs.ubc.ca/~rng/psdepository/kcap2005.pdf), K-Cap 2005 (rank A/B1), 196 citations

- Related papers from other authors
    - [Modeling content and structure for abstractive review
summarization](https://10.1016/j.csl.2016.06.005), Journal of Computer Speech and Language 2016
    - [Abstractive Summarization of Product Reviews Using Discourse Structure](http://emnlp2014.org/papers/pdf/EMNLP2014168.pdf), EMNLP 2014, **Giuseppe as second author**
    - [》》](https://qiita.com/badrabbit/private/5e95da6e90997f45952e)[Building a Sentiment Summarizer
for Local Service Reviews](https://s3.amazonaws.com/academia.edu.documents/11179325/paper3.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1510335318&Signature=hgqYBjoSY0Hy16q%2Fgi%2B4cVd%2Frms%3D&response-content-disposition=inline%3B%20filename%3DBuilding_a_sentiment_summarizer_for_loca.pdf), WWW-2008 workshop on NLP in the Information Explosion Era, Google Inc.

#ABSTRACT & CONCLUSION
- compare two approaches of summarization
    - sentence extraction based approach [**MEAD**](http://www.summarization.com/mead/) (an open source package)
    - language generation based approach **SEA**
- conclusion
    - both perform equally well quantitatively
        - **MEAD**: **varied language** and **details** but **lack in accuracy**, **fail to give an overview**
        - **SEA**: provide **general overview** but **sounding 'robotic'**, **repetitive** and **incoherent** (rời rạc, ko mạch lạc)
    - both perform different but for complementary reasons
    - should synthesize two approaches

#INTRODUCTION
- **INDUSTRIAL NEEDS**
    - Online customer reviews, summaries of this literature could be of great strategic value to product designers, planners and manufactures
    - Other important commercial applications such as **summarization of travel logs**
    - non-commercial applications such as the **summarization of candidate reviews**

- **PROBLEM**
    - how to effectively summarize a large corpora of evaluative text about a single entity e.g. a product
    - for factual documents, the goal is to **extract important facts** and present them in a **sensible ordering** while **avoiding repetition**
    - when documents contain **inconsistent info** e.g. conflicting report, the goal is to identify overlaps and inconsistencies and produce a summary that points out and explain those inconsistencies.
