# BERT-related Papers
a list of BERT-related papers. 

Any feedback is welcome.

## Table of Contents
- [Downstream task](#downstream-task)
- [Generation](#generation)
- [Modification (multi-task, masking strategy, etc.)](#modification-multi-task-masking-strategy-etc)
- [Probe](#probe)
- [Inside BERT](#inside-bert)
- [Multi-lingual](#multi-lingual)
- [Other than English models](#other-than-english-models)
- [Domain specific](#domain-specific)
- [Multi-modal](#multi-modal)
- [Model compression](#model-compression)
- [Misc.](#misc)

## Downstream task
### QA, MC, Dialogue
- [A BERT Baseline for the Natural Questions](https://arxiv.org/abs/1901.08634)
- [MultiQA: An Empirical Investigation of Generalization and Transfer in Reading Comprehension](https://arxiv.org/abs/1905.13453) (ACL2019)
- [Unsupervised Domain Adaptation on Reading Comprehension](https://arxiv.org/abs/1911.06137)
- [BERTQA -- Attention on Steroids](https://arxiv.org/abs/1912.10435)
- [A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning](https://arxiv.org/abs/1908.05514) (EMNLP2019)
- [SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering](https://arxiv.org/abs/1812.03593)
- [Multi-hop Question Answering via Reasoning Chains](https://arxiv.org/abs/1910.02610)
- [Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://arxiv.org/abs/1911.00484)
- [Multi-step Entity-centric Information Retrieval for Multi-Hop Question Answering](https://arxiv.org/abs/1909.07598) (EMNLP2019 WS)
- [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718) (NAALC2019)
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300) (ACL2019)
- [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/abs/1908.08167) (EMNLP2019)
- [Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470) (ICLR2020)
- [Learning to Ask Unanswerable Questions for Machine Reading Comprehension](https://arxiv.org/abs/1906.06045) (ACL2019)
- [Unsupervised Question Answering by Cloze Translation](https://arxiv.org/abs/1906.04980) (ACL2019)
- [Reinforcement Learning Based Graph-to-Sequence Model for Natural Question Generation](https://arxiv.org/abs/1908.04942)
- [A Recurrent BERT-based Model for Question Generation](https://www.aclweb.org/anthology/D19-5821/) (EMNLP2019 WS)
- [Learning to Answer by Learning to Ask: Getting the Best of GPT-2 and BERT Worlds](https://arxiv.org/abs/1911.02365)
- [Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension](https://www.aclweb.org/anthology/papers/P/P19/P19-1226/) (ACL2019)
- [Incorporating Relation Knowledge into Commonsense Reading Comprehension with Multi-task Learning](https://arxiv.org/abs/1908.04530) (CIKM2019)
- [SG-Net: Syntax-Guided Machine Reading Comprehension](https://arxiv.org/abs/1908.05147)
- [MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension](https://arxiv.org/abs/1910.00458)
- [Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning](https://arxiv.org/abs/1909.00277) (EMNLP2019)
- [ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning](https://arxiv.org/abs/2002.04326) (ICLR2020)
- [Robust Reading Comprehension with Linguistic Constraints via Posterior Regularization](https://arxiv.org/abs/1911.06948)
- [BAS: An Answer Selection Method Using BERT Language Model](https://arxiv.org/abs/1911.01528)
- [Beat the AI: Investigating Adversarial Human Annotations for Reading Comprehension](https://arxiv.org/abs/2002.00293)
- [A Simple but Effective Method to Incorporate Multi-turn Context with BERT for Conversational Machine Comprehension](https://arxiv.org/abs/1905.12848) (ACL2019 WS)
- [FlowDelta: Modeling Flow Information Gain in Reasoning for Conversational Machine Comprehension](https://arxiv.org/abs/1908.05117) (ACL2019 WS)
- [BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/abs/1905.05412) (SIGIR2019)
- [GraphFlow: Exploiting Conversation Flow with Graph Neural Networks for Conversational Machine Comprehension](https://arxiv.org/abs/1908.00059) (ICML2019 WS)
- [Beyond English-only Reading Comprehension: Experiments in Zero-Shot Multilingual Transfer for Bulgarian](https://arxiv.org/abs/1908.01519) (RANLP2019)
- [XQA: A Cross-lingual Open-domain Question Answering Dataset](https://www.aclweb.org/anthology/P19-1227/) (ACL2019)
- [Cross-Lingual Machine Reading Comprehension](https://arxiv.org/abs/1909.00361) (EMNLP2019)
- [Zero-shot Reading Comprehension by Cross-lingual Transfer Learning with Multi-lingual Language Representation Model](https://arxiv.org/abs/1909.09587)
- [Multilingual Question Answering from Formatted Text applied to Conversational Agents](https://arxiv.org/abs/1910.04659)
- [BiPaR: A Bilingual Parallel Dataset for Multilingual and Cross-lingual Reading Comprehension on Novels](https://arxiv.org/abs/1910.05040) (EMNLP2019)
- [MLQA: Evaluating Cross-lingual Extractive Question Answering](https://arxiv.org/abs/1910.07475)
- [Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension](https://arxiv.org/abs/1904.09679) (TACL)
- [SberQuAD - Russian Reading Comprehension Dataset: Description and Analysis](https://arxiv.org/abs/1912.09723)
- [Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension](https://arxiv.org/abs/1909.00109) (EMNLP2019)
- [BERT-DST: Scalable End-to-End Dialogue State Tracking with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1907.03040) (Interspeech2019)
- [Dialog State Tracking: A Neural Reading Comprehension Approach](https://arxiv.org/abs/1908.01946) 
- [A Simple but Effective BERT Model for Dialog State Tracking on Resource-Limited Systems](https://arxiv.org/abs/1910.12995)
- [Fine-Tuning BERT for Schema-Guided Zero-Shot Dialogue State Tracking](https://arxiv.org/abs/2002.00181)
- [Goal-Oriented Multi-Task BERT-Based Dialogue State Tracker](https://arxiv.org/abs/2002.02450)
- [Domain Adaptive Training BERT for Response Selection](https://arxiv.org/abs/1908.04812)
- [BERT Goes to Law School: Quantifying the Competitive Advantage of Access to Large Legal Corpora in Contract Understanding](https://arxiv.org/abs/1911.00473)

### Slot filling
- [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
- [Multi-lingual Intent Detection and Slot Filling in a Joint BERT-based Model](https://arxiv.org/abs/1907.02884)
- [A Comparison of Deep Learning Methods for Language Understanding](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1262.html) (Interspeech2019)

### Analysis
- [Fine-grained Information Status Classification Using Discourse Context-Aware Self-Attention](https://arxiv.org/abs/1908.04755)
- [GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge](https://arxiv.org/abs/1908.07245) (EMNLP2019)
- [Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations](https://arxiv.org/abs/1910.00194)  (EMNLP2019)
- [Using BERT for Word Sense Disambiguation](https://arxiv.org/abs/1909.08358)
- [Language Modelling Makes Sense: Propagating Representations through WordNet for Full-Coverage Word Sense Disambiguation](https://www.aclweb.org/anthology/P19-1569.pdf) (ACL2019)
- [Neural Aspect and Opinion Term Extraction with Mined Rules as Weak Supervision](https://arxiv.org/abs/1907.03750) (ACL2019) 
- [Assessing BERT’s Syntactic Abilities](https://arxiv.org/abs/1901.05287)
- [Does BERT agree? Evaluating knowledge of structure dependence through agreement relations](https://arxiv.org/abs/1908.09892)
- [Simple BERT Models for Relation Extraction and Semantic Role Labeling](https://arxiv.org/abs/1904.05255)
- [LIMIT-BERT : Linguistic Informed Multi-Task BERT](https://arxiv.org/abs/1910.14296)
- [A Simple BERT-Based Approach for Lexical Simplification](https://arxiv.org/abs/1907.06226)
- [Multi-headed Architecture Based on BERT for Grammatical Errors Correction](https://www.aclweb.org/anthology/papers/W/W19/W19-4426/) (ACL2019 WS) 
- [Towards Minimal Supervision BERT-based Grammar Error Correction](https://arxiv.org/abs/2001.03521)
- [BERT-Based Arabic Social Media Author Profiling](https://arxiv.org/abs/1909.04181)
- [Sentence-Level BERT and Multi-Task Learning of Age and Gender in Social Media](https://arxiv.org/abs/1911.00637)
- [Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)
- [NegBERT: A Transfer Learning Approach for Negation Detection and Scope Resolution](https://arxiv.org/abs/1911.04211)
- [xSLUE: A Benchmark and Analysis Platform for Cross-Style Language Understanding and Evaluation](https://arxiv.org/abs/1911.03663)
- [TabFact: A Large-scale Dataset for Table-based Fact Verification](https://arxiv.org/abs/1909.02164)
- [Rapid Adaptation of BERT for Information Extraction on Domain-Specific Business Documents](https://arxiv.org/abs/2002.01861)
- [LAMBERT: Layout-Aware language Modeling using BERT for information extraction](https://arxiv.org/abs/2002.08087)

### Word segmentation, parsing, NER
- [BERT Meets Chinese Word Segmentation](https://arxiv.org/abs/1909.09292)
- [Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning](https://arxiv.org/abs/1903.04190)
- [Establishing Strong Baselines for the New Decade: Sequence Tagging, Syntactic and Semantic Parsing with BERT](https://arxiv.org/abs/1908.04943)
- [Evaluating Contextualized Embeddings on 54 Languages in POS Tagging, Lemmatization and Dependency Parsing](https://arxiv.org/abs/1908.07448) 
- [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204)
- [Deep Contextualized Word Embeddings in Transition-Based and Graph-Based Dependency Parsing -- A Tale of Two Parsers Revisited](https://arxiv.org/abs/1908.07397) (EMNLP2019)
- [Parsing as Pretraining](https://arxiv.org/abs/2002.01685) (AAAI2020)
- [Cross-Lingual BERT Transformation for Zero-Shot Dependency Parsing](https://arxiv.org/abs/1909.06775)
- [Named Entity Recognition -- Is there a glass ceiling?](https://arxiv.org/abs/1910.02403) (CoNLL2019)
- [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)
- [Training Compact Models for Low Resource Entity Tagging using Pre-trained Language Models](https://arxiv.org/abs/1910.06294)
- [Robust Named Entity Recognition with Truecasing Pretraining](https://arxiv.org/abs/1912.07095) (AAAI2020)
- [LTP: A New Active Learning Strategy for Bert-CRF Based Named Entity Recognition](https://arxiv.org/abs/2001.02524)
- [MT-BioNER: Multi-task Learning for Biomedical Named Entity Recognition using Deep Bidirectional Transformers](https://arxiv.org/abs/2001.08904)
- [Portuguese Named Entity Recognition using BERT-CRF](https://arxiv.org/abs/1909.10649)
- [Towards Lingua Franca Named Entity Recognition with BERT](https://arxiv.org/abs/1912.01389)

### Pronoun/coreference resolution
- [Resolving Gendered Ambiguous Pronouns with BERT](https://arxiv.org/abs/1906.01161) (ACL2019 WS)
- [Anonymized BERT: An Augmentation Approach to the Gendered Pronoun Resolution Challenge](https://arxiv.org/abs/1905.01780) (ACL2019 WS)
- [Gendered Pronoun Resolution using BERT and an extractive question answering formulation](https://arxiv.org/abs/1906.03695) (ACL2019 WS)
- [MSnet: A BERT-based Network for Gendered Pronoun Resolution](https://arxiv.org/abs/1908.00308) (ACL2019 WS)
- [Fill the GAP: Exploiting BERT for Pronoun Resolution](https://www.aclweb.org/anthology/papers/W/W19/W19-3815/) (ACL2019 WS)
- [On GAP Coreference Resolution Shared Task: Insights from the 3rd Place Solution](https://www.aclweb.org/anthology/W19-3816/) (ACL2019 WS)
- [Look Again at the Syntax: Relational Graph Convolutional Network for Gendered Ambiguous Pronoun Resolution](https://arxiv.org/abs/1905.08868) (ACL2019 WS)
- [BERT Masked Language Modeling for Co-reference Resolution](https://www.aclweb.org/anthology/papers/W/W19/W19-3811/) (ACL2019 WS)
- [Coreference Resolution with Entity Equalization](https://www.aclweb.org/anthology/P19-1066/) (ACL2019)
- [BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091) (EMNLP2019) [[github](https://github.com/mandarjoshi90/coref)]
- [WikiCREM: A Large Unsupervised Corpus for Coreference Resolution](https://arxiv.org/abs/1908.08025) (EMNLP2019)
- [Ellipsis and Coreference Resolution as Question Answering](https://arxiv.org/abs/1908.11141)
- [Coreference Resolution as Query-based Span Prediction](https://arxiv.org/abs/1911.01746)

### Sentiment analysis
- [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588) (NAACL2019)
- [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://arxiv.org/abs/1904.02232) (NAACL2019)
- [Exploiting BERT for End-to-End Aspect-based Sentiment Analysis](https://arxiv.org/abs/1910.00883) (EMNLP2019 WS)
- [Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification](https://arxiv.org/abs/1908.11860) 
- [An Investigation of Transfer Learning-Based Sentiment Analysis in Japanese](https://arxiv.org/abs/1905.09642) (ACL2019)
- ["Mask and Infill" : Applying Masked Language Model to Sentiment Transfer](https://arxiv.org/abs/1908.08039)
- [Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/abs/2001.11316)
- [Utilizing BERT Intermediate Layers for Aspect Based Sentiment Analysis and Natural Language Inference](https://arxiv.org/abs/2002.04815)

### Relation extraction
- [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158) (ACL2019)
- [BERT-Based Multi-Head Selection for Joint Entity-Relation Extraction](https://arxiv.org/abs/1908.05908) (NLPCC2019)
- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)
- [Span-based Joint Entity and Relation Extraction with Transformer Pre-training](https://arxiv.org/abs/1909.07755)
- [Fine-tune Bert for DocRED with Two-step Process](https://arxiv.org/abs/1909.11898)
- [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://arxiv.org/abs/1909.03546) (EMNLP2019)
- [Fine-tuning BERT for Joint Entity and Relation Extraction in Chinese Medical Text](https://arxiv.org/abs/1908.07721)

### Knowledge base
- [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/abs/1909.03193)
- [Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066) (EMNLP2019) [[github](https://github.com/facebookresearch/LAMA)]
- [BERT is Not a Knowledge Base (Yet): Factual Knowledge vs. Name-Based Reasoning in Unsupervised QA](https://arxiv.org/abs/1911.03681)
- [Inducing Relational Knowledge from BERT](https://arxiv.org/abs/1911.12753) (AAAI2020)
- [Latent Relation Language Models](https://arxiv.org/abs/1908.07690) (AAAI2020)
- [Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model](https://openreview.net/forum?id=BJlzm64tDH) (ICLR2020)
- [Zero-shot Entity Linking with Dense Entity Retrieval](https://arxiv.org/abs/1911.03814)
- [Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking](https://www.aclweb.org/anthology/K19-1063/) (CoNLL2019)
- [Improving Entity Linking by Modeling Latent Entity Type Information](https://arxiv.org/abs/2001.01447) (AAAI2020)
- [How Can We Know What Language Models Know?](https://arxiv.org/abs/1911.12543)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://kentonl.com/pub/gltpc.2020.pdf)

### Text classification
- [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583)
- [X-BERT: eXtreme Multi-label Text Classification with BERT](https://arxiv.org/abs/1905.02331)
- [DocBERT: BERT for Document Classification](https://arxiv.org/abs/1904.08398)
- [Enriching BERT with Knowledge Graph Embeddings for Document Classification](https://arxiv.org/abs/1909.08402)
- [Classification and Clustering of Arguments with Contextualized Word Embeddings](https://arxiv.org/abs/1906.09821) (ACL2019)
- [BERT for Evidence Retrieval and Claim Verification](https://arxiv.org/abs/1910.02655)
- [Conditional BERT Contextual Augmentation](https://arxiv.org/abs/1812.06705)
- [Stacked DeBERT: All Attention in Incomplete Data for Text Classification](https://arxiv.org/abs/2001.00137)

### WSC, WNLI, NLI
- [Exploring Unsupervised Pretraining and Sentence Structure Modelling for Winograd Schema Challenge](https://arxiv.org/abs/1904.09705)
- [A Surprisingly Robust Trick for the Winograd Schema Challenge](https://arxiv.org/abs/1905.06290)
- [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641) (AAAI2020)
- [Improving Natural Language Inference with a Pretrained Parser](https://arxiv.org/abs/1909.08217)
- [Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599)
- [Adversarial Analysis of Natural Language Inference Systems](https://arxiv.org/abs/1912.03441) (ICSC2020)
- [Evaluating BERT for natural language inference: A case study on the CommitmentBank](https://www.aclweb.org/anthology/D19-1630/) (EMNLP2019)

### Commonsense
- [CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937) (NAACL2019)
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830) (ACL2019) [[website](https://rowanzellers.com/hellaswag/)]
- [Story Ending Prediction by Transferable BERT](https://arxiv.org/abs/1905.07504) (IJCAI2019)
- [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/abs/1906.02361) (ACL2019)
- [Align, Mask and Select: A Simple Method for Incorporating Commonsense Knowledge into Language Representation Models](https://arxiv.org/abs/1908.06725)
- [Informing Unsupervised Pretraining with External Linguistic Knowledge](https://arxiv.org/abs/1909.02339)
- [Commonsense Knowledge + BERT for Level 2 Reading Comprehension Ability Test](https://arxiv.org/abs/1909.03415)
- [BIG MOOD: Relating Transformers to Explicit Commonsense Knowledge](https://arxiv.org/abs/1910.07713)
- [Commonsense Knowledge Mining from Pretrained Models](https://arxiv.org/abs/1909.00505) (EMNLP2019)
- [Do Massively Pretrained Language Models Make Better Storytellers?](https://arxiv.org/abs/1909.10705) (CoNLL2019)
- [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641v1) (AAAI2020)
- [Why Do Masked Neural Language Models Still Need Common Sense Knowledge?](https://arxiv.org/abs/1911.03024)

### Extractive summarization
- [HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/abs/1905.06566) (ACL2019)
- [Deleter: Leveraging BERT to Perform Unsupervised Successive Text Compression](https://arxiv.org/abs/1909.03223)
- [Discourse-Aware Neural Extractive Model for Text Summarization](https://arxiv.org/abs/1910.14142)

### IR
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [Investigating the Successes and Failures of BERT for Passage Re-Ranking](https://arxiv.org/abs/1905.01758)
- [Understanding the Behaviors of BERT in Ranking](https://arxiv.org/abs/1904.07531)
- [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375)
- [CEDR: Contextualized Embeddings for Document Ranking](https://arxiv.org/abs/1904.07094) (SIGIR2019)
- [Deeper Text Understanding for IR with Contextual Neural Language Modeling](https://arxiv.org/abs/1905.09217) (SIGIR2019)
- [FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance](https://arxiv.org/abs/1905.02851) (SIGIR2019)
- [Multi-Stage Document Ranking with BERT](https://arxiv.org/abs/1910.14424)            

## Generation
- [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://arxiv.org/abs/1902.04094) (NAACL2019 WS)
- [Pretraining-Based Natural Language Generation for Text Summarization](https://arxiv.org/abs/1902.09243)
- [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) (EMNLP2019) [[github (original)](https://github.com/nlpyang/PreSumm)] [[github (huggingface)](https://github.com/huggingface/transformers/tree/master/examples/summarization)]
- [Multi-stage Pretraining for Abstractive Summarization](https://arxiv.org/abs/1909.10599)
- [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
- [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450) (ICML2019) [[github](https://github.com/microsoft/MASS)], [[github](https://github.com/microsoft/MASS/tree/master/MASS-fairseq)]
- [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197) (NeurIPS2019)
- [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)
- [Towards Making the Most of BERT in Neural Machine Translation](https://arxiv.org/abs/1908.05672)
- [Improving Neural Machine Translation with Pre-trained Representation](https://arxiv.org/abs/1908.07688)
- [On the use of BERT for Neural Machine Translation](https://arxiv.org/abs/1909.12744)
- [Incorporating BERT into Neural Machine Translation](https://openreview.net/forum?id=Hyl7ygStwB) (ICLR2020)
- [Recycling a Pre-trained BERT Encoder for Neural Machine Translation](https://www.aclweb.org/anthology/D19-5603/)
- [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
- [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324) (EMNLP2019)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)
- [Cross-Lingual Natural Language Generation via Pre-Training](https://arxiv.org/abs/1909.10481) (AAAI2020) [[github](https://github.com/CZWin32768/XNLG)]
- [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)
- [PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://arxiv.org/abs/1910.07931)
- [Unsupervised Pre-training for Natural Language Generation: A Literature Review](https://arxiv.org/abs/1911.06171)

## Modification (multi-task, masking strategy, etc.)
- [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504) (ACL2019)
- [The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/2002.07972)
- [BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning](https://arxiv.org/abs/1902.02671) (ICML2019)
- [Unifying Question Answering and Text Classification via Span Extraction](https://arxiv.org/abs/1904.09286)
- [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129) (ACL2019)
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)
- [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412) (AAAI2020)
- [Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)
- [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529) [[github](https://github.com/facebookresearch/SpanBERT)]
- [Blank Language Models](https://arxiv.org/abs/2002.03079)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) [[github](https://github.com/pytorch/fairseq/tree/master/examples/roberta)]
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (ICLR2020)
- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) (ICLR2020)
- [FreeLB: Enhanced Adversarial Training for Language Understanding](https://openreview.net/forum?id=BygzbyHFvB) (ICLR2020)
- [KERMIT: Generative Insertion-Based Modeling for Sequences](https://arxiv.org/abs/1906.01604)
- [DisSent: Sentence Representation Learning from Explicit Discourse Relations](https://arxiv.org/abs/1710.04334) (ACL2019)
- [StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577) (ICLR2020)
- [Syntax-Infused Transformer and BERT models for Machine Translation and Natural Language Understanding](https://arxiv.org/abs/1911.06156)
- [SenseBERT: Driving Some Sense into BERT](https://arxiv.org/abs/1908.05646)
- [Semantics-aware BERT for Language Understanding](https://arxiv.org/abs/1909.02209) (AAAI2020)
- [K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/abs/1909.07606)
- [Knowledge Enhanced Contextual Word Representations](https://arxiv.org/abs/1909.04164) (EMNLP2019)
- [KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation](https://arxiv.org/abs/1911.06136)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (EMNLP2019)
- [SBERT-WK: A Sentence Embedding Method By Dissecting BERT-based Word Models](https://arxiv.org/abs/2002.06652)
- [Universal Text Representation from BERT: An Empirical Study](https://arxiv.org/abs/1910.07973)
- [Symmetric Regularization based BERT for Pair-wise Semantic Reasoning](https://arxiv.org/abs/1909.03405)
- [Transfer Fine-Tuning: A BERT Case Study](https://arxiv.org/abs/1909.00931) (EMNLP2019)
- [Improving Pre-Trained Multilingual Models with Vocabulary Expansion](https://arxiv.org/abs/1909.12440) (CoNLL2019)
- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB)
- [SesameBERT: Attention for Anywhere](https://arxiv.org/abs/1910.03176)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) [[github](https://github.com/google-research/text-to-text-transfer-transformer)]
- [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/abs/1911.03437)
## Transformer variants
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) (ACL2019) [[github](https://github.com/kimiyoung/transformer-xl)]
- [The Evolved Transformer](https://arxiv.org/abs/1901.11117) (ICML2019)
- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (ICLR2020) [[github](https://github.com/google/trax/tree/master/trax/models/reformer)]
- [Transformer on a Diet](https://arxiv.org/abs/2002.06170)

## Probe
- [A Structural Probe for Finding Syntax in Word Representations](https://aclweb.org/anthology/papers/N/N19/N19-1419/) (NAACL2019)
- [Linguistic Knowledge and Transferability of Contextual Representations](https://arxiv.org/abs/1903.08855) (NAACL2019) [[github](https://github.com/nelson-liu/contextual-repr-analysis)]
- [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544) (*SEM2019)
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) (ACL2019)
- [Probing Neural Network Comprehension of Natural Language Arguments](https://arxiv.org/abs/1907.07355) (ACL2019)
- [Cracking the Contextual Commonsense Code: Understanding Commonsense Reasoning Aptitude of Deep Contextual Representations](https://arxiv.org/abs/1910.01157) (EMNLP2019 WS)
- [What do you mean, BERT? Assessing BERT as a Distributional Semantics Model](https://arxiv.org/abs/1911.05758)
- [Quantity doesn't buy quality syntax with neural language models](https://arxiv.org/abs/1909.00111) (EMNLP2019)
- [Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction](https://openreview.net/forum?id=H1xPR3NtPB) (ICLR2020)
- [oLMpics -- On what Language Model Pre-training Captures](https://arxiv.org/abs/1912.13283)
- [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](http://colinraffel.com/publications/arxiv2020how.pdf)

## Inside BERT
- [What does BERT learn about the structure of language?](https://hal.inria.fr/hal-02131630/document) (ACL2019)
- [Open Sesame: Getting Inside BERT's Linguistic Knowledge](https://arxiv.org/abs/1906.01698) (ACL2019 WS)
- [Analyzing the Structure of Attention in a Transformer Language Model](https://arxiv.org/abs/1906.04284) (ACL2019 WS)
- [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341) (ACL2019 WS)
- [Do Attention Heads in BERT Track Syntactic Dependencies?](https://arxiv.org/abs/1911.12246)
- [Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models and Brains](https://arxiv.org/abs/1906.01539) (ACL2019 WS)
- [Inducing Syntactic Trees from BERT Representations](https://arxiv.org/abs/1906.11511) (ACL2019 WS)
- [A Multiscale Visualization of Attention in the Transformer Model](https://arxiv.org/abs/1906.05714) (ACL2019 Demo)
- [Visualizing and Measuring the Geometry of BERT](https://arxiv.org/abs/1906.02715)
- [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://arxiv.org/abs/1909.00512) (EMNLP2019) 
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) (NeurIPS2019)
- [On the Validity of Self-Attention as Explanation in Transformer Models](https://arxiv.org/abs/1908.04211)
- [Visualizing and Understanding the Effectiveness of BERT](https://arxiv.org/abs/1908.05620) (EMNLP2019)
- [Attention Interpretability Across NLP Tasks](https://arxiv.org/abs/1909.11218)
- [Revealing the Dark Secrets of BERT](https://arxiv.org/abs/1908.08593) (EMNLP2019)
- [Investigating BERT's Knowledge of Language: Five Analysis Methods with NPIs](https://arxiv.org/abs/1909.02597) (EMNLP2019)
- [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](https://arxiv.org/abs/1909.01380) (EMNLP2019) 
- [A Primer in BERTology: What we know about how BERT works](https://arxiv.org/abs/2002.12327)
- [Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://arxiv.org/abs/1909.07940) (EMNLP2019)
- [How Does BERT Answer Questions? A Layer-Wise Analysis of Transformer Representations](https://arxiv.org/abs/1909.04925) (CIKM2019)
- [Whatcha lookin' at? DeepLIFTing BERT's Attention in Question Answering](https://arxiv.org/abs/1910.06431)
- [What does BERT Learn from Multiple-Choice Reading Comprehension Datasets?](https://arxiv.org/abs/1910.12391)
- [exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformers Models](https://arxiv.org/abs/1910.05276) [[github](https://github.com/bhoov/exbert)]

## Multi-lingual
- [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://arxiv.org/abs/1812.11760) (ACL2019)
- [Language Model Pretraining](https://arxiv.org/abs/1901.07291) (NeurIPS2019) [[github](https://github.com/facebookresearch/XLM)]
- [75 Languages, 1 Model: Parsing Universal Dependencies Universally](https://arxiv.org/abs/1904.02099) (EMNLP2019) [[github](https://github.com/hyperparticle/udify)]
- [Zero-shot Dependency Parsing with Pre-trained Multilingual Sentence Representations](https://arxiv.org/abs/1910.05479) (EMNLP2019 WS)
- [Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT](https://arxiv.org/abs/1904.09077) (EMNLP2019)
- [How multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) (ACL2019)
- [How Language-Neutral is Multilingual BERT?](https://arxiv.org/abs/1911.03310)
- [Is Multilingual BERT Fluent in Language Generation?](https://arxiv.org/abs/1910.03806)
- [BERT is Not an Interlingua and the Bias of Tokenization](https://www.aclweb.org/anthology/D19-6106/) (EMNLP2019 WS)
- [Cross-Lingual Ability of Multilingual BERT: An Empirical Study](https://openreview.net/forum?id=HJeT3yrtDr) (ICLR2020)
- [Multilingual Alignment of Contextual Word Representations](https://arxiv.org/abs/2002.03518) (ICLR2020)
- [On the Cross-lingual Transferability of Monolingual Representations](https://arxiv.org/abs/1910.11856)
- [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
- [Emerging Cross-lingual Structure in Pretrained Language Models](https://arxiv.org/abs/1911.01464)
- [Can Monolingual Pretrained Models Help Cross-Lingual Classification?](https://arxiv.org/abs/1911.03913)
- [Fully Unsupervised Crosslingual Semantic Textual Similarity Metric Based on BERT for Identifying Parallel Data](https://www.aclweb.org/anthology/K19-1020/) (CoNLL2019)

## Other than English models
- [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894)
- [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372)
- [Multilingual is not enough: BERT for Finnish](https://arxiv.org/abs/1912.07076)
- [BERTje: A Dutch BERT Model](https://arxiv.org/abs/1912.09582)
- [RobBERT: a Dutch RoBERTa-based Language Model](https://arxiv.org/abs/2001.06286)
- [Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language](https://arxiv.org/abs/1905.07213)

## Domain specific
- [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)
- [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474) (ACL2019 WS) 
- [BERT-based Ranking for Biomedical Entity Normalization](https://arxiv.org/abs/1908.03548)
- [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146) (EMNLP2019)
- [Pre-trained Language Model for Biomedical Question Answering](https://arxiv.org/abs/1909.08229)
- [How to Pre-Train Your Model? Comparison of Different Pre-Training Models for Biomedical Question Answering](https://arxiv.org/abs/1911.00712)
- [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)
- [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) (NAACL2019 WS)
- [Progress Notes Classification and Keyword Extraction using Attention-based Deep Learning Models with BERT](https://arxiv.org/abs/1910.05786)
- [SciBERT: Pretrained Contextualized Embeddings for Scientific Text](https://arxiv.org/abs/1903.10676) [[github](https://github.com/allenai/scibert)]
- [PatentBERT: Patent Classification with Fine-Tuning a pre-trained BERT Model](https://arxiv.org/abs/1906.02124)
        
## Multi-modal
- [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766) (ICCV2019)
- [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265) (NeurIPS2019)
- [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557)
- [Selfie: Self-supervised Pretraining for Image Embedding](https://arxiv.org/abs/1906.02940)
- [ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data](https://arxiv.org/abs/2001.07966)
- [Contrastive Bidirectional Transformer for Temporal Representation Learning](https://arxiv.org/abs/1906.05743)
- [M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787)
- [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) (EMNLP2019)
- [Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054) (EMNLP2019)
- [Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/abs/1909.11059) [[github](https://github.com/LuoweiZhou/VLP)]
- [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline](https://arxiv.org/abs/1912.02379)
- [VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530) (ICLR2020)
- [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066)
- [UNITER: Learning UNiversal Image-TExt Representations](https://arxiv.org/abs/1909.11740)
- [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950)
- [Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks](https://arxiv.org/abs/1912.03063)
- [BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations](https://arxiv.org/abs/2002.10832)
- [BERT for Large-scale Video Segment Classification with Test-time Augmentation](https://arxiv.org/abs/1912.01127) (ICCV2019WS)
- [SpeechBERT: Cross-Modal Pre-trained Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559)
- [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453)
- [Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912)
- [Understanding Semantics from Speech Through Pre-training](https://arxiv.org/abs/1909.10924)
- [Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307)
## Model compression
- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
- [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355) (EMNLP2019)
- [Small and Practical BERT Models for Sequence Labeling](https://arxiv.org/abs/1909.00100) (EMNLP2019)
- [Pruning a BERT-based Question Answering Model](https://arxiv.org/abs/1910.06360)
- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) [[github](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)]
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (NeurIPS2019 WS) [[github](https://github.com/huggingface/transformers/tree/master/examples/distillation)]
- [PoWER-BERT: Accelerating BERT inference for Classification Tasks](https://arxiv.org/abs/2001.08950)
- [WaLDORf: Wasteless Language-model Distillation On Reading-comprehension](https://arxiv.org/abs/1912.06638)
- [Extreme Language Model Compression with Optimal Subwords and Shared Projections](https://arxiv.org/abs/1909.11687)
- [BERT-of-Theseus: Compressing BERT by Progressive Module Replacing](https://arxiv.org/abs/2002.02925)
- [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307)
- [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
- [Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT](https://arxiv.org/abs/1909.05840)
- [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188) (NeurIPS2019 WS)
## Misc.
- [Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/abs/1903.07785)
- [Learning and Evaluating General Linguistic Intelligence](https://arxiv.org/abs/1901.11373)
- [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://arxiv.org/abs/1903.05987) (ACL2019 WS)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) (ICLR2020)
- [Machine Translation Evaluation with BERT Regressor](https://arxiv.org/abs/1907.12679)
- [SumQE: a BERT-based Summary Quality Estimation Model](https://arxiv.org/abs/1909.00578) (EMNLP2019)
- [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962) (ICLR2020)
- [Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models](https://openreview.net/forum?id=HkgaETNtDB) (ICLR2020)
- [A Mutual Information Maximization Perspective of Language Representation Learning](https://openreview.net/forum?id=Syx79eBKwr) (ICLR2020)
- [Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment](https://arxiv.org/abs/1907.11932) (AAAI2020)
- [Thieves on Sesame Street! Model Extraction of BERT-based APIs](https://arxiv.org/abs/1910.12366) (ICLR2020)
- [Graph-Bert: Only Attention is Needed for Learning Graph Representations](https://arxiv.org/abs/2001.05140)
- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305)
- [Extending Machine Language Models toward Human-Level Language Understanding](https://arxiv.org/abs/1912.05877)
- [Glyce: Glyph-vectors for Chinese Character Representations](https://arxiv.org/abs/1901.10125)
- [Back to the Future -- Sequential Alignment of Text Representations](https://arxiv.org/abs/1909.03464)
- [Improving Cuneiform Language Identification with BERT](https://www.aclweb.org/anthology/papers/W/W19/W19-1402/) (NAACL2019 WS)
- [BERT has a Moral Compass: Improvements of ethical and moral values of machines](https://arxiv.org/abs/1912.05238)
- [SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction](https://dl.acm.org/citation.cfm?id=3342186) (ACM-BCB2019)
- [On the comparability of Pre-trained Language Models](https://arxiv.org/abs/2001.00781)
- [Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771)
- [Evolution of transfer learning in natural language processing](https://arxiv.org/abs/1910.07370)

#  collect BERT related resources. 

## Papers: 

1. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
, Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

2. [arXiv:1812.06705](https://arxiv.org/abs/1812.06705), Conditional BERT Contextual Augmentation, Authors: Xing Wu, Shangwen Lv, Liangjun Zang, Jizhong Han, Songlin Hu

3. [arXiv:1812.03593](https://arxiv.org/pdf/1812.03593), SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering, Authors: Chenguang Zhu, Michael Zeng, Xuedong Huang

4. [arXiv:1901.02860](https://arxiv.org/abs/1901.02860), Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context,  Authors: Zihang Dai, Zhilin Yang, Yiming Yang, William W. Cohen, Jaime Carbonell, Quoc V. Le and Ruslan Salakhutdinov.

5. [arXiv:1901.04085](https://arxiv.org/pdf/1901.04085.pdf), Passage Re-ranking with BERT, Authors: Rodrigo Nogueira, Kyunghyun Cho

6. [arXiv:1902.02671](https://arxiv.org/pdf/1902.02671.pdf), BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning, Authors: Asa Cooper Stickland, Iain Murray

7. [arXiv:1904.02232](https://arxiv.org/abs/1904.02232), BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis, Authors: Hu Xu, Bing Liu, Lei Shu, Philip S. Yu, [[code](https://github.com/howardhsu/BERT-for-RRC-ABSA)]




# Github Repositories: 

## official implement:

1.  [google-research/bert](https://github.com/google-research/bert),  **officical** TensorFlow code and pre-trained models for BERT ,
![](https://img.shields.io/github/stars/google-research/bert.svg)


## implement of BERT besides tensorflow: 

1. [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch),   Google AI 2018 BERT pytorch implementation,
![](https://img.shields.io/github/stars/codertimo/BERT-pytorch.svg)

2. [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT),   A PyTorch implementation of Google AI's BERT model with script to load Google's pre-trained models,
![](https://img.shields.io/github/stars/huggingface/pytorch-pretrained-BERT.svg)

3. [Separius/BERT-keras](https://github.com/Separius/BERT-keras), Keras implementation of BERT with pre-trained weights, 
![](https://img.shields.io/github/stars/Separius/BERT-keras.svg)

4. [soskek/bert-chainer](https://github.com/soskek/bert-chainer),  Chainer implementation of "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
![](https://img.shields.io/github/stars/soskek/bert-chainer.svg)

5. [innodatalabs/tbert](https://github.com/innodatalabs/tbert), PyTorch port of BERT ML model
![](https://img.shields.io/github/stars/innodatalabs/tbert.svg)

6. [guotong1988/BERT-tensorflow](https://github.com/guotong1988/BERT-tensorflow), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
![](https://img.shields.io/github/stars/guotong1988/BERT-tensorflow.svg)

7. [dreamgonfly/BERT-pytorch](https://github.com/dreamgonfly/BERT-pytorch), 
PyTorch implementation of BERT in "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 
![](https://img.shields.io/github/stars/dreamgonfly/BERT-pytorch.svg)

8. [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert), Implementation of BERT that could load official pre-trained models for feature extraction and prediction
![](https://img.shields.io/github/stars/CyberZHG/keras-bert.svg)

9. [soskek/bert-chainer](https://github.com/soskek/bert-chainer), Chainer implementation of "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
![](https://img.shields.io/github/stars/soskek/bert-chainer.svg)

10. [MaZhiyuanBUAA/bert-tf1.4.0](https://github.com/MaZhiyuanBUAA/bert-tf1.4.0), bert-tf1.4.0
![](https://img.shields.io/github/stars/MaZhiyuanBUAA/bert-tf1.4.0.svg)

11. [dhlee347/pytorchic-bert](https://github.com/dhlee347/pytorchic-bert), Pytorch Implementation of Google BERT,
![](https://img.shields.io/github/stars/dhlee347/pytorchic-bert.svg)

12. [kpot/keras-transformer](https://github.com/kpot/keras-transformer), Keras library for building (Universal) Transformers, facilitating BERT and GPT models,
![](https://img.shields.io/github/stars/kpot/keras-transformer.svg)

13. [miroozyx/BERT_with_keras](https://github.com/miroozyx/BERT_with_keras), A Keras version of Google's BERT model,
![](https://img.shields.io/github/stars/miroozyx/BERT_with_keras.svg)

14. [conda-forge/pytorch-pretrained-bert-feedstock](https://github.com/conda-forge/pytorch-pretrained-bert-feedstock), A conda-smithy repository for pytorch-pretrained-bert. ,
![](https://img.shields.io/github/stars/conda-forge/pytorch-pretrained-bert-feedstock.svg)


15. [Rshcaroline/BERT_Pytorch_fastNLP](https://github.com/Rshcaroline/BERT_Pytorch_fastNLP), A PyTorch & fastNLP implementation of Google AI's BERT model.
![](https://img.shields.io/github/stars/Rshcaroline/BERT_Pytorch_fastNLP.svg)

17. [nghuyong/ERNIE-Pytorch](https://github.com/nghuyong/ERNIE-Pytorch), ERNIE Pytorch Version,
![](https://img.shields.io/github/stars/nghuyong/ERNIE-Pytorch.svg)


18. [dmlc/gluon-nlp](https://github.com/dmlc/gluon-nlp), Gluon + MXNet implementation that reproduces BERT pretraining and finetuning on GLUE benchmark, SQuAD, etc,
![](https://img.shields.io/github/stars/dmlc/gluon-nlp.svg)

19. [dbiir/UER-py](https://github.com/dbiir/UER-py),  UER-py is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of different pre-training models (e.g. BERT), and provides interfaces for users to further extend upon.  
![](https://img.shields.io/github/stars/dbiir/UER-py.svg)

## improvement over BERT:
1. [thunlp/ERNIE](https://github.com/https://github.com/thunlp/ERNIE), Source code and dataset for ACL 2019 paper "ERNIE: Enhanced Language Representation with Informative Entities", imporove bert with heterogeneous information fusion. 
![](https://img.shields.io/github/stars/thunlp/ERNIE.svg)

2. [PaddlePaddle/LARK](https://github.com/PaddlePaddle/LARK),  LAnguage Representations Kit, PaddlePaddle implementation of BERT. It also contains an improved version of BERT, ERNIE, for chinese NLP tasks.  
![](https://img.shields.io/github/stars/PaddlePaddle/LARK.svg)
 
3. [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm), Pre-Training with Whole Word Masking for Chinese BERT  https://arxiv.org/abs/1906.08101,  
![](https://img.shields.io/github/stars/ymcui/Chinese-BERT-wwm.svg)

4. [zihangdai/xlnet](https://github.com/zihangdai/xlnet), XLNet: Generalized Autoregressive Pretraining for Language Understanding, 
![](https://img.shields.io/github/stars/zihangdai/xlnet.svg)

5. [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl), Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, This repository contains the code in both PyTorch and TensorFlow for our paper. 
![](https://img.shields.io/github/stars/kimiyoung/transformer-xl.svg)

6. [GaoPeng97/transformer-xl-chinese](https://github.com/GaoPeng97/transformer-xl-chinese), （transformer xl for text generation of chinese）, 
![](https://img.shields.io/github/stars/GaoPeng97/transformer-xl-chinese.svg)
 


## other resources for BERT: 

1. [brightmart/bert_language_understanding](https://github.com/brightmart/bert_language_understanding), Pre-training of Deep Bidirectional Transformers for Language Understanding: pre-train TextCNN,
![](https://img.shields.io/github/stars/brightmart/bert_language_understanding.svg)

2. [Y1ran/NLP-BERT--ChineseVersion](https://github.com/Y1ran/NLP-BERT--ChineseVersion), 
![](https://img.shields.io/github/stars/Y1ran/NLP-BERT--ChineseVersion.svg)



9. [yangbisheng2009/cn-bert](https://github.com/yangbisheng2009/cn-bert), 
![](https://img.shields.io/github/stars/yangbisheng2009/cn-bert.svg)

4. [JayYip/bert-multiple-gpu](https://github.com/JayYip/bert-multiple-gpu), A multiple GPU support version of BERT,
![](https://img.shields.io/github/stars/JayYip/bert-multiple-gpu.svg)

5. [HighCWu/keras-bert-tpu](https://github.com/HighCWu/keras-bert-tpu), Implementation of BERT that could load official pre-trained models for feature extraction and prediction on TPU, 
![](https://img.shields.io/github/stars/HighCWu/keras-bert-tpu.svg)

6. [Willyoung2017/Bert_Attempt](https://github.com/Willyoung2017/Bert_Attempt), PyTorch Pretrained Bert,
![](https://img.shields.io/github/stars/Willyoung2017/Bert_Attempt.svg)

7. [Pydataman/bert_examples](https://github.com/Pydataman/bert_examples), some examples of bert, run_classifier.py
![](https://img.shields.io/github/stars/Pydataman/bert_examples.svg)

8. [guotong1988/BERT-chinese](https://github.com/guotong1988/BERT-chinese), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 
![](https://img.shields.io/github/stars/guotong1988/BERT-chinese.svg)

9. [zhongyunuestc/bert_multitask](https://github.com/zhongyunuestc/bert_multitask), 多任务task
![](https://img.shields.io/github/stars/zhongyunuestc/bert_multitask.svg)

10. [Microsoft/AzureML-BERT](https://github.com/Microsoft/AzureML-BERT), End-to-end walk through for fine-tuning BERT using Azure Machine Learning , 
![](https://img.shields.io/github/stars/Microsoft/AzureML-BERT.svg)

11. [bigboNed3/bert_serving](https://github.com/bigboNed3/bert_serving), export bert model for serving, 
![](https://img.shields.io/github/stars/nghuyong/ERNIE-Pytorch.svg)

12. [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese), BERT with SentencePiece for Japanese text. 
![](https://img.shields.io/github/stars/bigboNed3/bert_serving.svg)

13. [whqwill/seq2seq-keyphrase-bert](https://github.com/whqwill/seq2seq-keyphrase-bert), add BERT to encoder part for https://github.com/memray/seq2seq-keyphrase-pytorch,
![](https://img.shields.io/github/stars/whqwill/seq2seq-keyphrase-bert.svg)

14. [algteam/bert-examples](https://github.com/algteam/bert-examples), bert-demo, 
![](https://img.shields.io/github/stars/algteam/bert-examples.svg)

15. [cedrickchee/awesome-bert-nlp](https://github.com/cedrickchee/awesome-bert-nlp), A curated list of NLP resources focused on BERT, attention mechanism, Transformer networks, and transfer learning. 
![](https://img.shields.io/github/stars/cedrickchee/awesome-bert-nlp.svg)

16. [cnfive/cnbert](https://github.com/cnfive/cnbert),  
![](https://img.shields.io/github/stars/cnfive/cnbert.svg)

17. [brightmart/bert_customized](https://github.com/brightmart/bert_customized), bert with customized features,
![](https://img.shields.io/github/stars/brightmart/bert_customized.svg)


19. [JayYip/bert-multitask-learning](https://github.com/JayYip/bert-multitask-learning), BERT for Multitask Learning, 
![](https://img.shields.io/github/stars/JayYip/bert-multitask-learning.svg)

20. [yuanxiaosc/BERT_Paper_Chinese_Translation](https://github.com/yuanxiaosc/BERT_Paper_Chinese_Translation), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 。Chinese Translation! https://yuanxiaosc.github.io/2018/12/…, 
![](https://img.shields.io/github/stars/yuanxiaosc/BERT_Paper_Chinese_Translation.svg)

21. [yaserkl/BERTvsULMFIT](https://github.com/yaserkl/BERTvsULMFIT), Comparing Text Classification results using BERT embedding and ULMFIT embedding,
![](https://img.shields.io/github/stars/yaserkl/BERTvsULMFIT.svg)

22. [kpot/keras-transformer](https://github.com/kpot/keras-transformer), Keras library for building (Universal) Transformers, facilitating BERT and GPT models, 
![](https://img.shields.io/github/stars/kpot/keras-transformer.svg)

23. [1234560o/Bert-model-code-interpretation](https://github.com/1234560o/Bert-model-code-interpretation), 
![](https://img.shields.io/github/stars/1234560o/Bert-model-code-interpretation.svg)

24. [cdathuraliya/bert-inference](https://github.com/cdathuraliya/bert-inference), A helper class for Google BERT (Devlin et al., 2018) to support online prediction and model pipelining. 
![](https://img.shields.io/github/stars/cdathuraliya/bert-inference.svg)


26. [gameofdimension/java-bert-predict](https://github.com/gameofdimension/java-bert-predict), turn bert pretrain checkpoint into saved model for a feature extracting demo in java
![](https://img.shields.io/github/stars/gameofdimension/java-bert-predict.svg)

27. [1234560o/Bert-model-code-interpretation](https://github.com/1234560o/Bert-model-code-interpretation), 
![](https://img.shields.io/github/stars/1234560o/Bert-model-code-interpretation.svg)




## domain specific BERT: 

1. [allenai/scibert](https://github.com/allenai/scibert), A BERT model for scientific text. https://arxiv.org/abs/1903.10676,
![](https://img.shields.io/github/stars/allenai/scibert.svg)

2. [MeRajat/SolvingAlmostAnythingWithBert](https://github.com/MeRajat/SolvingAlmostAnythingWithBert), BioBert Pytorch
![](https://img.shields.io/github/stars/MeRajat/SolvingAlmostAnythingWithBert.svg)

3. [kexinhuang12345/clinicalBERT](https://github.com/kexinhuang12345/clinicalBERT), ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission https://arxiv.org/abs/1904.05342
![](https://img.shields.io/github/stars/kexinhuang12345/clinicalBERT.svg)

4. [EmilyAlsentzer/clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT), repository for Publicly Available Clinical BERT Embeddings
![](https://img.shields.io/github/stars/EmilyAlsentzer/clinicalBERT.svg)


## BERT Deploy Tricks:

1. [zhihu/cuBERT](https://github.com/zhihu/cuBERT), Fast implementation of BERT inference directly on NVIDIA (CUDA, CUBLAS) and Intel MKL
![](https://img.shields.io/github/stars/zhihu/cuBERT.svg)

2. [xmxoxo/BERT-train2deploy](https://github.com/xmxoxo/BERT-train2deploy), Bert Model training and deploy,  
![](https://img.shields.io/github/stars/xmxoxo/BERT-train2deploy.svg)

## BERT QA & RC task:

1. [sogou/SMRCToolkit](https://github.com/sogou/SMRCToolkit), This toolkit was designed for the fast and efficient development of modern machine comprehension models, including both published models and original prototypes., 
![](https://img.shields.io/github/stars/sogou/SMRCToolkit.svg)


1. [benywon/ChineseBert](https://github.com/benywon/ChineseBert), This is a chinese Bert model specific for question answering,
![](https://img.shields.io/github/stars/benywon/ChineseBert.svg)

3. [matthew-z/R-net](https://github.com/matthew-z/R-net), R-net in PyTorch, with BERT and ELMo,
![](https://img.shields.io/github/stars/matthew-z/R-net.svg)

4. [nyu-dl/dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert), Passage Re-ranking with BERT,
![](https://img.shields.io/github/stars/nyu-dl/dl4marco-bert.svg)

5. [xzp27/BERT-for-Chinese-Question-Answering](https://github.com/xzp27/BERT-for-Chinese-Question-Answering), 
![](https://img.shields.io/github/stars/xzp27/BERT-for-Chinese-Question-Answering.svg)

6. [chiayewken/bert-qa](https://github.com/chiayewken/bert-qa), BERT for question answering starting with HotpotQA,
![](https://img.shields.io/github/stars/chiayewken/bert-qa.svg)

8. [ankit-ai/BertQA-Attention-on-Steroids](https://github.com/ankit-ai/BertQA-Attention-on-Steroids), BertQA - Attention on Steroids,
![](https://img.shields.io/github/stars/ankit-ai/BertQA-Attention-on-Steroids.svg)

10. [NoviScl/BERT-RACE](https://github.com/NoviScl/BERT-RACE), This work is based on Pytorch implementation of BERT (https://github.com/huggingface/pytorch-pretrained-BERT). I adapted the original BERT model to work on multiple choice machine comprehension.
![](https://img.shields.io/github/stars/NoviScl/BERT-RACE.svg)

11. [eva-n27/BERT-for-Chinese-Question-Answering](https://github.com/eva-n27/BERT-for-Chinese-Question-Answering), 
![](https://img.shields.io/github/stars/eva-n27/BERT-for-Chinese-Question-Answering.svg)

12. [allenai/allennlp-bert-qa-wrapper](https://github.com/allenai/allennlp-bert-qa-wrapper),  This is a simple wrapper on top of pretrained BERT based QA models from pytorch-pretrained-bert to make AllenNLP model archives, so that you can serve demos from AllenNLP.
![](https://img.shields.io/github/stars/allenai/allennlp-bert-qa-wrapper.svg)

13. [edmondchensj/ChineseQA-with-BERT](https://github.com/edmondchensj/ChineseQA-with-BERT), EECS 496: Advanced Topics in Deep Learning Final Project: Chinese Question Answering with BERT (Baidu DuReader Dataset)
![](https://img.shields.io/github/stars/edmondchensj/ChineseQA-with-BERT.svg)

14. [graykode/toeicbert](https://github.com/graykode/toeicbert), TOEIC(Test of English for International Communication) solving using pytorch-pretrained-BERT model.,
![](https://img.shields.io/github/starsgraykode/toeicbert.svg)

15. [graykode/KorQuAD-beginner](https://github.com/graykode/KorQuAD-beginner), https://github.com/graykode/KorQuAD-beginner
![](https://img.shields.io/github/stars/graykode/KorQuAD-beginner.svg)

16. [krishna-sharma19/SBU-QA](https://github.com/krishna-sharma19/SBU-QA), This repository uses pretrain BERT embeddings for transfer learning in QA domain
![](https://img.shields.io/github/stars/krishna-sharma19/SBU-QA.svg)



## BERT classification task:

1. [zhpmatrix/Kaggle-Quora-Insincere-Questions-Classification](https://github.com/zhpmatrix/Kaggle-Quora-Insincere-Questions-Classification), 
![](https://img.shields.io/github/stars/zhpmatrix/Kaggle-Quora-Insincere-Questions-Classification.svg)

2. [maksna/bert-fine-tuning-for-chinese-multiclass-classification](https://github.com/maksna/bert-fine-tuning-for-chinese-multiclass-classification), use google pre-training model bert to fine-tuning for the chinese multiclass classification
![](https://img.shields.io/github/stars/maksna/bert-fine-tuning-for-chinese-multiclass-classification.svg)

3. [NLPScott/bert-Chinese-classification-task](https://github.com/NLPScott/bert-Chinese-classification-task), 
![](https://img.shields.io/github/stars/NLPScott/bert-Chinese-classification-task.svg)

4. [Socialbird-AILab/BERT-Classification-Tutorial](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial),
![](https://img.shields.io/github/stars/Socialbird-AILab/BERT-Classification-Tutorial.svg)

5. [fooSynaptic/BERT_classifer_trial](https://github.com/fooSynaptic/BERT_classifer_trial), BERT trial for chinese corpus classfication
![](https://img.shields.io/github/stars/fooSynaptic/BERT_classifer_trial.svg)

6. [xiaopingzhong/bert-finetune-for-classfier](https://github.com/xiaopingzhong/bert-finetune-for-classfier), 
![](https://img.shields.io/github/stars/xiaopingzhong/bert-finetune-for-classfier.svg)

8. [pengming617/bert_classification](https://github.com/pengming617/bert_classification), ,
![](https://img.shields.io/github/stars/pengming617/bert_classification.svg)

9. [xieyufei1993/Bert-Pytorch-Chinese-TextClassification](https://github.com/xieyufei1993/Bert-Pytorch-Chinese-TextClassification), Pytorch Bert Finetune in Chinese Text Classification,
![](https://img.shields.io/github/stars/xieyufei1993/Bert-Pytorch-Chinese-TextClassification.svg)

10. [liyibo/text-classification-demos](https://github.com/liyibo/text-classification-demos), Neural models for Text Classification in Tensorflow, such as cnn, dpcnn, fasttext, bert ...,
![](https://img.shields.io/github/stars/liyibo/text-classification-demos.svg)

11. [circlePi/BERT_Chinese_Text_Class_By_pytorch](https://github.com/circlePi/BERT_Chinese_Text_Class_By_pytorch), A Pytorch implements of Chinese text class based on BERT_Pretrained_Model,
![](https://img.shields.io/github/stars/circlePi/BERT_Chinese_Text_Class_By_pytorch.svg)

12. [kaushaltrivedi/bert-toxic-comments-multilabel](https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel), Multilabel classification for Toxic comments challenge using Bert,
![](https://img.shields.io/github/stars/kaushaltrivedi/bert-toxic-comments-multilabel.svg)

13. [lonePatient/BERT-chinese-text-classification-pytorch](https://github.com/lonePatient/BERT-chinese-text-classification-pytorch), This repo contains a PyTorch implementation of a pretrained BERT model for text classification.,
![](https://img.shields.io/github/stars/lonePatient/BERT-chinese-text-classification-pytorch.svg)



## BERT Sentiment Analysis

1. [Chung-I/Douban-Sentiment-Analysis](https://github.com/Chung-I/Douban-Sentiment-Analysis), Sentiment Analysis on Douban Movie Short Comments Dataset using BERT.
![](https://img.shields.io/github/stars/Chung-I/Douban-Sentiment-Analysis.svg)

14. [lynnna-xu/bert_sa](https://github.com/lynnna-xu/bert_sa), bert sentiment analysis tensorflow serving with RESTful API
![](https://img.shields.io/github/stars/lynnna-xu/bert_sa.svg)

15. [HSLCY/ABSA-BERT-pair](https://github.com/HSLCY/ABSA-BERT-pair), Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence (NAACL 2019) https://arxiv.org/abs/1903.09588,
![](https://img.shields.io/github/stars/HSLCY/ABSA-BERT-pair.svg)

16. [songyouwei/ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), Aspect Based Sentiment Analysis, PyTorch Implementations.,
![](https://img.shields.io/github/stars/songyouwei/ABSA-PyTorch.svg)

17. [howardhsu/BERT-for-RRC-ABSA](https://github.com/howardhsu/BERT-for-RRC-ABSA), code for our NAACL 2019 paper: "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
![](https://img.shields.io/github/stars/howardhsu/BERT-for-RRC-ABSA.svg)

7. [brightmart/sentiment_analysis_fine_grain](https://github.com/brightmart/sentiment_analysis_fine_grain), Multi-label Classification with BERT; Fine Grained Sentiment Analysis from AI challenger,
![](https://img.shields.io/github/stars/brightmart/sentiment_analysis_fine_grain.svg)


## BERT  NER  task:  

2. [zhpmatrix/bert-sequence-tagging](https://github.com/zhpmatrix/bert-sequence-tagging), 
![](https://img.shields.io/github/stars/zhpmatrix/bert-sequence-tagging.svg)

3. [kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER), Use google BERT to do CoNLL-2003 NER ! ,
![](https://img.shields.io/github/stars/kyzhouhzau/BERT-NER.svg)

4. [king-menin/ner-bert](https://github.com/king-menin/ner-bert), NER task solution (bert-Bi-LSTM-CRF) with google bert https://github.com/google-research.
![](https://img.shields.io/github/stars/king-menin/ner-bert.svg)

5. [macanv/BERT-BiLSMT-CRF-NER](https://github.com/macanv/BERT-BiLSMT-CRF-NER), Tensorflow solution of NER task Using BiLSTM-CRF model with Google BERT Fine-tuning  ,
![](https://img.shields.io/github/stars/macanv/BERT-BiLSMT-CRF-NER.svg)

6. [FuYanzhe2/Name-Entity-Recognition](https://github.com/FuYanzhe2/Name-Entity-Recognition), Lstm-crf,Lattice-CRF,bert-ner
![](https://img.shields.io/github/stars/FuYanzhe2/Name-Entity-Recognition.svg)

7. [mhcao916/NER_Based_on_BERT](https://github.com/mhcao916/NER_Based_on_BERT), this project is based on google bert model, which is a Chinese NER
![](https://img.shields.io/github/stars/mhcao916/NER_Based_on_BERT.svg)

8. [ProHiryu/bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner),
![](https://img.shields.io/github/stars/ProHiryu/bert-chinese-ner.svg)

9. [sberbank-ai/ner-bert](https://github.com/sberbank-ai/ner-bert), BERT-NER (nert-bert) with google bert,
![](https://img.shields.io/github/stars/sberbank-ai/ner-bert.svg)

10. [kyzhouhzau/Bert-BiLSTM-CRF](https://github.com/kyzhouhzau/Bert-BiLSTM-CRF), This model base on bert-as-service. Model structure : bert-embedding bilstm crf. ,
![](https://img.shields.io/github/stars/kyzhouhzau/Bert-BiLSTM-CRF.svg)

11. [Hoiy/berserker](https://github.com/Hoiy/berserker), Berserker - BERt chineSE woRd toKenizER, Berserker (BERt chineSE woRd toKenizER) is a Chinese tokenizer built on top of Google's BERT model. ,
![](https://img.shields.io/github/stars/Hoiy/berserker.svg)

12. [Kyubyong/bert_ner](https://github.com/Kyubyong/bert_ner), Ner with Bert,
![](https://img.shields.io/github/stars/Kyubyong/bert_ner.svg)

13. [jiangpinglei/BERT_ChineseWordSegment](https://github.com/jiangpinglei/BERT_ChineseWordSegment),  A Chinese word segment model based on BERT, F1-Score 97%,
![](https://img.shields.io/github/stars/jiangpinglei/BERT_ChineseWordSegment.svg)

14. [yanwii/ChineseNER](https://github.com/yanwii/ChineseNER), 
![](https://img.shields.io/github/stars/yanwii/ChineseNER.svg)

15. [lemonhu/NER-BERT-pytorch](https://github.com/lemonhu/NER-BERT-pytorch), PyTorch solution of NER task Using Google AI's pre-trained BERT model.
![](https://img.shields.io/github/stars/lemonhu/NER-BERT-pytorch.svg)


## BERT Text Summarization Task: 

1. [nlpyang/BertSum](https://github.com/nlpyang/BertSum), Code for paper Fine-tune BERT for Extractive Summarization,
![](https://img.shields.io/github/stars/nlpyang/BertSum.svg)

2. [santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning](https://github.com/santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning), Abstractive summarisation using Bert as encoder and Transformer Decoder,
![](https://img.shields.io/github/stars/santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning.svg)

3. [nayeon7lee/bert-summarization](https://github.com/nayeon7lee/bert-summarization), Implementation of 'Pretraining-Based Natural Language Generation for Text Summarization', Paper: https://arxiv.org/pdf/1902.09243.pdf
![](https://img.shields.io/github/stars/nayeon7lee/bert-summarization.svg)

4. [dmmiller612/lecture-summarizer](https://github.com/dmmiller612/lecture-summarizer), Lecture summarizer with BERT
![](https://img.shields.io/github/stars/dmmiller612/lecture-summarizer.svg)




## BERT Text Generation Task: 
1. [asyml/texar](https://github.com/asyml/texar), Toolkit for Text Generation and Beyond https://texar.io, Texar is a general-purpose text generation toolkit, has also implemented BERT here for classification, and text generation applications by combining with Texar's other modules.
![](https://img.shields.io/github/stars/asyml/texar.svg)

2. [voidful/BertGenerate](https://github.com/voidful/BertGenerate), Fine tuning bert for text generation,
![](https://img.shields.io/github/stars/voidful/BertGenerate.svg)

3. [Tiiiger/bert_score](https://github.com/Tiiiger/bert_score), BERT score for language generation,
![](https://img.shields.io/github/stars/Tiiiger/bert_score.svg)



## BERT  Knowledge Graph Task : 

1. [lvjianxin/Knowledge-extraction](https://github.com/lvjianxin/Knowledge-extraction), 
![](https://img.shields.io/github/stars/lvjianxin/Knowledge-extraction.svg)

2. [sakuranew/BERT-AttributeExtraction](https://github.com/sakuranew/BERT-AttributeExtraction), USING BERT FOR Attribute Extraction in KnowledgeGraph. fine-tuning and feature extraction.,
![](https://img.shields.io/github/stars/sakuranew/BERT-AttributeExtraction.svg)

3. [aditya-AI/Information-Retrieval-System-using-BERT](https://github.com/aditya-AI/Information-Retrieval-System-using-BERT),
![](https://img.shields.io/github/stars/aditya-AI/Information-Retrieval-System-using-BERT.svg)

4. [jkszw2014/bert-kbqa-NLPCC2017](https://github.com/jkszw2014/bert-kbqa-NLPCC2017), A trial of kbqa based on bert for NLPCC2016/2017 Task 5, https://blog.csdn.net/ai_1046067944/article/details/86707784  ,
![](https://img.shields.io/github/stars/jkszw2014/bert-kbqa-NLPCC2017.svg)

5. [yuanxiaosc/Schema-based-Knowledge-Extraction](https://github.com/yuanxiaosc/Schema-based-Knowledge-Extraction), Code for http://lic2019.ccf.org.cn/kg,
![](https://img.shields.io/github/stars/yuanxiaosc/Schema-based-Knowledge-Extraction.svg)

6. [yuanxiaosc/Entity-Relation-Extraction](https://github.com/yuanxiaosc/Entity-Relation-Extraction),  Entity and Relation Extraction Based on TensorFlow.Schema based Knowledge Extraction, SKE 2019 http://lic2019.ccf.org.cn,
![](https://img.shields.io/github/stars/yuanxiaosc/Entity-Relation-Extraction.svg)

7. [WenRichard/KBQA-BERT](https://github.com/WenRichard/KBQA-BERT), https://zhuanlan.zhihu.com/p/62946533  ,
![](https://img.shields.io/github/stars/WenRichard/KBQA-BERT.svg)


## BERT  Coreference Resolution 
1. [ianycxu/RGCN-with-BERT](https://github.com/ianycxu/RGCN-with-BERT), Gated-Relational Graph Convolutional Networks (RGCN) with BERT for Coreference Resolution Task
![](https://img.shields.io/github/stars/ianycxu/RGCN-with-BERT.svg)

2. [isabellebouchard/BERT_for_GAP-coreference](https://github.com/isabellebouchard/BERT_for_GAP-coreference),  BERT finetuning for GAP unbiased pronoun resolution
![](https://img.shields.io/github/stars/isabellebouchard/BERT_for_GAP-coreference.svg)



## BERT  visualization toolkit: 
1. [jessevig/bertviz](https://github.com/jessevig/bertviz), Tool for visualizing BERT's attention,
![](https://img.shields.io/github/stars/jessevig/bertviz.svg)

## BERT chatbot :
1. [GaoQ1/rasa_nlu_gq](https://github.com/GaoQ1/rasa_nlu_gq), turn natural language into structured data,
![](https://img.shields.io/github/stars/GaoQ1/rasa_nlu_gq.svg)

2. [GaoQ1/rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn), 
![](https://img.shields.io/github/stars/GaoQ1/rasa_chatbot_cn.svg)

3. [GaoQ1/rasa-bert-finetune](https://github.com/GaoQ1/rasa-bert-finetune), 
![](https://img.shields.io/github/stars/GaoQ1/rasa-bert-finetune.svg)

5. [geodge831012/bert_robot](https://github.com/geodge831012/bert_robot)
![](https://img.shields.io/github/stars/geodge831012/bert_robot.svg)

6. [yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification), This is the template code to use BERT for sequence lableing and text classification, in order to facilitate BERT for more tasks. Currently, the template code has included conll-2003 named entity identification, Snips Slot Filling and Intent Prediction.
![](https://img.shields.io/github/stars/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification.svg)

7. [guillaume-chevalier/ReuBERT](https://github.com/guillaume-chevalier/ReuBERT), A question-answering chatbot, simply.
![](https://img.shields.io/github/stars/guillaume-chevalier/ReuBERT.svg)

## BERT language model and embedding: 

1.  [hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service),    Mapping a variable-length sentence to a fixed-length vector using pretrained BERT model,
![](https://img.shields.io/github/stars/hanxiao/bert-as-service.svg)

2. [YC-wind/embedding_study](https://github.com/YC-wind/embedding_study),
![](https://img.shields.io/github/stars/YC-wind/embedding_study.svg)

3. [Kyubyong/bert-token-embeddings](https://github.com/Kyubyong/bert-token-embeddings), Bert Pretrained Token Embeddings,
![](https://img.shields.io/github/stars/Kyubyong/bert-token-embeddings.svg)

4. [xu-song/bert_as_language_model](https://github.com/xu-song/bert_as_language_model), bert as language model, fork from https://github.com/google-research/bert,
![](https://img.shields.io/github/stars/xu-song/bert_as_language_model.svg)

5. [yuanxiaosc/Deep_dynamic_word_representation](https://github.com/yuanxiaosc/Deep_dynamic_word_representation), TensorFlow code and pre-trained models for deep dynamic word representation (DDWR). It combines the BERT model and ELMo's deep context word representation.,
![](https://img.shields.io/github/stars/yuanxiaosc/Deep_dynamic_word_representation.svg)

6. [imgarylai/bert-embedding](https://github.com/imgarylai/bert-embedding), Token level embeddings from BERT model on mxnet and gluonnlp http://bert-embedding.readthedocs.io/,
![](https://img.shields.io/github/stars/imgarylai/bert-embedding.svg)

7. [terrifyzhao/bert-utils](https://github.com/terrifyzhao/bert-utils),
![](https://img.shields.io/github/stars/terrifyzhao/bert-utils.svg)

8. [fennuDetudou/BERT_implement](https://github.com/fennuDetudou/BERT_implement),
![](https://img.shields.io/github/stars/fennuDetudou/BERT_implement.svg)

9. [whqwill/seq2seq-keyphrase-bert](https://github.com/whqwill/seq2seq-keyphrase-bert), add BERT to encoder part for https://github.com/memray/seq2seq-keyphrase-pytorch,
![](https://img.shields.io/github/stars/whqwill/seq2seq-keyphrase-bert.svg)

10. [charles9n/bert-sklearn](https://github.com/charles9n/bert-sklearn), a sklearn wrapper for Google's BERT model,
![](https://img.shields.io/github/stars/charles9n/bert-sklearn.svg)


12. [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Ongoing research training transformer language models at scale, including: BERT,
![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg)

13. [hankcs/BERT-token-level-embedding](https://github.com/hankcs/BERT-token-level-embedding), Generate BERT token level embedding without pain
![](https://img.shields.io/github/stars/hankcs/BERT-token-level-embedding.svg)



## BERT Text Match: 

1. [pengming617/bert_textMatching](https://github.com/pengming617/bert_textMatching), 利用预训练的中文模型实现基于bert的语义匹配模型 数据集为LCQMC官方数据
![](https://img.shields.io/github/stars/pengming617/bert_textMatching.svg)

14. [Brokenwind/BertSimilarity](https://github.com/Brokenwind/BertSimilarity), Computing similarity of two sentences with google's BERT algorithm
![](https://img.shields.io/github/stars/Brokenwind/BertSimilarity.svg)

25. [policeme/chinese_bert_similarity](https://github.com/policeme/chinese_bert_similarity), bert chinese similarity
![](https://img.shields.io/github/stars/policeme/chinese_bert_similarity.svg)

26. [lonePatient/bert-sentence-similarity-pytorch](https://github.com/lonePatient/bert-sentence-similarity-pytorch), This repo contains a PyTorch implementation of a pretrained BERT model for sentence similarity task.
![](https://img.shields.io/github/stars/lonePatient/bert-sentence-similarity-pytorch.svg)

27. [nouhadziri/DialogEntailment](https://github.com/nouhadziri/DialogEntailment), The implementation of the paper "Evaluating Coherence in Dialogue Systems using Entailment" https://arxiv.org/abs/1904.03371
![](https://img.shields.io/github/stars/nouhadziri/DialogEntailment.svg)

## ko bert
https://github.com/jeongukjae/KR-BERT-SimCSE

## BERT tutorials: 

1. [graykode/nlp-tutorial](https://github.com/graykode/nlp-tutorial), Natural Language Processing Tutorial for Deep Learning Researchers https://www.reddit.com/r/MachineLearn…,
![](https://img.shields.io/github/stars/graykode/nlp-tutorial.svg)

2. [dragen1860/TensorFlow-2.x-Tutorials](https://github.com/dragen1860/TensorFlow-2.x-Tutorials), TensorFlow 2.x version's Tutorials and Examples, including CNN, RNN, GAN, Auto-Encoders, FasterRCNN, GPT, BERT examples, etc. TF 2.0。,
![](https://img.shields.io/github/stars/dragen1860/TensorFlow-2.x-Tutorials.svg)

## 한국어 sentence bert 모델
- 깃허브에 sentence-transformers 다국어 모델과의 벤치마크 성능 비교를 기재해두었습니다) ko-sentence-transformers 라이브러리를 설치하시면 허깅페이스 허브에서 바로 다운받아 사용 가능합니다. 
- 허깅페이스 모델: https://huggingface.co/jhgan/ko-sbert-multitask
- 깃허브 저장소: https://github.com/jhgan00/ko-sentence-transformers
