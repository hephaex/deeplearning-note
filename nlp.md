# natural language processing

## Flaxformer: transformer architectures in JAX/Flax
* https://github.com/google/flaxformer

## Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance
* https://ai.googleblog.com/.../pathways-language-model...

## Part-of-speech Tagging
* (2000) [A Statistical Part-of-Speech Tagger](https://arxiv.org/pdf/cs/0003055.pdf)
  - **TLDR**: Seminal paper demonstrating a powerful HMM-based POS tagger. Many tips and tricks for building such classical systems included. 
* (2003) [Feature-rich part-of-speech tagging with a cyclic dependency network](https://nlp.stanford.edu/pubs/tagging.pdf)
  - **TLDR**: Proposes a number of powerful linguistic features for building a (then) SOTA POS-tagging system
* (2015) [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  - **TLDR**: Proposes an element sequence-tagging model combining neural networks with conditional random fields, achieving SOTA in POS-tagging, NER, and chunking. 
## https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code

## Parsing
* (2003) [Accurate unlexicalized parsing](https://people.eecs.berkeley.edu/~klein/papers/unlexicalized-parsing.pdf) :bulb:
  - **TLDR**: Beautiful paper demonstrating that unlexicalized probabilistic context free grammars can exceed the performance of lexicalized PCFGs.
* (2014) [A Fast and Accurate Dependency Parser using Neural Networks](cs.stanford.edu/~danqi/papers/emnlp2014.pdf)
  - **TLDR**: Very important work ushering in a new wave of neural network-based parsing architectures, achieving SOTA performance as well as blazing parsing speeds. 

## Named Entity Recognition
* (2005) [Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling](http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf)
  - **TLDR**: Using cool Monte Carlo methods combined with a conditional random field model, this work achieves a huge error reduction in certain information extraction benchmarks.
* (2015) [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  - **TLDR**: Proposes an element sequence-tagging model combining neural networks with conditional random fields, achieving SOTA in POS-tagging, NER, and chunking. 

## Coreference Resolution
* (2010) [A multi-pass sieve for coreference resolution](https://nlp.stanford.edu/pubs/conllst2011-coref.pdf) :bulb:
  - **TLDR**: Proposes a sieve-based approach to coreference resolution that for many years (until deep learning approaches) was SOTA.
* (2015) [Entity-Centric Coreference Resolution with Model Stacking](http://cs.stanford.edu/~kevclark/resources/clark-manning-acl15-entity.pdf) 
  - **TLDR**: This work offers a nifty approach to building coreference chains iteratively using entity-level features.
* (2016) [Improving Coreference Resolution by Learning Entity-Level Distributed Representations](https://cs.stanford.edu/~kevclark/resources/clark-manning-acl16-improving.pdf)
  - **TLDR**: One of the earliest effective approaches to using neural networks for coreference resolution, significantly outperforming the SOTA.

## Sentiment Analysis
* (2012) [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018)
  - **TLDR**: Very elegant paper, illustrating that simple Naive Bayes models with bigram features can outperform more sophisticated methods like support vector machines on tasks such as sentiment analysis.
* (2013) [Recursive deep models for semantic compositionality over a sentiment treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) :vhs:
  - **TLDR**: Introduces the Stanford Sentiment Treebank, a wonderful resource for fine-grained sentiment annotation on sentences. Also introduces the Recursive Neural Tensor Network, a neat linguistically-motivated deep learning architecture. 

## Natural Logic/Inference
* (2007) [Natural Logic for Textual Inference](https://nlp.stanford.edu/pubs/natlog-wtep07.pdf)
  - **TLDR**: Proposes a rigorous logic-based approach to the problem of textual inference called natural logic. Very cool mathematically-motivated transforms are used to deduce the relationship between phrases. 
* (2008) [An Extended Model of Natural Logic](dl.acm.org/citation.cfm?id=1693772)
  - **TLDR**: Extends previous work on natural logic for inference, adding phenomena such as semantic exclusion and implicativity to enhance the premise-hypothesis transform process.
* (2014) [Recursive Neural Networks Can Learn Logical Semantics](https://arxiv.org/abs/1406.1827)
  - **TLDR**: Demonstrates that deep learning architectures such as neural tensor networks can effectively be applied to natural language inference. 
* (2015) [A large annotated corpus for learning natural language inference](http://nlp.stanford.edu/pubs/snli_paper.pdf) :vhs:
  - **TLDR**: Introduces the Stanford Natural Language Inference corpus, a wonderful NLI resource larger by two orders of magnitude over previous datasets. 

## Machine Translation
* (1993) [The Mathematics of Statistical Machine Translation](https://www.aclweb.org/anthology/J93-2003) :bulb:
  - **TLDR**: Introduces the IBM machine translation models, several seminal models in statistical MT. 
* (2002) [BLEU: A Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf) :vhs:
  - **TLDR**: Proposes BLEU, the defacto evaluation technique used for machine translation (even today!)
* (2003) [Statistical Phrase-Based Translation](http://dl.acm.org/citation.cfm?id=1073462)
  - **TLDR**: Introduces a phrase-based translation model for MT, doing nice analysis that demonstrates why phrase-based models outperform word-based ones. 
* (2014) [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) :bulb:
  - **TLDR**: Introduces the sequence-to-sequence neural network architecture. While only applied to MT in this paper, it has since become one of the cornerstone architectures of modern natural language processing. 
* (2015) [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) :bulb:
  - **TLDR**: Extends previous sequence-to-sequence architectures for MT by using the attention mechanism, a powerful tool for allowing a target word to softly search for important signal from the source sentence. 
* (2015) [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025)
  - **TLDR**: Introduces two new attention mechanisms for MT, using them to achieve SOTA over existing neural MT systems.
* (2016) [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)
  - **TLDR**: Introduces byte pair encoding, an effective technique for allowing neural MT systems to handle (more) open-vocabulary translation.
* (2016) [Pointing the Unknown Words](https://www.aclweb.org/anthology/P16-1014)
  - **TLDR**: Proposes a copy-mechanism for allowing MT systems to more effectively copy words from a source context sequence.
* (2016) [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) 
  - **TLDR**: A wonderful case-study demonstrating what a production-capacity machine translation system (in this case that of Google) looks like. 

## Semantic Parsing
* (2013) [Semantic Parsing on Freebase from Question-Answer Pairs](www.aclweb.org/anthology/D13-1160) :bulb: :vhs:
  - **TLDR**: Proposes an elegant technique for semantic parsing that learns directly from question-answer pairs, without the need for annotated logical forms, allowing the system to scale up to Freebase. 
* (2014) [Semantic Parsing via Paraphrasing](http://aclweb.org/anthology/P14-1133)
  - **TLDR**: Develops a unique paraphrase model for learning appropriate candidate logical forms from question-answer pairs, improving SOTA on existing Q/A datasets. 
* (2015) [Building a Semantic Parser Overnight](https://cs.stanford.edu/~pliang/papers/overnight-acl2015.pdf) :vhs:
  - **TLDR**: Neat paper showing that a semantic parser can be built from scratch starting with no training examples!
* (2015) [Bringing Machine Learning and Computational Semantics Together](http://www.stanford.edu/~cgpotts/manuscripts/liang-potts-semantics.pdf)
  - **TLDR**: A nice overview of a computational semantics framework that uses machine learning to effectively learn logical forms for semantic parsing. 

## Question Answering/Reading Comprehension
* (2016) [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858)
  - **TLDR**: A great wake-up call paper, demonstrating that SOTA performance can be achieved on certain reading comprehension datasets using simple systems with carefully chosen features. Don't forget non-deep learning methods!
* (2017) [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) :vhs:
  - **TLDR**: Introduces the SQUAD dataset, a question-answering corpus that has become one of the defacto benchmarks used today. 

## Natural Language Generation/Summarization
* (2004) [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013) :vhs:
  - **TLDR**: Introduces ROUGE, an evaluation metric for summarization that is used to this day on a variety of sequence transduction tasks. 
* (2015) [Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/abs/1508.01745)
  - **TLDR**: Proposes a neural natural language generator that jointly optimises sentence planning and surface realization, outperforming other systems on human eval. 
* (2016) [Pointing the Unknown Words](https://arxiv.org/abs/1603.08148)
  - **TLDR**: Proposes a copy-mechanism for allowing MT systems to more effectively copy words from a source context sequence.
* (2017) [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
  - **TLDR**: This work offers an elegant soft copy mechanism, that drastically outperforms the SOTA on abstractive summarization. 

## Dialogue Systems
* (2011) [Data-drive Response Generation in Social Media](http://dl.acm.org/citation.cfm?id=2145500)
  - **TLDR**: Proposes using phrase-based statistical machine translation methods to the problem of response generation. 
* (2015) [Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/abs/1508.01745)
  - **TLDR**: Proposes a neural natural language generator that jointly optimises sentence planning and surface realization, outperforming other systems on human eval.
* (2016) [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/abs/1603.08023) :bulb:
  - **TLDR**: Important work demonstrating that existing automatic metrics used for dialogue woefully do not correlate well with human judgment. 
* (2016) [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/abs/1604.04562)
  - **TLDR**: Proposes a neat architecture for decomposing a dialogue system into a number of individually-trained neural network components. 
* (2016) [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055)
  - **TLDR**: Introduces a maximum mutual information objective function for training dialogue systems. 
* (2016) [The Dialogue State Tracking Challenge Series: A Review](https://pdfs.semanticscholar.org/4ba3/39bd571585fadb1fb1d14ef902b6784f574f.pdf)
  - **TLDR**: A nice overview of the dialogue state tracking challenges for dialogue systems. 
* (2017) [A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](https://arxiv.org/abs/1701.04024)
  - **TLDR**: Shows that simple sequence-to-sequence architectures with a copy mechanism can perform competitively on existing task-oriented dialogue datasets. 
* (2017) [Key-Value Retrieval Networks for Task-Oriented Dialogue](https://arxiv.org/abs/1705.05414) :vhs:
  - **TLDR**: Introduces a new multidomain dataset for task-oriented dataset as well as an architecture for softly incorporating information from structured knowledge bases into dialogue systems. 
* (2017) [Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/abs/1704.07130) :vhs:
  - **TLDR**: Introduces a new collaborative dialogue dataset, as well as an architecture for representing structured knowledge via knowledge graph embeddings. 
* (2017) [Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning](https://arxiv.org/abs/1702.03274)
  - **TLDR**: Introduces a hybrid dialogue architecture that can be jointly trained via supervised learning as well as reinforcement learning and combines neural network techniques with fine-grained rule-based approaches. 

## Interactive Learning
* (1971) [Procedures as a Representation for Data in a Computer Program for Understanding Natural Language](http://hci.stanford.edu/~winograd/shrdlu/AITR-235.pdf)
  - **TLDR**: One of the seminal papers in computer science, introducing SHRDLU an early system for computers understanding human language commands. 
* (2016) [Learning language games through interaction](http://arxiv.org/abs/1606.02447)
  - **TLDR**: Introduces a novel setting for interacting with computers to accomplish a task where only natural language can be used to communicate with the system!
* (2017) [Naturalizing a programming language via interactive learning](https://arxiv.org/abs/1704.06956)
  - **TLDR**: Very cool work allowing a community of workers to iteratively naturalize a language starting with a core set of commands in an interactive task. 

## Language Modelling
* (1996) [An Empirical Study of Smoothing Techniques for Language Modelling](https://aclweb.org/anthology/P96-1041)
  - **TLDR**: Performs an extensive survey of smoothing techniques in traditional language modelling systems.
* (2003) [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) :bulb:
  - **TLDR**: A seminal work in deep learning for NLP, introducing one of the earliest effective models for neural network-based language modelling. 
* (2014) [One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](https://arxiv.org/abs/1312.3005) :vhs:
  - **TLDR**: Introduces the Google One Billion Word language modelling benchmark. 
* (2015) [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
  - **TLDR**: Proposes a language model using convolutional neural networks that can employ character-level information, performing on-par with word-level LSTM systems. 
* (2016) [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
  - **TLDR**: Introduces a mega language model system using deep learning that uses a variety of techniques and significantly performs the SOTA on the One Billion Words Benchmark. 
* (2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) :bulb: :vhs:
  - **TLDR**: This paper introduces ELMO, a super powerful collection of word embeddings learned from the intermediate representations of a deep bidirectional LSTM language model. Achieved SOTA on 6 diverse NLP tasks. 
* (2018) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) :bulb:
  - **TLDR**: One of the most important papers of 2018, introducing BERT a powerful architecture pretrained using language modelling which is then effectively transferred to other domain-specific tasks.
* (2019) [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) :bulb:
  - **TLDR**: Generalized autoregressive pretraining method that improves upon BERT by maximizing the expected likelihood over all permutations of the factorization order.

## Miscellanea
* (1997) [Long Short-Term Memory](www.bioinf.jku.at/publications/older/2604.pdf) :bulb:
  - **TLDR**: Introduces the LSTM recurrent unit, a cornerstone of modern neural network-based NLP
* (2000) [Maximum Entropy Markov Models for Information Extraction and Segmentation](https://www.seas.upenn.edu/~strctlrn/bib/PDF/memm-icml2000.pdf) :bulb:
  - **TLDR**: Introduces Markov Entropy Markov models for information extraction, a commonly used ML technique in classical NLP. 
* (2010) [From Frequency to Meaning: Vector Space Models of Semantics](https://arxiv.org/pdf/1003.1141.pdf)
  - **TLDR**: A wonderful survey of existing vector space models for learning semantics in text. 
* (2012) [An Introduction to Conditional Random Fields](http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
  - **TLDR**: A nice, in-depth overview of conditional random fields, a commonly-used sequence-labelling model. 
* (2014) [Glove: Global vectors for word representation](https://nlp.stanford.edu/pubs/glove.pdf) :bulb: :vhs:
  - **TLDR**: Introduces Glove word embeddings, one of the most commonly used pretrained word embedding techniques across all flavors of NLP models
* (2014) [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](http://www.aclweb.org/anthology/P14-1023) 
  - **TLDR**: Important paper demonstrating that context-predicting distributional semantics approaches outperform count-based techniques.
* (2015) [Improving Distributional Similarity with Lessons Learned From Word Embeddings](https://www.aclweb.org/anthology/Q15-1016) :bulb:
  - **TLDR**: Demonstrates that traditional distributional semantics techniques can be enhanced with certain design choices and hyperparameter optimizations that make their performance rival that of neural network-based embedding methods. 
* (2018) [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
  - **TLDR**: Provides a smorgasbord of nice techniques for finetuning language models that can be effectively transferred to text classification tasks. 
* (2019) [Analogies Explained: Towards Understanding Word Embeddings](https://arxiv.org/pdf/1901.09813.pdf)
  - **TLDR**: Very nice work providing a mathematical formalism for understanding some of the paraphrasing properties of modern word embeddings.

# Deep learning Paper Reading Meeting Archive
This page is an archive of the Deep Learning Paper Reading Meeting.
If you would like to attend the meeting or have any questions,
 Write in the GitHub issue table or email us at 'tfkeras@kakao.com'
 
| Tasks | Paper | Link| Performance Index|
|:---------------|:-------------:|:-------------:|-------------:|
| NLP | Attention is all you need |[Youtube](https://www.youtube.com/watch?v=EyXehqvkfF0)<br> [Paper](https://arxiv.org/pdf/1706.03762.pdf)|NLP|
| NLP | BERT | [Youtube](https://www.youtube.com/watch?v=vo3cyr_8eDQ&t=7s) <br> [paper](https://arxiv.org/pdf/1810.04805.pdf)| NLP, Laguage representation|
| NLP | ERNIE| [Youtube](https://www.youtube.com/watch?v=_K6izzEeKYg&t=985s) <br> [paper](https://arxiv.org/pdf/1905.07129.pdf)| NLP, Laguage representation|
| NLP | RoBERTa | [Youtube](https://www.youtube.com/watch?v=GiotNYiTiMw&t=6s) <br> [paper](https://arxiv.org/pdf/1907.11692.pdf)| NLP, Laguage representation|
| NLP | XLNET | [Youtube](https://www.youtube.com/watch?v=29oxqDRaPNo&t=898s) <br> [paper](https://arxiv.org/pdf/1906.08237.pdf)| NLP, Laguage representation|
| NLP | SentenceBert |[Youtube](https://www.youtube.com/watch?v=izCeQOOuZpY&t=291s)| |
| NLP | Defending Against neural fake news | [Youtube](https://www.youtube.com/watch?v=cjjfJhYqyeg&t=10s) | |
| NLP | TransformerXL |[Youtube](https://www.youtube.com/watch?v=2SDI7hUoDSU&t=3s) <br> [blog](https://ghk829.github.io/whitecross.github.com//transformer-xl) | |
| NLP | Understanding back translation at scale | [Youtube](https://www.youtube.com/watch?v=htzBkroOLg4&t=12s) <br> [blog]( https://dev-sngwn.github.io/2020-01-07-back-translation/ ) ||
| NLP | Deep Contextualized Word Representations |[Youtube](https://www.youtube.com/watch?v=Vc13QVAKyGk)||
| NLP | Univiersal LM Fine-tuning for text classification | [Youtube](https://www.youtube.com/watch?v=ZJKtwX2LSbY&t=1173s)||
| NLP | Subword-level Word Vector Representations for Korean | [Youtube](https://www.youtube.com/watch?v=QR5XFn5rdMQ)|
| NLP | A Decomposable Attention Model for Natural Language Inference | [Youtube](https://youtu.be/8FcJtvCxI68)|
| NLP | Reformer | [Youtube](https://youtu.be/B6eLtGKgK68)
| NLP | Neural Machine Translation by Jointly Learning to Align and Translate | [Youtube](https://youtu.be/l9pWT6BHpj0)
| NLP | ELECTRA | [Youtube](https://youtu.be/BGRculoppT8)
| NLP | SBERT_WK | [Youtube](https://youtu.be/qXZ80xn8DDU)|
| NLP | Revealing the Dark Secrets of BERT | [Youtube](https://youtu.be/gcar30nhgqQ)|
| NLP | PEGASUS | [Youtube](https://youtu.be/JhGmeQBbDdA)|
| NLP | Document-level Neural Machine Translation with Inter-Sentence Attention | [Youtube](https://youtu.be/4QTydWc2xYs) |
| NLP |  Phrase-Based & Neural Unsupervised Machine | [Youtube](https://www.youtube.com/watch?v=G3rUzTFf_l4) |
| NLP | BART | [Youtube](https://youtu.be/VmYMnpDLPEo) |
| NLP | BAE | [Youtube](https://youtu.be/ukkcBtvPB3k) |
| NLP | A Generative Model for Joint Natural Language Understanding and Generation | [Youtube](https://youtu.be/Om0aAcZfjxE)|
| NLP | Learning Contextual Representations for Semantic Parsing with Generation-Augmented Pre-Training | [Youtube](https://youtu.be/L_5plaUpBaA)|
| NLP | Graph Attention Networks | [Youtube](https://youtu.be/shdNuppfClU)|
| NLP | Switch Transformers | [Youtube](https://youtu.be/82kpTjm-M_g)|
| NLP | DeText: A Deep Text Ranking Framework with BERT | [Youtube](https://youtu.be/tE_1uiaUf1k)|
| NLP | Face book Chat bot , Blender bot | [Youtube](https://youtu.be/wd64FDWCmDs)|
| NLP | Extracting Training Data from Large Language Models | [Youtube](https://youtu.be/NGoDUEz3tZg)|
| NLP | Longformer: The Long-Document Transformer | [Youtube](https://youtu.be/i7aiBMDExmA)|
| NLP | Visualizing and Measuring the Geometry of BERT | [Youtube](https://youtu.be/4DXU3MaGHcU)|
| NLP | Encode, Tag, Realize HighPrecision Text Editing | [Youtube](https://youtu.be/bs_GjHGV5T4)|
| NLP | multimodal transformer for unaligned multimodal language sequences | [Youtube](https://youtu.be/uEwxvQ9lXAQ)|
| NLP | SCGPT : Few-shot Natural Language Generation for Task-Oriented Dialog | [Youtube](https://youtu.be/BAZxrp2nrF8)|
| NLP | ColBERT: Efficient and Effective Passage Search viaContextualized Late Interaction over BERT | [Youtube](https://youtu.be/5mynfZA2t7U)|
| NLP | Restoring and Mining the Records ofthe Joseon Dynasty via Neural LanguageModeling and Machine Translation | [Youtube](https://youtu.be/BkyVMuvO5bE)|
| NLP | Improving Factual Completeness and Consistency of Image to Text Radiology Report Generation | [Youtube](https://youtu.be/OZLcZML7SO8)|
| NLP | FinBERT | [Youtube](https://youtu.be/FQN8sOi1PTI)|
| NLP | LayoutLM: Pre-training of Text and Layout for Document Image Understanding | [Youtube](https://youtu.be/3HqAgrcPprQ)|
| NLP | Query Suggestions as Summarization inExploratory Search | [Youtube](https://youtu.be/AVTZq2rWS0k)|
| NLP | H-Transformer-1D Paper : Fast One Dimensional Hierarchical Attention For Sequences | [Youtube](https://youtu.be/P19XuHOVyVk)|
| NLP | End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization | [Youtube](https://youtu.be/kmuET_G1brY)|
| NLP | DISEASES : Text mining and data integration of disease–gene associations | [Youtube](https://youtu.be/_2rfSxBSFnc)|
| NLP | RoFormer: Enhanced Transformer with Rotary Position Embedding | [Youtube](https://youtu.be/tRe2XHF6UbQ)|
| NLP | A Multiscale Visualization of Attention in the Transformer Model | [Youtube](https://youtu.be/Gl2WUBQYEfg)|
| NLP | CAST: Enhancing Code Summarization with Hierarchical Splitting and Reconstruction of Abstract Syntax Trees | [Youtube](https://youtu.be/h8YBJzuBSsA)|
| NLP | MERL:Multimodal Event Representation Learning in Heterogeneous Embedding Spaces | [Youtube](https://youtu.be/shnfzksjm1M)|
| NLP | Big Bird - Transformers for Longer Sequences | [Youtube](https://youtu.be/vV7fN1eUqbI)|
| NLP | Decoding-Enhanced BERT with Disentangled Attention | [Youtube](https://youtu.be/hNTkpNk7v-I)|
| NLP | SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based Sentiment Analysis | [Youtube](https://youtu.be/V9DlySYag_Q)|
| NLP | IMPROVING BERT FINE-TUNING VIA SELF-ENSEMBLE AND SELF-DISTILL ATION | [Youtube](https://youtu.be/3PVada1CVYQ)|
| NLP | ACHIEVING HUMAN PARITY ON VISUAL QUESTION ANSWERING | [Youtube](https://youtu.be/Gcbf0M0Qx9U)|
| NLP | Deep Encoder, Shallow Decoder: Reevaluating non- autoregressive machine translation | [Youtube](https://youtu.be/sP7Ue_MbiKE)|
| NLP | LaMDA : Language Models for Dialog Applications | [Youtube](https://youtu.be/re2AiBnFGGk)|
| Vision | YOLO | [Youtube](https://www.youtube.com/watch?v=Ae-p7QVOdbA&t=285s) <br> [paper](https://arxiv.org/pdf/1506.02640.pdf) |Object detection|
| Vision  | YOLO-v2 |[Youtube](https://www.youtube.com/watch?v=9FiGYp6khxo&t=8s) | |
| Vision  | Resnet | [Youtube](https://www.youtube.com/watch?v=JI5kXF_OUkY&t=125s) <br> [paper](https://arxiv.org/pdf/1512.03385.pdf) |Image classification|
| Vision  | GAN | [Youtube](https://www.youtube.com/watch?v=UZpuIG1eF8Y&t=147s) ||
| Vision | Image Style Transfer Using CNN | [Youtube](https://www.youtube.com/watch?v=8jS0xxslTco&t=905s) | |
| Vision  | SINGAN | [Youtube](https://www.youtube.com/watch?v=pgYIuA4O95E) | |
| Vision  | FCN | [Youtube](https://www.youtube.com/watch?v=_52dopGu3Cw) | |
| Vision | DeepLabV3| [Youtube](https://youtu.be/TjHR9Z9iNLA)
| Vision | Unet | [Youtube](https://www.youtube.com/watch?v=evPZI9B2LvQ&t=9s) <br> [paper](https://arxiv.org/pdf/1411.1792.pdf)| 
| Vision | CyCADA | [Youtube](https://youtu.be/DODYdEwebTg)
| Vision | D-SNE | [Youtube](https://youtu.be/OJe9SgS-GM8)
| Vision | Faster-RCNN| [Youtube](https://youtu.be/HmJWvwIpW5g)
| Vision | Weakly Supervised Object DetectionWith Segmentation Collaboration| [Youtube](https://youtu.be/qvWf0aIqaLE)
| Vision | Don't Judge an Object by Its Context: Learning to Overcome Contextual Bias| [Youtube](https://youtu.be/xTm1gWWSEHM) ||
| Vision | data efficient image recognition with contrastive predictive coding| [Youtube](https://youtu.be/-LzFAqOnfTo)|
| Vision | Deep Feature Consistent Variational Autoencoder| [Youtube](https://youtu.be/Iy0zCVZBO_A)|
| Vision | Attention Branch Network: Learning of Attention Mechanism for Visual Explanation| [Youtube](https://youtu.be/8BAGbC0HCVg)|
| Vision | RELATION-SHAPE CONVOLUTIONAL NEURAL NETWORK FOR POINT CLOUD ANALYSIS| [Youtube](https://youtu.be/E_odPJHLW7Y)|
| Vision | EfficientNet| [Youtube](https://youtu.be/Vy0BvuFSNxQ) |
| Vision | Deep Clustering for Unsupervised Learning of Visual Features| [Youtube](https://youtu.be/cCwzxVwfrgM)|
| Vision | Boosting Few-shot visual learning with self-supervision| [Youtube](https://youtu.be/6ZrXjdMfqxk)|
| Vision | Rethinking Pre-training and Self-training| [Youtube](https://youtu.be/d8EDoHDEgvI) |
| Vision | BYOL : Bootstrap Your Own Latent| [Youtube](https://youtu.be/BuyWUSPJicM) |
| Vision | Deep Image Prior| [Youtube](https://youtu.be/wvc_JX4WUHo) |
| Vision | Object-Centric Learning with Slot Attention| [Youtube](https://youtu.be/dMGFQ_ISdFg) |
| Vision | Yolo V4| [Youtube](https://youtu.be/D6mj_T8K_bo) |
| Vision | Dynamic Routing Between Capsules|[Youtube](https://youtu.be/aH7Hn-Ca_uk) |
| Vision | Semi-Supervised Classification with Graph Convolutional Network|[Youtube](https://youtu.be/Ft2Q8WQ8ETM) |
| Vision | Generative Pretraining from Pixels|[Youtube](https://youtu.be/QC9VWEv7qrw) |
| Vision | MaskFlownet | [Youtube](https://youtu.be/8J3_BeVoCUg) |
| Vision | Adversarial Robustness through Local Linearization| [Youtube](https://youtu.be/-h2W5-A8qNU) |
| Vision | Locating Objects Without Bounding Boxes| [Youtube](https://youtu.be/TkcRI31v5_I) |
| Vision | Training data-efficient image transformers & distillation through attention| [Youtube](https://youtu.be/LYYxv9mv5qw) |
| Vision | What Makes Training Multi-modalClassification Networks Hard?| [Youtube](https://youtu.be/ZjDRVgA9F1I) |
| Vision | 2020 CVPR Meta-Transfer Learning for Zero-Shot Super-Resolution| [Youtube](https://youtu.be/lEqbXLrUlW4) |
| Vision | 2020 ACCV Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation| [Youtube](https://youtu.be/RrGVDQLEnq4) |
| Vision | Style GAN| [Youtube](https://youtu.be/YGQTzYNIL0s) |
| Vision | HighPerformance Large Scale ImageRecognition Without Normalization| [Youtube](https://youtu.be/HP4evlugOIo) |
| Vision | Focal Loss for Dense Object Detection| [Youtube](https://youtu.be/d5cHhLyWoeg) |
| Vision | Editing in Style : Uncovering the Local Semantics of GANs| [Youtube](https://youtu.be/2Gx3_0xpNvM) |
| Vision | Efficient Net 2| [Youtube](https://youtu.be/uLKqMbOA_vU) |
| Vision | Style Clip| [Youtube](https://youtu.be/5FwzEP3bYLg) |
| Vision | Swin Transformer| [Youtube](https://youtu.be/L3sH9tjkvKI) |
| Vision | NBDT : Neural-backed Decision Tree| [Youtube](https://youtu.be/MdDAug75J6s) |
| Vision | [2020 CVPR] Efficient DET | [Youtube](https://youtu.be/Mq4aqDgZ2bI) |
| Vision | MLP - MIXER : An all-MLP Architecture for Vision | [Youtube](https://youtu.be/L3vEetyNG_w) |
| Vision | You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection | [Youtube](https://youtu.be/hqRSw_Bu24w) |
| Vision | Video Prediction ! Hierarchical Long-term Video Frame Prediction without Supervision | [Youtube](https://youtu.be/q15QO9LXYlI) |
| Vision | Closed-Form Factorization of Latent Semantics in GANs | [Youtube](https://youtu.be/TKmIalR5x8I) |
| Vision | YOLOR : You Only Learn One Representation: Unified Network for Multiple Tasks |[Youtube](https://youtu.be/bQgQkhaGZG8) |
| Vision | StyleSpace Analysis |[Youtube](https://youtu.be/JcIe5U5PmLQ) |
| Vision | Representative graph neural network |[Youtube](https://youtu.be/z-UUq8x1oRw) |
| Vision | YOLOX |[Youtube](https://youtu.be/N2rLSzEqqI8) |
| Vision | Joint Contrastive Learning with Infinite Possibilities |[Youtube](https://youtu.be/0NLq-ikBP1I) |
| Vision | Auto Deep Lab - Hierarchical Neural Architecture Search for Semantic Image Segmentation |[Youtube](https://youtu.be/2886fuyKo9g) |
| Vision | Explaining in style training a gan to explain a classifier in stylespace |[Youtube](https://youtu.be/pkbLrLSDQ9Q) |
| Vision | End-to-End Semi-Supervised Object Detection with Soft Teacher |[Youtube](https://youtu.be/7gza1VFjb0k) |
| Vision | Understanding Dimensional Collapse in Contrastive Self Supervised Learning |[Youtube](https://youtu.be/dO-gD54OlO0) |
| Vision | Encoding in Style: a Style Encoder for Image-to-Image Translation |[Youtube](https://youtu.be/2QXQZHvx6Ds) |
| Vision | Detection in Crowded Scenes: One Proposal, Multiple Predictions |[Youtube](https://youtu.be/LPC4m66YZfg) |
| Vision | A Normalized Gaussian Wasserstein Distance for Tiny Object Detection |[Youtube](https://youtu.be/eGKlg4sZ0Zw) |
| Vision | Siamese Neural network for one-shot image recognition |[Youtube](https://youtu.be/SthmLerAeis) |
| Vision | Grounded Language-Image Pre-training |[Youtube](https://youtu.be/krP3t31fWvI) |
| Vision | Transfer Learning for Pose Estimation of Illustrated Characters |[Youtube](https://youtu.be/wo3Ob174AfY) |
| Vision | Sparse - RCNN paper explained |[Youtube](https://youtu.be/EvPMwKALqvs) |
| Recommend System | Matrix Factorization Technique for Recommender System | [Youtube](https://www.youtube.com/watch?v=Z49JNxS4vsc&t=260s) <br> [paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)| Recommendation system |
| Recommend System| Collaborative Filtering for Implicit Feedback Dataset | [Youtube](https://www.youtube.com/watch?v=ePvzTeLOBi4&t=6s) | |
| Speech | A comparison of S2S models for speech recognition | [Youtube](https://www.youtube.com/watch?v=fltpFsNL8TA&t=463s)  <br> [paper](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF) | Speech Recognition|
| Fundamental | RAdam | [Youtube](https://www.youtube.com/watch?v=_F5_hgX_lSE) <br> [blog](https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html) <br> [paper](https://arxiv.org/pdf/1908.03265.pdf)| Regularization|
| Fundamental | Stacked Auto Encoder for the P300 Component Detection | [Youtube](https://www.youtube.com/watch?v=ydpZaS1CCRg) |
| Fundamental | A survey on Image Data Augmentation for DL | [Youtube](https://www.youtube.com/watch?v=TioeCk3yMCo&t=1073s) <br>[paper](https://link.springer.com/content/pdf/10.1186%2Fs40537-019-0197-0.pdf) | Data augmentation|
| Fundamental | Training Confidence-calibrated classifiers for detecting out of distribution samples | [Youtube](https://www.youtube.com/watch?v=NOzDB2Rpbi0&t=150s) |
| Fundamental | AdamW | [Youtube](https://youtu.be/-Sd_zH_LHBo) <br> [blog](https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html) ||
| Fundamental | Stargan | [Youtube](https://youtu.be/KO_mOGKdxOw) ||
| Fundamental | Drop-out | [Youtube](https://www.youtube.com/watch?v=hOgQDK2-lA8) ||
| Fundamental | BLEU - a Method for Automatic Evaluation of Machine Translation | [Youtube](https://youtu.be/61my462pZk0) | |
| Fundamental | t-SNE| [Youtube](https://youtu.be/zCYKD3YfcSM) ||
| Fundamental | Gpipe| [Youtube](https://youtu.be/bbb0bLR0Faw) ||
| Fundamental | explainable ai| [Youtube](https://youtu.be/1WeLdfhRocI) | |
| Fundamental | TAPAS| [Youtube](https://youtu.be/V2hPGrPqR0U) ||
| Fundamental | Learning both Weights and Connections for Efficient Neural Networks| [Youtube](https://youtu.be/Gt2gvhcsPD8) ||
| Fundamental | ReVACNN| [Youtube](https://youtu.be/EBaMig0nMoI) ||
| Fundamental | THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS| [Youtube](https://youtu.be/EBaMig0nMoI) | |
| Fundamental | ALPHAGO : Mastering the game of Go with Deep Neural Networks and Tree Search| [Youtube](https://youtu.be/huMitou6zDs) |
| Fundamental | A_BASELINE_FOR_FEW_SHOT_IMAGE_CLASSIFICATION| [Youtube](https://youtu.be/qpCpo7wATto) |
| Fundamental |  Sharp Minima Can Generalize For Deep Nets| [Youtube](https://youtu.be/5E9SFe5WU1s) |
| Fundamental |  Pediatric Sleep Stage Classification  Using Multi-Domain Hybrid Neural Networks | [Youtube](https://youtu.be/mp72ClQT40s) |
| Fundamental |  Pruning from Scratch| [Youtube](https://youtu.be/ZBAhBHbXg40) |
| Fundamental |  Do We Need Zero Training Loss After Achieving Zero Training Error?| [Youtube](https://youtu.be/7HkFFFJar_E) |
| Fundamental |  Deep Recurrent Q-Learning for Partially Observable MDPs| [Youtube](https://youtu.be/M6hjaQSXEcE) |
| Fundamental |  Large Margin Deep Networks for Classification| [Youtube](https://youtu.be/Wl4kex_ZLqo) |
| Fundamental |  generating wikipedia by summarizing long sequences| [Youtube](https://youtu.be/C2xr5IA-4CM) |
| Fundamental |  Plug and Play Language Models: A Simple Approach to Controlled Text Generation| [Youtube](https://youtu.be/QnsCHRCWEP4) |
| Fundamental | What Uncertainties Do We Need in Bayesian DeepLearning for Computer Vision?|[Youtube](https://youtu.be/d7y42HfE6uI) |
| Fundamental | KRED |[Youtube](https://youtu.be/Xq_FmQ-Sy1U) |
| Fundamental | Early Stopping as nonparametric Variational  |[Youtube](https://youtu.be/q5AxUQr9KBg) |
| Fundamental | Sharpness Aware Minimization for efficeintly improving generalization   |[Youtube](https://youtu.be/iC3Y85W5tmM) |
| Fundamental | Neural Graph Collaborative Filtering   |[Youtube](https://youtu.be/ce0LrvVblCU) |
| Fundamental | Restricting the Flow: Information Bottlenecks for Attribution   |[Youtube](https://youtu.be/eUuXgkzR9MQ) |
| Fundamental | Real world Anomaly Detection in Surveillance Videos   |[Youtube](https://youtu.be/DYnvX5RaUL0) |
| Fundamental |  Deep learning model to 2Bit Quantization?! BRECQ Paper review (2021 ICLR)   |[Youtube](https://youtu.be/aT0Fv1PzyV8) |
| Fundamental | Deep sets (2017 NIPS)   |[Youtube](https://youtu.be/EIZ3z823JQU) |
| Fundamental | StyleGAN2   |[Youtube](https://youtu.be/XMZWeqx5Vgg) |
| Fundamental | SOTA - Beyond Synthetic Noise:Deep Learning on Controlled Noisy Labels  |[Youtube](https://youtu.be/s83wjMtdQh8) |
| Fundamental | Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems   |[Youtube](https://youtu.be/kTT7YsHWodU) |
| Fundamental | Longformer: The Long-Document Transformer   |[Youtube](https://youtu.be/i7aiBMDExmA) |
| Fundamental | soft actor critic   |[Youtube](https://youtu.be/HK7Y20Bt7qM) |
| Fundamental | Loss Function Discovery for Object Detection Via Convergence- Simulation Driven Search   |[Youtube](https://youtu.be/P_yXwbPefQ8) |
| Fundamental | [2021 ICLR] The Deep Bootstrap Framework:Good Online Learners are good Offline Generalizers    |[Youtube](https://youtu.be/WwXzLCmWvqM) |
| Fundamental | Meta HIN  |[Youtube](https://youtu.be/v8bma8QMK7k) |
| Fundamental | When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations  |[Youtube](https://youtu.be/gB_YmwC1AY0) |
| Fundamental | Self similarity Student for Partial Label Histopathology Image Segmentation  |[Youtube](https://youtu.be/2Kw-xgpHTqY) |
| Fundamental | ANALYSING MATHEMATICAL REASONING ABILITIES OF NEURAL MODELS  |[Youtube](https://youtu.be/jE1gJQH5OJI) |
| Fundamental | Self-training Improves Pre-training for Natural Language Understanding |[Youtube](https://youtu.be/9iJLzmrUN-8) |
| Fundamental | Preference Amplification in Recommender Systems |[Youtube](https://youtu.be/dM3kDjpSzBk) |
| Fundamental | Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation |[Youtube](https://youtu.be/UO5iqW3iTFU) |
| Fundamental | Evaluating Classifiers by Mean of Test Data with Noisy Labels |[Youtube](https://youtu.be/xByR5oix9ms) |
| Fundamental | Progressive Identification of True Labels for Partial-Label Learning |[Youtube](https://youtu.be/QsvgzjfSFhg) |
| Fundamental | Fine-grained Interest Matching For Neural News Recommendation |[Youtube](https://youtu.be/XW93QvbFlaQ) |
| Fundamental | Adversarial Reinforced Learning for Unsupervised Domain Adaptation |[Youtube](https://youtu.be/zeBMMXKj39U) |
| Fundamental | Neural Tangent Kernel - Convergence and Generalization in neural Network |[Youtube](https://youtu.be/vDQWwOqQ7mo) |
| Fundamental | Intriguing Properties of Contrastive Losses |[Youtube](https://youtu.be/uzsI-dEoK2c) |
| Fundamental | Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets |[Youtube](https://youtu.be/mcnSN645xUE) |
| Fundamental | Transformer Interpretability Beyond Attention Visualization |[Youtube](https://youtu.be/XCED5bd2WT0) |
| Fundamental | How does unlabeled data improve generalization in self-training? |[Youtube](https://youtu.be/t7dY-k-JBPA) |
| Fundamental | Rainbow: Combining Improvements in Deep Reinforcement Learning |[Youtube](https://youtu.be/oC1AOIefjT8) |
