# 한국어 자연어 처리 
## 텐서플로우와 머신러닝으로 시작하는 자연어처리(로지스틱회귀부터 트랜스포머 챗봇까지)
  - NLP-KR/tensorflow-ml-nlp (https://github.com/NLP-kr/tensorflow-ml-nlp)
  * **준비 단계** - 자연어 처리에 대한 배경과 개발에 대한 준비를 위한 챕터입니다.
      - 1. [들어가며](./1.Intro)
      - 2. [자연어 처리 개발 준비](./2.NLP_PREP)
      - 3. [자연어 처리 개요](./3.NLP_INTRO)
  * **자연어 처리 기본** - 자연어 처리에 기본적인 모델에 대한 연습 챕터입니다.
      - 4. [텍스트 분류](./4.TEXT_CLASSIFICATION)
      - 5. [텍스트 유사도](./5.TEXT_SIM)
  * **자연어 처리 심화** - 챗봇 모델을 통해 보다 심화된 자연어 처리에 대한 연습 챕터입니다.
      - 6. [챗봇 만들기](./6.CHATBOT)
  * 저자 (Authors)
      - ChangWookJun / @changwookjun (changwookjun@gmail.com)  
      - Taekyoon  / @taekyoon (tgchoi03@gmail.com)  
      - JungHyun Cho  / @JungHyunCho (reniew2@gmail.com)  

## 송영숙님 쳇봇 데이터
  - songys/Chatbot_data (https://github.com/songys/Chatbot_data)
  - Chatbot_data_for_Korean v1.0 
  - License : MIT
  * Data description.     
      - 1. 챗봇 트레이닝용 문답 페어 11,876개           
      - 2. 일상다반서 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링  

#### Natural Language Processing (자연어 처리)

* Single Headed Attention RNN: Stop Thinking With Your Head (2020)
    * [Original Paper Link](https://arxiv.org/abs/1911.11423) / Paper Review Video / Summary PDF / Code Practice
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019)
    * [Original Paper Link](https://arxiv.org/abs/1810.04805) / Paper Review Video / Summary PDF / Code Practice
* Attention is All You Need (NIPS 2017)
    * [Original Paper Link](https://arxiv.org/abs/1706.03762) / Paper Review Video / [Summary PDF](/lecture_notes/Transformer.pdf) / [Code Practice](/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb)
* Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015 Oral)
    * [Original Paper Link](https://arxiv.org/abs/1409.0473) / Paper Review Video / Summary PDF / [Code Practice](/code_practices/Sequence_to_Sequence_with_Attention_Tutorial.ipynb)
* Sequence to Sequence Learning with Neural Networks (NIPS 2014)
    * [Original Paper Link](https://arxiv.org/abs/1409.3215) / [Paper Review Video](https://www.youtube.com/watch?v=4DzKM0vgG1Y) / [Summary PDF](/lecture_notes/Seq2Seq.pdf) / [Code Practice](/code_practices/Sequence_to_Sequence_with_LSTM_Tutorial.ipynb)


|Q | A | Label|
| -- | -- | ---- |
|12시 땡!|하루가 또 가네요.|0|
|1지망 학교 떨어졌어|위로해 드립니다.|0|
|3박4일 놀러가고 싶다|여행은 언제나 좋죠.|0|
|3박4일 정도 놀러가고 싶다|여행은 언제나 좋죠.|0|
|PPL 심하네|눈살이 찌푸려지죠.|0|
|SD카드 망가졌어|다시 새로 사는 게 마음 편해요.|0|
|SD카드 안돼|다시 새로 사는 게 마음 편해요.|0|
|1000일 만난 여자친구와 이별|더 오래 만날 사람 만날 거예요.|1|
|10년 연애. 헤어졌습니다.|더 공허함이 크시겠네요.|1|
|10년 연애사 되돌아보니 다 부질없네|더 좋은 사람 만나실 거예요.|1|
|10년 연예의끝|더 마음이 허하겠어요.|1|
|내가 사랑하는 사람이 나를 사랑했으면|그 사람도 당신을 사랑하고 있을지도 몰라요.|2|
|내가 사랑해도 될까?|당연하죠.|2|
|내가 사랑했던 그 사람 맞아?|사랑했던 그 사람의 다른 모습이 보인거예요. 그 사람 맞아요.|2|
|내가 생각하는 만큼 걔도 날 생각할까?|확신이 없나보네요.|2|
|내가 생각했던 사람이 맞는지 확신이 안들어|새로운 모습을 봤나봐요.|2|
