##  “BELEBELE”라는 122개 언어에 대한 읽기 이해 평가를 위한 데이터셋을 공개

다국어 모델의 텍스트 이해 능력을 평가하기 위한 벤치마크 테스트셋으로, 언어마다 동일한 데이터셋이 있어 모델의 성능을 모든 언어에 대해 직접적으로 비교 할 수 있습니다. 
단일 언어는 물론, 다국어 및 교차 언어 모델까지 평가가 가능합니다.

더불어, 균형잡힌 다국어 데이터를 이용해 사전 훈련된 언어 모델이 더 많은 언어를 이해하는 데 있어 큰 영어 중심의 언어 모델을 능가한다는 것을 발견하네요.

그동안 언어 모델들에 대해 다국어 평가, 특히 한국어에 대한 평가를 정확히 할 수 있는 평가셋이 없어서 성능 비교가 어려웠는데요. MRC 뿐이긴 해도, 이제 이 데이터셋을 이용해 한국어 모델들의 성능을 조금은 더 객관적으로 살펴 볼 수 있습니다. 
- 논문: https://arxiv.org/pdf/2308.16884.pdf
- 코드: https://github.com/facebookresearch/belebele

## RedPajama-Data-1T 데이터셋을 공개
    * LLaMA 논문에 설명된 레시피에 따라서 생성한 1.2조개의 토큰으로 구성된 완전 개방형 데이터 셋
    * HuggingFace를 통해 다운로드 가능. 전체 5TB(3TB로 압축하여 배포)
    * 7개의 데이터 조각으로 구성 : 각각 전처리와 필터링하여 LLaMA 논문과 비슷한 갯수로 구성(전처리 방법 및 필터 역시 GitHub에 공개)
        * CommonCrawl (878b) - 웹 크롤링 데이터
        * C4 (175b) - Colossal, Cleaned version of Common Crawl
        * GitHub (59b) - 라이센스와 품질로 필터링된 GitHub의 데이터
        * arXiv (28b) - 과학 논문과 기사들 (boilerplate 제거)
        * Books (26b) - 콘텐츠 유사성에 따라서 중복을 제거한 공개 서적 Corpus
        * Wikipedia (24b) - 위키피디어의 일부 페이지들 (boilerplate 제거)
        * StackExchange (20b) - 스택익스체인지의 일부 페이지들 (boilerplate 제거)
        
## CVPR 2021 datasets
 - https://cvpr2021.thecvf.com

## CVPR 2022 datasets
 - https://cvpr2022.thecvf.com/dataset-contributions
   
## CVPR 2023 datasets
 - https://cvpr2023.thecvf.com 
