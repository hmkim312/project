# Naver_myplace review Emotion analysis (toy project)
<img src = "https://user-images.githubusercontent.com/60168331/81460645-aaf3e080-91e1-11ea-8ed6-e86192db5ace.png">

## 00. Naver_myplace
>- 실제 방문한 음식점을 영수증 기반으로 인증하여 리뷰를 작성하는 플랫폼
   
## 01. Dataset 개요
>- 리뷰 의 수 : 약 8000건
>- 음식점의 수 : 약 350곳

## 02. Toy project의 진행 이유
>- FastCampus의 Machin Learning project를 진행하기에 앞서 자연어 분석에 대해 공부하고자 연휴기간중 (2020년 4월 30일 ~ 5월 5일)에 진행하였음

## 03. Projec 내용
>- Naver_myplace의 리뷰를 셀레니움을 이용하여 크롤링하였으며, 별점 3점 이하는 negative, 이상은 positive로 기준치를 잡음
>- 이후 한국어 형태소 분석기인 konlpy를 사용하여 형태소 분석을 하였음
>- LGBM, Randomforest, DecisionTree를 이용하여 분류를 진행하였고 Valdation Accuracy는 LGMB이 약 0.9 정도로 높은 수치를 보였음
>- 테스트에서는 0.82정도로 높지 않은 Accuracy 수치를 보였으며, test data의 문장을 넣고 확인해본 결과 negative를 positive로 예측하는 등의 좋지 않은 성능을 보였다.
>- 다만, 다른 문장 Ex ('싫어요', '별로에요', '맛 없어요', '친절해요')은 그런대로  negative와 positive를 구별해 내었다 (LGBM 기준)

## 04. 프로젝트 회고
>-  FastCampas에서 데이터 사이언티스트 스쿨 과정 중 Machine Learning Project를 진행하기에 앞서 자연어 처리에 익숙해 지기 위해 연휴 기간 중 혼자 진행해본 toyproject였다. 
>- positive와 negative의 비율이 약 8.5 : 1.5로 많은 불균형이 있었으며, 이는 그냥 모든 테스트 데이터에 positive이라고 예측하여도 0.85의 예측력을 가지는 문제가 있다.
>- 물론 학습과정에서 불균형 데이터는 over_fitting하여 맞추어 주었으나, 그럼에도 불구하고 전체 데이터의 갯수가 많지 않음으로 인해 높은 예측력을 가지진 못하였다.
>- myplace의 리뷰 크롤링에서도 랜덤하게 에러를 띄우거나, 인터넷망이 느려짐으로 인해 겪는 오류들로 인해 크롤링이 쉽지 않았던점도 어려운 점으로 들수 있겠다.
>- 추후 천천히 데이터를 더욱 수집하여 (다른곳의 리뷰 등) 발전시켜봐야 겠다.

## 05.Code
#### NLP
>- https://github.com/hmkim312/project/blob/master/NLP%20Toyproject/project01_Review_sentiment_analysis_by_naver_myplace.ipynb
#### Crawling
>- https://github.com/hmkim312/project/blob/master/NLP%20Toyproject/crawling03_naver_myplace_review_crawling.ipynb