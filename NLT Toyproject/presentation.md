# Naver_myplace review Emotion analysis (toy project)
<img src = "01.png">

## 00. Naver_myplace
>- 실제 방문한 음식점, 카페등을 리뷰하는 플랫폼
   
## 01. Dataset 개요
>- 리뷰 의 수 : 약 7200건
>- 음식점의 수 : 약 320곳

## 02. Toy project의 이유
>- 한국어 감정 분석을 연습하기 위해

## 03. Projec 내용
>- Naver_myplace의 리뷰를 크롤링하여, 별점 3점 이하는 negative, 이상은 positive로 기준치를 잡음
>- 이후 한국어 형태소 분석기를 사용하여 LGBM, Randomforest, DecisionTree를 이용하여 분류를 진행하였고 Valdation Accurucy는 LGMB이 약 0.9 정도로 높은 확률을 보였다.
>- 테스트에서는 0.82정도로 높지 않은 수치를 보였으며, 실제 문장을 넣고 확인해본 결과 negative를 positive로 예측하는 등의 좋지 않은 성능을 보였다.
>- 다만, 다른 문장 Ex ('싫어요', '별로에요', '맛 없어요', '친절해요')은 그런대로  negative와 positive를 구별해 내었다 (LGBM 기준)

## 04. 프로젝트 회고
>-  FastCampas에서 데이터 사이언티스트 스쿨 과정 중 Machine Learning Project를 진행하기에 앞서, 자연어 처리를 위해 연휴 기간중 진행해본 toyproject였다. 
>- positive와 negative의 비율이 약 8.5 : 1.5로 많은 불균형이 있었으며, 이는 그냥 모든 테스트 데이터에 positive이라고 예측하여도 0.85의 예측력을 가지는 문제가 있다.
>- 물론 학습과정에서 불균형 데이터는 over_fitting하여 맞추어 주었으나, 그럼에도 불구하고 전체 데이터의 갯수가 많지 않음으로 인해 높은 예측력을 가지진 못하였다.
>- myplace의 리뷰 크롤링에서도 랜덤하게 에러를 띄우거나, 인터넷망이 느려짐으로 인해 겪는 오류들로 인해 크롤링이 쉽지 않았던점도 어려운 점으로 들수 있겠다.
>- 추후 천천히 데이터를 더욱 수집하여 (다른곳의 리뷰 등) 발전시켜봐야 겠다.

## 05.Code
>- 
>-