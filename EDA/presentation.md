# Instacart dataset EDA project
<img src = "https://user-images.githubusercontent.com/60168331/79730764-448b4900-832c-11ea-83f9-663fdfcec1cd.png">

## 00. instarcart 개요
>- '식품계의 우버'라 불리는 미국의 신선식품 대발 서비스 스타트업으로, 국내의 마켓컬리와 유사하다.
>- instarcart는 월마트, 세이프웨이, 코스트코 등 대형마트부터 지역의 슈퍼마켓의 제품을 대신 구매하여 배달하여 주는 서비스를 제공한다.
>- 설립 2년만에 유니콘을 달성하였고, 8년간 16억 달러 (약 1조 8800억원)을 투자 유치 받았다.
   

## 01. Dataset 개요
>- 총 유저수 : 206,209(약 20만 명)
>- 판매된 총 제품수 : 33,819,106(약 3380만 건)
>- 품목 수 : 49,688(약 5만 종)
>- 총 주문량 : 3,421,083(약 340만 건)
>- 품목당 평균 판매 횟수 : 약 680건
>- 유저 1명당 평균 주문 횟수 : 16.5회
>- 재구매율 : 약 59%

## 02. EDA의 선택 이유
>- 충분한 데이터의 양
>- 실제 생활과 밀접한 데이터

## 03. Data Columns
### 시간 Columns
>- order_dow : 주문한 요일
>- order_hour_of_day : 주문한 시간

### 제품 Columns
>- product_id: 제품 ID
>- product_name: 제품 이름
>- aisle_id: 소분류 ID
>- aisle: 소분류 이름
>- department_id: 대분류 ID
>- department: 대분류 이름

### 주문 Columns
>- user_id: 고객 ID
>- order_id: 주문 ID
>- add_to_cart_order: 각 제품이 장바구니에 추가 된 순서
>- reordered: 사용자가 해당 제품을 과거에 주문한 경우 1, 그렇지 않으면 0

### etc
>- days_since_prior: 마지막 주문 이후 일 수, 30 일 제한 (NA는 order_number1) >재주문 기간의 텀
>- aisle_id : foreign key
>- department_id: foreign key
>- eval_set: 이 순서가 속하는 평가 세트
>- order_number: 이 사용자의 주문 순서 번호 (1 = 첫 번째, n = n 번째)

## 04. 가설
### 이용자들은 어떤 물건을 재구매할 확률이 높을까?
>- 가설 1 : 시간 및 요일등에 따라 품목과 재구매 정도의 차이가 있을것이다.
>- 가설 2 : 판매량이 높은 제품이 재구매도 많을 것이다.
>- 가설 3 : 장바구니의 첫번째 담은 물건의 재구매의 확률이 높을 것이다.

## 05. 가설 1 검증 : 시간 및 요일등에 따라 품목과 재구매 정도의 차이가 있을것이다.
### 시간 및 요일별 구매 횟수 
<img src = "https://user-images.githubusercontent.com/60168331/79730606-0aba4280-832c-11ea-8a9f-1992976016bf.png">

### 시간 및 요일별 재구매 횟수 
<img src = "https://user-images.githubusercontent.com/60168331/79730615-0beb6f80-832c-11ea-8e68-16b0e9b583b0.png">

>- 09시 ~ 17시 사이에 주문이 집중되는것을 볼수 있었다.
>- 토,일요일에 주문이 집중(0,1)된다.
>- 시간, 요일에 따른 구매횟수와 재주문 횟수는 비슷한 빈도를 보이는것으로 확인되었다.

### 구매가 가장 많이 일어나는 토요일, 일요일의에는 무슨 품목이 많이 팔릴까?
<img src = "https://user-images.githubusercontent.com/60168331/79730619-0beb6f80-832c-11ea-9a5c-ccdfcd1f99ec.PNG">
<img src = "https://user-images.githubusercontent.com/60168331/79730663-1f96d600-832c-11ea-9d01-b7fc75cdd87d.png">

>- 토요일과 일요일은 사람들이 바나나와 딸기, 아보카도, 레몬 등 식품을 많이 구매한다.

### 요일별 구매상품은 어떤게 많이 팔릴까?
<img src = 'https://user-images.githubusercontent.com/60168331/79730685-24f42080-832c-11ea-9047-49ec8198051f.png'>
<img src = 'https://user-images.githubusercontent.com/60168331/79730693-29203e00-832c-11ea-9099-9ed3fe1e84b9.png'>

>- 항상 바나나가 많이 팔리고, 레몬, 아보카도, 시금치, 딸기 등이 팔리는 것을 볼수 있다.
>


### ※가설 1의 결론:
>-  토,일요일에는 바나나(유기농), 딸기, 시금치, 아보카도 등이 많이 팔린다
>- 요일, 시간별 구매횟수와 재구매 횟수는 비슷한 분포를 보인다.

## 06. 가설 2 검증 : 판매량이 높은 제품은 재구매도 많을 것이다.
### 몇개가 팔려야 판매량이 높은 제품일까?
>- 1000개 이상 팔린 제품의 총 갯수는 26,503,548개
>- 총 팔린 제품의 갯수는 32,434,489개
>- 한 제품이 1000개 이상 팔리면 전체 품목에서 81.71%를 차지한다.

### 그렇다면 1000개 이상 팔린 제품은 처음구매와 재구매는 어떤 비율을 가질까
<img src = 'https://user-images.githubusercontent.com/60168331/79730695-2a516b00-832c-11ea-98c7-01bec7b80d37.png'>

### 1000개 이상 판매된 제품은 재구매율과 어떤 관계를 가질까?
<img src = 'https://user-images.githubusercontent.com/60168331/79730706-2de4f200-832c-11ea-996d-c5f5b3c185ce.png'>

>- 완전한 선형은 아니지만, 1000개이상 판매된 제품은 재구매율도 높은것으로 보인다.
>- 제품의 갯수가 많아서 잘 안보여진것일까? 제품의 소분류로도 확인해보자

### Aisle(제품의 소분류)별로 재구매율 확인해보기
<img src = 'https://user-images.githubusercontent.com/60168331/79731130-cb402600-832c-11ea-89ad-7ecef41775b5.png'>

>- 제품을 소분류한 컬럼으로 확인하였을때 확실히 이전보다 양의 상관관계를 보이는것으로 파악된다.

### ※가설2 결론
>- 그래프를 그려본 결과 1000개이상 팔린 제품과, 그 제품의 소분류는 재구매율과 상관관계를 가지는것으로 확인 된다.

## 07. 가설 3 : 장바구니에 첫번째 담은 물건은 재구매의 확률이 높을 것이다.
### 전체 데이터에서 총 주문의 갯수가 1개인 것은 제외
<img src = 'https://user-images.githubusercontent.com/60168331/79730720-35a49680-832c-11ea-85fb-6516de153544.PNG'>

>- 한번 주문시 주문하였을때의 갯수가 1개인것은 항상 장바구니에 첫번째에 담기게 됨으로, 알아보려고하는것과 어긋나는 부분이 있어, 전처리 과정에서 제외 시켰다.


### 장바구니에 처음 담긴 제품의 재구매율 상위 5개를 확인
<img src = 'https://user-images.githubusercontent.com/60168331/79730721-363d2d00-832c-11ea-89fd-e29e4d1114f4.PNG'>

>- 장바구니에 처음 담긴 제품의 재구매율 상위 5개를 확인한 결과, 전체 주문에서 극 소량만 팔린 제품이 상위권을 차지하고 있었다.
>- 따라서 제품의 판매된 수량이 일정한양을 가지고 있는것에 대해서 재구매율을 확인 해야 한다.

### 몇개가 팔려야 될까?
<img src = 'https://user-images.githubusercontent.com/60168331/79730726-36d5c380-832c-11ea-9bc0-db784d2a4585.PNG'>
<img src = 'https://user-images.githubusercontent.com/60168331/79730729-376e5a00-832c-11ea-8814-194098b70c6e.PNG'>
<img src = 'https://user-images.githubusercontent.com/60168331/79730730-376e5a00-832c-11ea-917a-40d285e1f214.PNG'>

>- 장바구니에 첫번째로 넣어진 제품중 1000개이상 팔린 제품이 전체의 약 48%를 대표하는 것으로 확인되었다.

### 장바구니에 첫번째로 들어가고 1000개 이상 팔린 제품의 재구매 갯수와 재구매비율의 관계는?
<img src = 'https://user-images.githubusercontent.com/60168331/79730732-3806f080-832c-11ea-93b4-71aec74e0905.png'>

>- 사이즈가 크게 차이나는 관계로 재구매갯수와 재구매 비율은 log 스케일링 해준 상태로 그래프를 그렸다
>- 그래프를 보면 재구매 횟수가 많은 제품이 재구매 비율도 높은것으로 확인되는것을 알수 있다.

### ※가설 3결론
>- 장바구니에 첫번째로 들어간 제품들은 처음 구매하는 제품인것보다 재구매하는 제품일 확률이 높을것으로 확인되었다.

## 08. 결론
### 가설 1 결론
>- 요일, 시간별 구매횟수와 재구매 횟수는 비슷한 분포를 보인다.

### 가설 2 결론
>- 양의 상관계를 보이는것으로 보아, 주문량이 증가하면 재구매율도 같이 높다. 

### 가설 3 결론
>- 장바구니에 첫번째로 들어간 제품은 재구매 한 제품일 확률이 높다.

### 그래서 이용자들은 어떤 제품을 재구매할 확률이 높을까?
>- 재구매율은 시간 및 요일보다, 제품의 주문량이 높고, 장바구니에 첫번째로 들어간 제품일수록 높아진다.

### So what?
>- 사업이 발전하려면, 재구매율을 높여야 한다.
>- 이를 위해, 주문량이 높고, 장바구니에 처음들어간 제품을 우선으로 프로모션하고,
유사한 제품을 취급하는 채널들과의 연계를 발전시켜야 한다.

### And...
<img src = 'https://user-images.githubusercontent.com/60168331/79731297-0b9fa400-832d-11ea-82fe-bdd785b667b6.png'>

>- 처음 구매한 주문이 약 40%를 차지 한다는건, 취급하는 상점이 증가되거나, 신규유저가 지속적으로 유입되서 새로운 주문이 증가했다고 볼수 있다.
>- 즉 이 데이터셋이 수집된 기간 동안 사업이 지속적으로 성장했다고도 볼수 있다.

## 09. 프로젝트 회고
>-  FastCampas에서 데이터 사이언티스트 스쿨 과정을 들으며 처음 진행했던 프로젝트로, 미흡했던점이 굉장히 많았다.
>- 하지만 처음으로 팀원과 같이 진행하고, 밤을 새가면서 진행했던 프로젝트로 기억에 많이 남는다.
>- 추후에 기회가 된다면 예측모델을 만들어 보면 좋을것 같다.