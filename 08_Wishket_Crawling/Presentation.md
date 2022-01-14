## 위시켓 프로젝트 크롤링 하기
---

### 1) 크롤링 동기
- **위시켓** <https://www.wishket.com/>은 개발, 디자인 등의 프로젝트를 중계해주는 플랫폼이다.
- 클라이언트는 진행할 프로젝트를 등록하고, 파트너는 자신이 할수있는 프로젝트를 확인하여 지원하는 방식이다.
- 파트너의 경우 위시켓에서 매일 오후 5시에 자신이 선택한 프로젝트 분야에 맞춰 이메일을 보내주는 서비스를 진행하고 있으나, 내가 설정한 키워드만 골라서 볼수는 없었다.
- 위와 같은 불편함을 해소하기 위해 위시켓에 등록된 파트너 모집 진행중인 프로젝트 중 내가 설정한 키워드만 볼수 있게 크롤링을 해보기로 하였다.
- 위와 같이 크롤링된 프로젝트 중 내가 지원할수 있는 프로젝트가 있는지 체크를 하려고 한다.

<img src="https://user-images.githubusercontent.com/60168331/149546813-90057b12-e6d7-47be-a99d-e242e6a257f2.png">


### 2) 크롤링 전 **robots.txt** 확인
- 웹페이지를 서비스하는 회사에서는 1)웹서버의 과부하 2)무단 자산(데이터) 취득의 이슈 때문에 자신의 페이지를 무단으로 크롤링해가는것을 좋아하지 않는다.
- 보통 웹페이지에 robots.txt라는 크롤링에 대한 허용사항을 명시해놓고 있으며 웹페에지를 크롤링하기전에 허용하고 있는지 확인 후 진행하는 것이 좋다.
- **위시켓**의 robots.txt를 확인해본 결과 project의 url은 크롤링을 허용하고 있었기 때문에 감사한 마음으로 크롤링 하기로 한다.
<img src="https://user-images.githubusercontent.com/60168331/149541784-05d2d7be-c285-4d23-9e7d-856df36e84f0.png">


### 3) 정적 페이지 or 동적 페이지
- 정적 페이지 : 미리 웹서버에 저장된 내용을 URL에 따라서 보여주는 방식으로, URL이 변경되면 미리 저장된 정보를 보여준다.
- 동적 페이지 : 웹페이지의 정보를 불러와도 URL이 바뀌지 않는 페이지. 쇼핑몰에서 *더보기*를 눌러도 URl은 변경되지 않는 페이지
- 크롤링은 동적 페이지보다는 정적 페이지가 더 쉽다. URL 양식에 맞춰 웹서버에 query를 보내면 내가 원하는 페이지가 return 되기 떄문이다.
- 위시켓의 URL을 확인해보면 처음에는 정적 페이지인것 처럼 보인다. url의 query를 변경하면 페이지의 내용도 바뀌었기 때문이다. 그래서 Beautifulsoup으로 크롤링해보았으나, html이 제대로 파싱되지 않았다. 이는 프로젝트를 보여주는 부분만 동적 페이지인것으로 확인되었기 떄문에 셀레니움을 사용하기로 했다.

### 4) query 확인

```python
url = url = f'https://www.wishket.com/project/?hide_close_project=hide_close_project&order_by=submit&page=1&search_text={keyword}
```
- hide_close_project : 마감된 프로젝트는 보이지 않게 한다.
- order_by : *order_by*를 *sumit*으로 바꾸면 최신 등록된 프로젝트를 볼수있다.
- search_text : 지정한 키워드를 *search_text*의 value로 query를 요청하면 해당 키워드만 가진 프로젝트 정보를 리턴 받는다.
- page : 기본 1페이지부터 시작하며 키워드를 요청한 내용이 많을수록 N개의 페이지로 이루어진다.

### 5) 크롤링 할 정보
- 프로젝트의 모든 정보를 가져와서 일일이 확인하기에는 너무 많은 노력이 들어갈것으로 판단하여 간략한 정보만 크롤링을 하기로 했다.
- 간략하게 크롤링한 정보를 보고 자세한 내용은 상세 프로젝트 URL에 접속하여 확인하면 된다고 생각했다.
- 따라서, 키워드, 등록일시, 프로젝트의 제목, 금액, 프로젝트 상세 URL의 정보만 크롤링 하기로 하였다.


### 6) 크롤링 구조 요약
1) 위시켓의 프로젝트 페이지에서
2) 가장 최신으로 등록된 프로젝트들을
3) 지원할만한 프로젝트가 있는지 확안히기 위해
3) 내가 설정한 키워드만 검색하여
4) 셀레니움을 이용하여 크롤링 후 엑셀로 저장한다.

### 7) Python Code 설명

```python
from selenium import webdriver 
import pandas as pd
```

- 크롤링을 위한 셀레니움과 데이터 프레임을 조작하기 위한 판다스를 import 하다.

```python
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
```

- 리눅스 서버와 같이 GUI가 없는 환경에서 사용할떄는 위의 option을 추가해 준다.

```python
driver = webdriver.Chrome('./chromedriver', chrome_options = chrome_options)
```
- 셀레니움 객체 생성

```python
keywords = ['인공지능', '데이터', 'AI']
```
- 키워드 설정 

``` python
results = {'키워드' : [],
           '등록일시' : [],
           '제목' : [],
           '가격' : [],
           '링크' : []
          }
```

- 크롤링 결과 저장할 dict in list 생성
- 키워드 : 내가 설정한 키워드 저장
- 등록일시 : 프로젝트가 등록된 일시 저장
- 제목 : 프로젝트 제목 저장
- 가격 : 프로젝트 가격 저장
- 링크 : 프로젝트 상세 설명이 있는 URL 저장

``` python
url = f'https://www.wishket.com/project/?hide_close_project=hide_close_project&order_by=submit&page=1&search_text={keyword}'
driver.get(url)

max_page = len(driver.find_elements_by_class_name('page-link'))
pages = range(1, max_page)
```

- 최초 URL을 통해 해당 키워드의 전체 페이지를 획득한다.
- 키워드 별로 프로젝트가 하나도 없으수도 있고 10page가 넘을수도 있기 때문이다.

```python
url = f'https://www.wishket.com/project/?hide_close_project=hide_close_project&order_by=submit&page={page}&search_text={keyword}'
driver.get(url)

titles = driver.find_elements_by_css_selector('.subtitle-2-medium.project-link')
prices = driver.find_elements_by_css_selector('.body-2-medium.estimated.estimated-price')
dates = driver.find_elements_by_class_name('project-recruit-guide.caption-1.text300')
```

- css selector를 사용하여 하나의 프로젝트 리스트페이지에 있는 모든 제목, 가격, 등록일시를 가져 온다.
- 하나의 프로젝트 페이지에는 최대 10개의 프로젝트가 보이며, 제목, 가격, 등록일시는 같은 class name을 가지고 있어서, 매우 쉽게 가져올수 있었다.
- 프로젝트 상세 링크는 제목고 같이 저장되어 있었다.

```python

for title, price, date in zip(titles, prices, dates):
    results['키워드'].append(keyword)
    results['등록일시'].append(date.text[6:-1])
    results['제목'].append(title.text)
    results['가격'].append(price.text[5:])
    results['링크'].append(title.get_attribute('href'))

```
- 크롤링한 정보들을 results 안에 append해준다.
- 위의 방식을 모든 페이지만큼 for문을 이용하여 실행하면 된다.
- 하나의 키워드에 대해 모든 페이지를 크롤링 하였으면, 다음 키워드를 크롤링하면 된다.

- 아래는 python 전체 코드 이다.

```python
for keyword in keywords:

    url = f'https://www.wishket.com/project/?hide_close_project=hide_close_project&order_by=submit&page=1&search_text={keyword}'
    driver.get(url)

    max_page = len(driver.find_elements_by_class_name('page-link'))
    pages = range(1, max_page)

    for page in pages:
        print(keyword, page)
        url = f'https://www.wishket.com/project/?hide_close_project=hide_close_project&order_by=submit&page={page}&search_text={keyword}'
        driver.get(url)

        titles = driver.find_elements_by_css_selector('.subtitle-2-medium.project-link')
        prices = driver.find_elements_by_css_selector('.body-2-medium.estimated.estimated-price')
        dates = driver.find_elements_by_class_name('project-recruit-guide.caption-1.text300')

        for title, price, date in zip(titles, prices, dates):
            results['키워드'].append(keyword)
            results['등록일시'].append(date.text[6:-1])
            results['제목'].append(title.text)
            results['가격'].append(price.text[5:])
            results['링크'].append(title.get_attribute('href'))

            print(date.text[6:-1], title.text, price.text[5:], title.get_attribute('href'))

driver.quit()        
results = pd.DataFrame(results).drop_duplicates('링크')
```




    데이터 1
    2022.01.12 펀드 관리 시스템 기획 고도화, 디자인, 퍼블리싱 18,000,000원 https://www.wishket.com/project/113517/
    2022.01.12 전자정부프레임워크 홈페이지 게시판 커스텀 및 퍼블리싱 적용( 일 6시간 근무 ) 3,000,000원 https://www.wishket.com/project/113515/
    2022.01.12 반려견 헬스케어 웨어러블 디바이스 연동 Android, iOS 앱 개발 20,000,000원 https://www.wishket.com/project/113489/
    2022.01.12 인천국제공항공사 모바일 A-CDM 앱 전환 사업 기획 12,000,000원 https://www.wishket.com/project/113384/
    2022.01.11 AI 및 Block Chain 기반 데이터 SaaS 플랫폼 및 소개 사이트 ... 100,000,000원 https://www.wishket.com/project/113478/
    2022.01.11 SKB 21년 Oasis Cloud 전환 구축 90,000,000원 https://www.wishket.com/project/113473/
    2022.01.11 B2B BI 플랫폼 서비스 웹 화면 디자인 3,000,000원 https://www.wishket.com/project/113468/
    2022.01.11 운영 중인 쇼핑몰 정산 시스템 구축 40,000,000원 https://www.wishket.com/project/113454/
    2022.01.10 Unity3D 기반 딥러닝 데이터 연동 및 리깅 2,000,000원 https://www.wishket.com/project/113440/
    2022.01.10 PLC와 연동하는 알람 및 제어 애플리케이션 구축 20,000,000원 https://www.wishket.com/project/113411/
    데이터 2
    2022.01.10 퍼스트몰 솔루션 커스터마이징 및 이전 작업 14,000,000원 https://www.wishket.com/project/113406/
    2022.01.10 프린터 주문데이터 후킹 개발 3,000,000원 https://www.wishket.com/project/113388/
    2022.01.07 소모임 IOS 앱 개발 4,000,000원 https://www.wishket.com/project/113377/
    2022.01.07 프랜차이즈 ERP 및 매장관리 보고 시스템 구축 250,000,000원 https://www.wishket.com/project/113372/
    2022.01.07 iOS Lidar 센서를 이용한 3차원 데이터 수집 개발 32,000,000원 https://www.wishket.com/project/113367/
    2022.01.07 2차원 이미지의 3차원화 Python/Tensorflow 개발 32,000,000원 https://www.wishket.com/project/113360/
    2022.01.06 고객 관리 커뮤니케이션 프로그램 구축 110,000,000원 https://www.wishket.com/project/113310/
    2022.01.06 기존 홈페이지 이미지/텍스트 수정 작업 150,000원 https://www.wishket.com/project/113343/
    2022.01.06 각 플랫폼 광고API를 활용한 광고 관리 웹 구축 30,000,000원 https://www.wishket.com/project/113322/
    2022.01.06 품질데이터 관리 및 보고서/성적서 발급 시스템 구축 40,000,000원 https://www.wishket.com/project/113325/
    데이터 3
    2022.01.06 클라우드 기반 데이터 관리 솔루션 기획/설계 10,000,000원 https://www.wishket.com/project/113314/
    2022.01.05 한국 교회 통계 모바일 웹 시스템 개발 25,000,000원 https://www.wishket.com/project/113286/
    2022.01.04 공장 설비 환경진단 웹 JAVA 플랫폼 Frontend 개발 32,000,000원 https://www.wishket.com/project/113258/
    2022.01.04 공장설비 빅데이터 수집/가공 서비스 Java/Spring Boot/Kafka 개발 36,000,000원 https://www.wishket.com/project/113254/
    2022.01.04 vba 기반 거래처별 출재고 청구서 구축 10,000,000원 https://www.wishket.com/project/113248/
    2022.01.04 검색 결과 제공 모바일 애플리케이션 개발 5,000,000원 https://www.wishket.com/project/113181/
    2022.01.03 매월 갱신되는 데이터 시각화 모듈 PC 프로그램 개발 10,000,000원 https://www.wishket.com/project/113212/
    2022.01.03 웨어러블 의료 데이터 분석 시스템 iOS 운영 및 유지보수 21,000,000원 https://www.wishket.com/project/113187/
    2022.01.03 웨어러블 의료 데이터 분석 시스템 Android 운영 및 유지보수 21,000,000원 https://www.wishket.com/project/113189/
    2022.01.03 웹서비스 플랫폼 운영 및 보완 개발 39,000,000원 https://www.wishket.com/project/113180/
    데이터 4
    2022.01.03 앱 데이터 이전 및 덮어쓰기 작업 5,000,000원 https://www.wishket.com/project/113173/
    2021.12.31 Python/Tensorflow/Elasticsearch/Gensim 자연 ... 10,000,000원 https://www.wishket.com/project/113126/
    2021.12.31 게임 프로젝트 mysql 성능 튜닝 및 유지보수 40,000,000원 https://www.wishket.com/project/113165/
    2021.12.30 한국 가정의 거주용 전력 사용량 플랫폼 웹/앱 개발 6,000,000원 https://www.wishket.com/project/112995/
    2021.12.30 피부 모발 진단 AI Solution 개발 30,000,000원 https://www.wishket.com/project/113146/
    2021.12.30 QML 기반 어플리케이션 Frontend 개발 5,000,000원 https://www.wishket.com/project/113115/
    2021.12.30 추천 관광 플랫폼 백엔드 개발 20,000,000원 https://www.wishket.com/project/113094/
    2021.12.29 ESS 기반 에너지 블록체인 전력거래 서비스 개발 및 운영(재택근무) 12,500,000원 https://www.wishket.com/project/113067/
    AI 1
    2022.01.12 스터디 플랫폼 Android, iOS 앱 유지보수 5,000,000원 https://www.wishket.com/project/113507/
    2022.01.12 NFT마켓을 위한 Klaytn API Service 및 지갑 연동 7,000,000원 https://www.wishket.com/project/113498/
    2022.01.11 AI 및 Block Chain 기반 데이터 SaaS 플랫폼 및 소개 사이트 ... 100,000,000원 https://www.wishket.com/project/113478/
    2022.01.11 B2B BI 플랫폼 서비스 웹 화면 디자인 3,000,000원 https://www.wishket.com/project/113468/
    2022.01.11 솔루션 기반 금융권 EAI,FEP 개발 운영관리 240,000,000원 https://www.wishket.com/project/113456/
    2022.01.10 컨테이너(AWS EKS) 및 서비스 매쉬(Istio)의 설계 및 구성 90,000,000원 https://www.wishket.com/project/113439/
    2022.01.10 서비스 매쉬 기반 쿠버네티스 프로젝트 SWA 78,000,000원 https://www.wishket.com/project/113389/
    2022.01.10 인증 Android, iOS 앱 및 웹 페이지 리뉴얼 15,000,000원 https://www.wishket.com/project/113419/
    2022.01.10 APP Install 모션 그래픽 영상 제작 5,000,000원 https://www.wishket.com/project/113316/
    2022.01.10 고도몰 기반 쇼핑몰 웹을 하이브리드 앱으로 패키징 7,000,000원 https://www.wishket.com/project/113378/
    AI 2
    2022.01.07 고도몰 기반 브랜드 웹 쇼핑몰 퍼블리싱/개발 3,000,000원 https://www.wishket.com/project/113327/
    2022.01.07 급식관리시스템 고도화 개발 32,500,000원 https://www.wishket.com/project/113362/
    2022.01.07 2차원 이미지의 3차원화 Python/Tensorflow 개발 32,000,000원 https://www.wishket.com/project/113360/
    2022.01.06 중고차 거래 모바일 플랫폼 Backend 개발 60,000,000원 https://www.wishket.com/project/113313/
    2022.01.06 오픈씨에서 클레이튼 기반 대량 민팅시스템 구축 9,000,000원 https://www.wishket.com/project/113299/
    2022.01.05 건축사와 고객간 매칭 웹 플랫폼 구축 80,000,000원 https://www.wishket.com/project/113243/
    2022.01.03 제조기업 매칭 플랫폼 유지보수 및 개선 45,000,000원 https://www.wishket.com/project/112922/
    2022.01.03 솔루션 기반 외식 기업 반응형 홈페이지 구축 18,000,000원 https://www.wishket.com/project/113188/
    2021.12.31 Python/Tensorflow/Elasticsearch/Gensim 자연 ... 10,000,000원 https://www.wishket.com/project/113126/
    2021.12.30 한국 가정의 거주용 전력 사용량 플랫폼 웹/앱 개발 6,000,000원 https://www.wishket.com/project/112995/
    AI 3
    2021.12.30 피부 모발 진단 AI Solution 개발 30,000,000원 https://www.wishket.com/project/113146/



```python
results
# results.to_excel('wishket.xlsx')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>키워드</th>
      <th>등록일시</th>
      <th>제목</th>
      <th>가격</th>
      <th>링크</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>데이터</td>
      <td>2022.01.12</td>
      <td>펀드 관리 시스템 기획 고도화, 디자인, 퍼블리싱</td>
      <td>18,000,000원</td>
      <td>https://www.wishket.com/project/113517/</td>
    </tr>
    <tr>
      <th>1</th>
      <td>데이터</td>
      <td>2022.01.12</td>
      <td>전자정부프레임워크 홈페이지 게시판 커스텀 및 퍼블리싱 적용( 일 6시간 근무 )</td>
      <td>3,000,000원</td>
      <td>https://www.wishket.com/project/113515/</td>
    </tr>
    <tr>
      <th>2</th>
      <td>데이터</td>
      <td>2022.01.12</td>
      <td>반려견 헬스케어 웨어러블 디바이스 연동 Android, iOS 앱 개발</td>
      <td>20,000,000원</td>
      <td>https://www.wishket.com/project/113489/</td>
    </tr>
    <tr>
      <th>3</th>
      <td>데이터</td>
      <td>2022.01.12</td>
      <td>인천국제공항공사 모바일 A-CDM 앱 전환 사업 기획</td>
      <td>12,000,000원</td>
      <td>https://www.wishket.com/project/113384/</td>
    </tr>
    <tr>
      <th>4</th>
      <td>데이터</td>
      <td>2022.01.11</td>
      <td>AI 및 Block Chain 기반 데이터 SaaS 플랫폼 및 소개 사이트 ...</td>
      <td>100,000,000원</td>
      <td>https://www.wishket.com/project/113478/</td>
    </tr>
    <tr>
      <th>5</th>
      <td>데이터</td>
      <td>2022.01.11</td>
      <td>SKB 21년 Oasis Cloud 전환 구축</td>
      <td>90,000,000원</td>
      <td>https://www.wishket.com/project/113473/</td>
    </tr>
    <tr>
      <th>6</th>
      <td>데이터</td>
      <td>2022.01.11</td>
      <td>B2B BI 플랫폼 서비스 웹 화면 디자인</td>
      <td>3,000,000원</td>
      <td>https://www.wishket.com/project/113468/</td>
    </tr>
    <tr>
      <th>7</th>
      <td>데이터</td>
      <td>2022.01.11</td>
      <td>운영 중인 쇼핑몰 정산 시스템 구축</td>
      <td>40,000,000원</td>
      <td>https://www.wishket.com/project/113454/</td>
    </tr>
    <tr>
      <th>8</th>
      <td>데이터</td>
      <td>2022.01.10</td>
      <td>Unity3D 기반 딥러닝 데이터 연동 및 리깅</td>
      <td>2,000,000원</td>
      <td>https://www.wishket.com/project/113440/</td>
    </tr>
    <tr>
      <th>9</th>
      <td>데이터</td>
      <td>2022.01.10</td>
      <td>PLC와 연동하는 알람 및 제어 애플리케이션 구축</td>
      <td>20,000,000원</td>
      <td>https://www.wishket.com/project/113411/</td>
    </tr>
    <tr>
      <th>10</th>
      <td>데이터</td>
      <td>2022.01.10</td>
      <td>퍼스트몰 솔루션 커스터마이징 및 이전 작업</td>
      <td>14,000,000원</td>
      <td>https://www.wishket.com/project/113406/</td>
    </tr>
    <tr>
      <th>11</th>
      <td>데이터</td>
      <td>2022.01.10</td>
      <td>프린터 주문데이터 후킹 개발</td>
      <td>3,000,000원</td>
      <td>https://www.wishket.com/project/113388/</td>
    </tr>
    <tr>
      <th>12</th>
      <td>데이터</td>
      <td>2022.01.07</td>
      <td>소모임 IOS 앱 개발</td>
      <td>4,000,000원</td>
      <td>https://www.wishket.com/project/113377/</td>
    </tr>
    <tr>
      <th>13</th>
      <td>데이터</td>
      <td>2022.01.07</td>
      <td>프랜차이즈 ERP 및 매장관리 보고 시스템 구축</td>
      <td>250,000,000원</td>
      <td>https://www.wishket.com/project/113372/</td>
    </tr>
    <tr>
      <th>14</th>
      <td>데이터</td>
      <td>2022.01.07</td>
      <td>iOS Lidar 센서를 이용한 3차원 데이터 수집 개발</td>
      <td>32,000,000원</td>
      <td>https://www.wishket.com/project/113367/</td>
    </tr>
    <tr>
      <th>15</th>
      <td>데이터</td>
      <td>2022.01.07</td>
      <td>2차원 이미지의 3차원화 Python/Tensorflow 개발</td>
      <td>32,000,000원</td>
      <td>https://www.wishket.com/project/113360/</td>
    </tr>
    <tr>
      <th>16</th>
      <td>데이터</td>
      <td>2022.01.06</td>
      <td>고객 관리 커뮤니케이션 프로그램 구축</td>
      <td>110,000,000원</td>
      <td>https://www.wishket.com/project/113310/</td>
    </tr>
    <tr>
      <th>17</th>
      <td>데이터</td>
      <td>2022.01.06</td>
      <td>기존 홈페이지 이미지/텍스트 수정 작업</td>
      <td>150,000원</td>
      <td>https://www.wishket.com/project/113343/</td>
    </tr>
    <tr>
      <th>18</th>
      <td>데이터</td>
      <td>2022.01.06</td>
      <td>각 플랫폼 광고API를 활용한 광고 관리 웹 구축</td>
      <td>30,000,000원</td>
      <td>https://www.wishket.com/project/113322/</td>
    </tr>
    <tr>
      <th>19</th>
      <td>데이터</td>
      <td>2022.01.06</td>
      <td>품질데이터 관리 및 보고서/성적서 발급 시스템 구축</td>
      <td>40,000,000원</td>
      <td>https://www.wishket.com/project/113325/</td>
    </tr>
    <tr>
      <th>20</th>
      <td>데이터</td>
      <td>2022.01.06</td>
      <td>클라우드 기반 데이터 관리 솔루션 기획/설계</td>
      <td>10,000,000원</td>
      <td>https://www.wishket.com/project/113314/</td>
    </tr>
    <tr>
      <th>21</th>
      <td>데이터</td>
      <td>2022.01.05</td>
      <td>한국 교회 통계 모바일 웹 시스템 개발</td>
      <td>25,000,000원</td>
      <td>https://www.wishket.com/project/113286/</td>
    </tr>
    <tr>
      <th>22</th>
      <td>데이터</td>
      <td>2022.01.04</td>
      <td>공장 설비 환경진단 웹 JAVA 플랫폼 Frontend 개발</td>
      <td>32,000,000원</td>
      <td>https://www.wishket.com/project/113258/</td>
    </tr>
    <tr>
      <th>23</th>
      <td>데이터</td>
      <td>2022.01.04</td>
      <td>공장설비 빅데이터 수집/가공 서비스 Java/Spring Boot/Kafka 개발</td>
      <td>36,000,000원</td>
      <td>https://www.wishket.com/project/113254/</td>
    </tr>
    <tr>
      <th>24</th>
      <td>데이터</td>
      <td>2022.01.04</td>
      <td>vba 기반 거래처별 출재고 청구서 구축</td>
      <td>10,000,000원</td>
      <td>https://www.wishket.com/project/113248/</td>
    </tr>
    <tr>
      <th>25</th>
      <td>데이터</td>
      <td>2022.01.04</td>
      <td>검색 결과 제공 모바일 애플리케이션 개발</td>
      <td>5,000,000원</td>
      <td>https://www.wishket.com/project/113181/</td>
    </tr>
    <tr>
      <th>26</th>
      <td>데이터</td>
      <td>2022.01.03</td>
      <td>매월 갱신되는 데이터 시각화 모듈 PC 프로그램 개발</td>
      <td>10,000,000원</td>
      <td>https://www.wishket.com/project/113212/</td>
    </tr>
    <tr>
      <th>27</th>
      <td>데이터</td>
      <td>2022.01.03</td>
      <td>웨어러블 의료 데이터 분석 시스템 iOS 운영 및 유지보수</td>
      <td>21,000,000원</td>
      <td>https://www.wishket.com/project/113187/</td>
    </tr>
    <tr>
      <th>28</th>
      <td>데이터</td>
      <td>2022.01.03</td>
      <td>웨어러블 의료 데이터 분석 시스템 Android 운영 및 유지보수</td>
      <td>21,000,000원</td>
      <td>https://www.wishket.com/project/113189/</td>
    </tr>
    <tr>
      <th>29</th>
      <td>데이터</td>
      <td>2022.01.03</td>
      <td>웹서비스 플랫폼 운영 및 보완 개발</td>
      <td>39,000,000원</td>
      <td>https://www.wishket.com/project/113180/</td>
    </tr>
    <tr>
      <th>30</th>
      <td>데이터</td>
      <td>2022.01.03</td>
      <td>앱 데이터 이전 및 덮어쓰기 작업</td>
      <td>5,000,000원</td>
      <td>https://www.wishket.com/project/113173/</td>
    </tr>
    <tr>
      <th>31</th>
      <td>데이터</td>
      <td>2021.12.31</td>
      <td>Python/Tensorflow/Elasticsearch/Gensim 자연 ...</td>
      <td>10,000,000원</td>
      <td>https://www.wishket.com/project/113126/</td>
    </tr>
    <tr>
      <th>32</th>
      <td>데이터</td>
      <td>2021.12.31</td>
      <td>게임 프로젝트 mysql 성능 튜닝 및 유지보수</td>
      <td>40,000,000원</td>
      <td>https://www.wishket.com/project/113165/</td>
    </tr>
    <tr>
      <th>33</th>
      <td>데이터</td>
      <td>2021.12.30</td>
      <td>한국 가정의 거주용 전력 사용량 플랫폼 웹/앱 개발</td>
      <td>6,000,000원</td>
      <td>https://www.wishket.com/project/112995/</td>
    </tr>
    <tr>
      <th>34</th>
      <td>데이터</td>
      <td>2021.12.30</td>
      <td>피부 모발 진단 AI Solution 개발</td>
      <td>30,000,000원</td>
      <td>https://www.wishket.com/project/113146/</td>
    </tr>
    <tr>
      <th>35</th>
      <td>데이터</td>
      <td>2021.12.30</td>
      <td>QML 기반 어플리케이션 Frontend 개발</td>
      <td>5,000,000원</td>
      <td>https://www.wishket.com/project/113115/</td>
    </tr>
    <tr>
      <th>36</th>
      <td>데이터</td>
      <td>2021.12.30</td>
      <td>추천 관광 플랫폼 백엔드 개발</td>
      <td>20,000,000원</td>
      <td>https://www.wishket.com/project/113094/</td>
    </tr>
    <tr>
      <th>37</th>
      <td>데이터</td>
      <td>2021.12.29</td>
      <td>ESS 기반 에너지 블록체인 전력거래 서비스 개발 및 운영(재택근무)</td>
      <td>12,500,000원</td>
      <td>https://www.wishket.com/project/113067/</td>
    </tr>
    <tr>
      <th>38</th>
      <td>AI</td>
      <td>2022.01.12</td>
      <td>스터디 플랫폼 Android, iOS 앱 유지보수</td>
      <td>5,000,000원</td>
      <td>https://www.wishket.com/project/113507/</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AI</td>
      <td>2022.01.12</td>
      <td>NFT마켓을 위한 Klaytn API Service 및 지갑 연동</td>
      <td>7,000,000원</td>
      <td>https://www.wishket.com/project/113498/</td>
    </tr>
    <tr>
      <th>42</th>
      <td>AI</td>
      <td>2022.01.11</td>
      <td>솔루션 기반 금융권 EAI,FEP 개발 운영관리</td>
      <td>240,000,000원</td>
      <td>https://www.wishket.com/project/113456/</td>
    </tr>
    <tr>
      <th>43</th>
      <td>AI</td>
      <td>2022.01.10</td>
      <td>컨테이너(AWS EKS) 및 서비스 매쉬(Istio)의 설계 및 구성</td>
      <td>90,000,000원</td>
      <td>https://www.wishket.com/project/113439/</td>
    </tr>
    <tr>
      <th>44</th>
      <td>AI</td>
      <td>2022.01.10</td>
      <td>서비스 매쉬 기반 쿠버네티스 프로젝트 SWA</td>
      <td>78,000,000원</td>
      <td>https://www.wishket.com/project/113389/</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AI</td>
      <td>2022.01.10</td>
      <td>인증 Android, iOS 앱 및 웹 페이지 리뉴얼</td>
      <td>15,000,000원</td>
      <td>https://www.wishket.com/project/113419/</td>
    </tr>
    <tr>
      <th>46</th>
      <td>AI</td>
      <td>2022.01.10</td>
      <td>APP Install 모션 그래픽 영상 제작</td>
      <td>5,000,000원</td>
      <td>https://www.wishket.com/project/113316/</td>
    </tr>
    <tr>
      <th>47</th>
      <td>AI</td>
      <td>2022.01.10</td>
      <td>고도몰 기반 쇼핑몰 웹을 하이브리드 앱으로 패키징</td>
      <td>7,000,000원</td>
      <td>https://www.wishket.com/project/113378/</td>
    </tr>
    <tr>
      <th>48</th>
      <td>AI</td>
      <td>2022.01.07</td>
      <td>고도몰 기반 브랜드 웹 쇼핑몰 퍼블리싱/개발</td>
      <td>3,000,000원</td>
      <td>https://www.wishket.com/project/113327/</td>
    </tr>
    <tr>
      <th>49</th>
      <td>AI</td>
      <td>2022.01.07</td>
      <td>급식관리시스템 고도화 개발</td>
      <td>32,500,000원</td>
      <td>https://www.wishket.com/project/113362/</td>
    </tr>
    <tr>
      <th>51</th>
      <td>AI</td>
      <td>2022.01.06</td>
      <td>중고차 거래 모바일 플랫폼 Backend 개발</td>
      <td>60,000,000원</td>
      <td>https://www.wishket.com/project/113313/</td>
    </tr>
    <tr>
      <th>52</th>
      <td>AI</td>
      <td>2022.01.06</td>
      <td>오픈씨에서 클레이튼 기반 대량 민팅시스템 구축</td>
      <td>9,000,000원</td>
      <td>https://www.wishket.com/project/113299/</td>
    </tr>
    <tr>
      <th>53</th>
      <td>AI</td>
      <td>2022.01.05</td>
      <td>건축사와 고객간 매칭 웹 플랫폼 구축</td>
      <td>80,000,000원</td>
      <td>https://www.wishket.com/project/113243/</td>
    </tr>
    <tr>
      <th>54</th>
      <td>AI</td>
      <td>2022.01.03</td>
      <td>제조기업 매칭 플랫폼 유지보수 및 개선</td>
      <td>45,000,000원</td>
      <td>https://www.wishket.com/project/112922/</td>
    </tr>
    <tr>
      <th>55</th>
      <td>AI</td>
      <td>2022.01.03</td>
      <td>솔루션 기반 외식 기업 반응형 홈페이지 구축</td>
      <td>18,000,000원</td>
      <td>https://www.wishket.com/project/113188/</td>
    </tr>
  </tbody>
</table>
</div>

### 8) 결과 및 후기
- 위시켓은 크롤링하기 굉장히 쉬운 웹사이트였다.
- 아무래도 위시켓에서 중계하는 프로젝트가 여러방면으로 노출이되어야 더 많은 고객을 유치할수 있어서 인듯 하다.
- 하지만 지금 개발된 크롤링은 내가 매번 수동으로 크롤링을 실행시켜줘야 하는 번거로움과 엑셀로 저장되어있기 떄문에 프로젝트를 확인하기 위해 엑셀 파일읗 한번더 실행해야한다는것 번거로움 및 개선사항으로 남아있다.
- 수동으로 실행시키는 것에 대한 개선 방안으로는 리눅스의 크론탭을 이용하여 매일 정기적으로 크롤링을 진행하면 해결될듯 하다.
- 엑셀로 저장하여 보는 방식은, slack bot을 이용하여 매일 알람을 보내주는 방식을 이용하는게 제일 편할듯 하다.
- 또한, 더 많은 정보를 크롤링하여 위시켓에 등록되는 프로젝트에 관해 EDA 맟 예측 모델을 만들어 볼수도 있을것 같다.
