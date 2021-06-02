# Crawling project
## 가상 화폐 가격 Crawling
<img src = 'https://user-images.githubusercontent.com/60168331/76886645-755b0700-68c4-11ea-8ef3-3e93762f404d.png'>

  - 데이터 수집의 개요
	- 2017년 11월쯤 부터 가상화폐의 투자가 인기몰이를 하여, 많은 사람들이 투자를 하였고 2017년 12월에는 자고 일어나면 모든 가상화폐들이 약 2배 이상 올랐던 적이 있었다.
    - 그 당시엔 가상화폐에 투자하지 않으면 바보라는 소리가 나올정도로 엄청난 투자효과를 가져왔었고, 결국엔 1 비트코인은 중형차 한대 가격까지 오르게 되며,
    다들 잘 알고 있는 '가즈아!'라는 단어가 유행하기 시작하였다.
    - 하지만 2018년 1월쯤부터 가격은 폭락하기 시작, 많은 사람들이 엄청난 손해를 보게 되었으며, 그때부터 소위 '존버'라는 단어가 유행되기 시작했다.
    - 그 당시 많은 손해를 본 사람중 한명으로써 비트코인의 가격을 조금이라도 예측할 수 있었다면 (물론 힘들겠지만) 최소한 엄청난 손해는 보지 않았을 것으로 생각하였다.
    - 일단 데이터부터 있어야, 무언가를 예측해볼수 있을것으로 판단되어 가상화폐를 크롤링해보도록 하였다.
    - <img src = 'https://user-images.githubusercontent.com/60168331/76885224-3f1c8800-68c2-11ea-83b5-a403c20a52a9.png'>
   

## 데이터 수집의 계획 및 주기 작성
<img src='https://user-images.githubusercontent.com/60168331/76882236-b7347f00-68bd-11ea-8c8b-68ee6f1faee6.PNG'>
<img src='https://user-images.githubusercontent.com/60168331/76882383-ea770e00-68bd-11ea-8707-4914693ae52d.PNG'>

- 빗썸(bithumb)
  - 빗썸이라는 가상화폐 거래소 데이터 중 실시간 전체의 코인 가격을 scray로 크롤링을 하였다.
  - 대한민국 1 ~ 2위 하는 가상화폐 거래소
  - 2017년 기준 2651억, 2018년 기준 2561억의 순이익을 올리는 엄청난 회사

- Crawling 구성
  - scrapy
    - project 구성
    - <img src ='https://user-images.githubusercontent.com/60168331/76882755-7426db80-68be-11ea-8cff-988f4e531f51.PNG'>
  
  - items.py
    ```python
    import scrapy
    class BithumbItem(scrapy.Item):
        date = scrapy.Field()
        coin_names = scrapy.Field()
        coin_codes = scrapy.Field()
        coin_prices = scrapy.Field()
        price_changes = scrapy.Field()
        transaction_volumes = scrapy.Field()
        market_capitalizations = scrapy.Field()
      ```
    
  - Spider.py
     ```python
      import pandas as pd  # version - 0.25.1
      import numpy as np  # version - 1.16.5
      import requests  # version - 2.22.0
      import pymongo  # version - 2.8.1
      from bs4 import BeautifulSoup  # version - 4.8.0
      from datetime import datetime
      import scrapy
      import os
      import json
      import scrapy
      import time

      from bithumb.items import BithumbItem

      # 식별자 변수명은 지정된 변수명을 사용해야함
      class Spider(scrapy.Spider):
          # 거래가 실시간으로 바뀌기 때문에 크롤링할때마다 연결해주어야함
          name = 'BithumbSpider'
          # 1. 웹페이지 연결
          allow_domain = ['https://www.bithumb.com/']
          start_urls = ['https://www.bithumb.com/']

          def __init__(self, highprice, lowprice, name, *args, **kwargs):
              self.highprice = highprice
              self.lowprice = lowprice
              self.name = name
              super(Spider, self).__init__(*args, **kwargs)

          def parse(self, response):
              item = BithumbItem()
              # 실행시간 확인
              date = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))

              # 3. 코인 이름 가져오기
              selector = '//*[@id="sise_list"]/tbody/tr/td[1]/p/a/strong/text()'
              coin_names = response.xpath(selector).extract()

              # 4. 코인 코드 가져오기
              selector = '//*[@id="sise_list"]/tbody/tr/td[1]/p/a/span/text()'
              coin_codes = response.xpath(selector).extract()

              # 5. 전체 가격가져오기
              selector = '/html/body/div[2]/section/div[1]/div[4]/table/tbody/tr/td[2]/strong/text()'
              coin_prices = response.xpath(selector).extract()

              # 6. 변동률 가져오기 (전일 대비)
              selector = '/html/body/div[2]/section/div[1]/div[4]/table/tbody/tr/td[3]/div/strong/text()'
              price_changes = response.xpath(selector).extract()

              # 7. 거래량 가져오기(24th, 단위 백만)
              selector = '/html/body/div[2]/section/div[1]/div[4]/table/tbody/tr/td[4]/span/text()'
              transaction_volumes = response.xpath(selector).extract()

              # 8. 시가총액 가져오기(단위 억)
              selector = '/html/body/div[2]/section/div[1]/div[4]/table/tbody/tr/td[5]/strong/text()'
              market_capitalizations = response.xpath(selector).extract()

              # 9. 리스트 데이터 정리
              for item in zip(date, coin_names, coin_codes, coin_prices, price_changes,
                              transaction_volumes, market_capitalizations):
                  item = {
                      'date': date,
                      'coin_names': item[1].strip(),
                      'coin_codes': item[2].strip(),
                      'coin_prices': item[3].replace('원', '').replace(',', '').strip(),
                      'price_changes': item[4].replace('원', '').replace(',', '').strip(),
                      'transaction_volumes': item[5].replace('원', '').replace('≈', '').replace(',', '').strip()[:-6],
                      'market_capitalizations': item[6].replace('조', '').replace('억', '').replace(' ', ''),
                  }
                  yield item
      ```

  - mongodb.py
    ```python
    import pymongo
    client = pymongo.MongoClient('mongodb://{id}:{pw}@{ip}:27017')

    # db 생성
    db = client.bithumb

    # 컬렉션 생성
    collection = db.coins
    ```
    
  - pipelines.py
    ```python
    import json
    import requests
    from.mongodb import collection

    class BithumbPipeline(object):
        
        # 생성자 함수
        def __init__(self):#,Spider):
            self.webhook_url = 'url'

        # 아규먼트 가져오기
        def open_spider(self, Spider):
            self.highprice = Spider.highprice
            self.lowprice = Spider.lowprice
            self.name = Spider.name
        
        # 슬랙메세지 전송
        def send_msg(self, msg):        
            payload = {
                'channel' : '#coin',
                'username' : 'coin_highrow_bot',
                'icon_emoji' : ':moneybag:',
                'text' : msg,
            }
            # webhook_url로 json.dumps 형태의 페이로드를 전송하는 코드
            requests.post(self.webhook_url, json.dumps(payload))
            time.sleep(1)
            
        # mongodb저장 및 슬랙전송 코드   
        def process_item(self, item, Spider):
            # 몽고디비에 저장하는 코드 추가
            data ={
                'date' : item['date'],
                'coin_names' : item['coin_names'],
                'coin_codes' : item['coin_codes'],
                'coin_prices' : item['coin_prices'],
                'price_changes' : item['price_changes'],
                'transaction_volumes' :item['transaction_volumes'],
                'market_capitalizations' : item['market_capitalizations'],
            }
            
            collection.insert(data)

            if float(self.highprice) < float(item['coin_prices']) and self.name == item['coin_names'] :
                self.send_msg(f"고가 알림 : {item['coin_names']} 가격은 {item['coin_prices']}원 입니다. 알림 설정 금액은 {self.highprice}원 입니다.")
                
            if float(self.lowprice) > float(item['coin_prices']) and self.name == item['coin_names'] :             
                self.send_msg(f"저가 알림 : {item['coin_names']} 가격은 {item['coin_prices']}원 입니다. 알림 설정 금액은 {self.lowprice}원 입니다.")
            
            return item
    ```

    - columns 설명
      - id : mongoDB의 id값(고유값)
      - date : 날짜
      - coin_names : 한글 코인이름	
      - coin_codes : 코인 코드
      - coin_prices : 현재 코인 가격
      - price_changes	: 부호가 있는 변동률
      - transaction_volumes : 거래량(24th, 백만단위)
      - market_capitalizations : 시가총액(억 단위)
      - <img src ='https://user-images.githubusercontent.com/60168331/76883973-480c5a00-68c0-11ea-9fb1-460dd1fd15a9.PNG'>
  
    - 데이터 저장
      - mongodb의 bithumb(DataBase) coins(collection)에 저장됨
      - 현재 30분마다 한번씩 매일 mongoDB에 저장되고 있음.
      - <img src='https://user-images.githubusercontent.com/60168331/76884048-6b370980-68c0-11ea-9a43-f766fca2e49f.PNG'>
    
    - 동작 설명
      - run.sh 파일에 highprice lowprice name을 설정하고 해당 구간을 벗어나면 슬랙으로 알람이 온다.
        - highprice : 알림을 받을 고가 설정(ex 팔고 싶은 가격)
        - lowprice : 알림을 받을 저가 설정(ex 사고 싶은 가격)
        - name : 알림을 받을 코인 설정(ex 비트코인)
        - ``` 
            crapy crawl BithumbSpider -a highprice=7300000 -a lowprice=6000000 -a name='비트코인'```
      - <img src = 'https://user-images.githubusercontent.com/60168331/76884255-c0731b00-68c0-11ea-92fe-b28c715eac01.PNG'>   

    - EDA
      - 매 30분 단위의 3월 14일 ~ 3월 17일까지의 비트코인의 가격을 보았을때 하향세이였다가, 살짝 반등한것을 알수있었음
    - <img src ='https://user-images.githubusercontent.com/60168331/76886859-b4895800-68c4-11ea-9427-b5a467369d74.png'>

- 프로젝트 회고
  - 빗썸에서의 크롤링은 scrapy로 사용해 본것에 만족한다.
  - 다른 주제의 크롤링을 더 연습하여 다른 데이터를 쌓아 분석해보는것을 시도해 보겠다.
  - slack bot을 이용하여, 대답하는 chatbot을 만들어보록 해야겠다.