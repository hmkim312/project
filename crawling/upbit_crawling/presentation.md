# Crawling project
## 가상 화폐 가격 Crawling
 <img src="https://user-images.githubusercontent.com/60168331/76542237-0cd8e800-64c8-11ea-9ee7-ad256274bafe.PNG">

  - 데이터 수집의 개요
	- 2017년 11월쯤 부터 가상화폐의 투자가 인기몰이를 하여, 많은 사람들이 투자를 하였고 2017년 12월에는 자고 일어나면 모든 가상화폐들이 약 2배 이상 올랐던 적이 있었다.
    - 그 당시엔 가상화폐에 투자하지 않으면 바보라는 소리가 나올정도로 엄청난 투자효과를 가져왔었고, 결국엔 1 비트코인은 중형차 한대 가격까지 오르게 된다.
    - 하지만 2018년에 가격은 폭락하기 시작, 많은 사람들이 엄청난 손해를 보게 되었으며, 그때부터 소위 '존버'라는 단어가 유행되기 시작했다.
    - 그 당시 많은 손해를 본 사람중 한명으로써 비트코인의 가격을 조금이라도 예측할 수 있었다면 (물론 힘들겠지만) 최소한 엄청난 손해는 보지 않았을 것으로 생각하였다.
    - 일단 데이터부터 있어야, 무언가를 예측해볼수 있을것으로 판단되어 가상화폐를 크롤링해보도록 하였다.
   

## 데이터 수집의 계획 및 주기 작성
<img src='https://user-images.githubusercontent.com/60168331/76542699-b9b36500-64c8-11ea-9078-d019ce657fc8.PNG'>

- 업비트
  - 업비트라는 한국 거래소에서 제공하는 데이터로 크롤링을 하였다.
  - 시간 단위와 일 단위로 크롤링 하는 코드를 작성하였다.
  - 전체 코인과 지정코인, 그리고 코인의 리스트를 볼수 있게 코드를 작성하였다.

- 크롤링 하는 방법
  - 일단위 (upbit_day.py)
    - ```python
        import pandas as pd  # version - 0.25.1
        import numpy as np  # version - 1.16.5
        import requests  # version - 2.22.0
        import pymongo  # version - 2.8.1
        import getpass


        def call_code():
            # coin 종류 가져오기
            url = 'https://api.upbit.com/v1/market/all'
            response = requests.get(url)
            datas = response.json()
            # 데이터 프레임으로 변경
            df = pd.DataFrame(datas)
            # market 기준 한화로 변경
            coins_krw = df[df['market'].str.startswith('KRW')].reset_index(drop=True)
            return coins_krw


        def upbit_all(count):
            # 서버 접속 정보 확인
            ip = input('서버ip를 입력하세요:')
            id_s = input('서버접속 id를 입력하세요:')
            pw = getpass.getpass('서버접속 pw를 입력하세요:')

            # server 연결
            server = pymongo.MongoClient(f'mongodb://{id_s}:{pw}@{ip}:27017/')
            db = server.upbit_day

            # 마켓 코드 가져오기.
            url = 'https://api.upbit.com/v1/market/all'
            response = requests.get(url)
            datas = response.json()

            # 데이터 프레임으로 변경
            df = pd.DataFrame(datas)

            # market 기준 한화로 변경
            coins_krw = df[df['market'].str.startswith(
                'KRW')].reset_index(drop=True)

            # 데이터프레임을 코드와 네임의 딕셔너리로 변경
            a = coins_krw['market'].to_dict().values()
            b = coins_krw['english_name'].to_dict().values()
            coin_names = dict(zip(a, b))

            # database에 저장(mongodb)
            for code, name in coin_names.items():
                collection = db[name]
                response = requests.get(
                    f'https://crix-api-cdn.upbit.com/v1/crix/candles/days?\
                    code=CRIX.UPBIT.{code}&count={count}&ciqrandom=1582871221736')
                datas = response.json()
                idx = collection.insert(datas)
                print(code, len(idx), end=" ")

                
        # 지정한 한개의 코인을 가져오는 함수

        def upbit_coin(code, coin_englingsh_name, count):

            # 서버 접속 정보 확인
            ip = input('서버ip를 입력하세요:')
            id_s = input('서버접속 id를 입력하세요:')
            pw = getpass.getpass('서버접속 pw를 입력하세요:')

            # server 연결
            server = pymongo.MongoClient(f'mongodb://{id_s}:{pw}@{ip}:27017/')
            db = server.upbit_day

            # 지정된 코인 정보 가져오기
            response = requests.get(
                f'https://crix-api-cdn.upbit.com/v1/crix/candles/days?\
                code=CRIX.UPBIT.{code}&count={count}&ciqrandom=1582871221736')
            datas = response.json()

            # database에 저장(mongodb)
            collection = db[coin_englingsh_name]
            idx = collection.insert(datas)
            print(code, len(idx), end=" ")
        ```
    
    - 시간 단위(upbit_hour.py)
    - ```python
        
        import pandas as pd  # version - 0.25.1
        import numpy as np  # version - 1.18.1
        import requests  # version - 2.22.0
        import pymongo  # version - 2.8.1
        import getpass
        import time
        from datetime import datetime
        import getpass


        def call_code():
            # coin 종류 가져오기
            url = 'https://api.upbit.com/v1/market/all'
            response = requests.get(url)
            datas = response.json()
            # 데이터 프레임으로 변경
            df = pd.DataFrame(datas)
            # market 기준 한화로 변경
            coins_krw = df[df['market'].str.startswith('KRW')].reset_index(drop=True)
            return coins_krw


        def upbit_all(count):
            # 서버 접속 정보 확인
            ip = input('서버ip를 입력하세요:')
            id_s = input('서버접속 id를 입력하세요:')
            pw = getpass.getpass('서버접속 pw를 입력하세요:')

            # server 연결
            server = pymongo.MongoClient(f'mongodb://{id_s}:{pw}@{ip}:27017/')
            db = server.upbit_hour

            # 마켓 코드 가져오기.
            url = 'https://api.upbit.com/v1/market/all'
            response = requests.get(url)
            datas = response.json()

            # 데이터 프레임으로 변경
            df = pd.DataFrame(datas)

            # market 기준 한화로 변경
            coins_krw = df[df['market'].str.startswith(
                'KRW')].reset_index(drop=True)

            # 데이터프레임을 코드와 네임의 딕셔너리로 변경
            a = coins_krw['market'].to_dict().values()
            b = coins_krw['english_name'].to_dict().values()
            coin_names = dict(zip(a, b))

            # database에 저장(mongodb)
            day = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
            for code, name in coin_names.items():
                collection = db[name]
                response = requests.get(
                    f'https://crix-api-cdn.upbit.com/v1/crix/candles/minutes/60?\
                    code=CRIX.UPBIT.{code}&count={count}&to={day}T23:59:59Z&')
                datas = response.json()
                idx = collection.insert(datas)
                print(code, len(idx), end=" ")

        # 지정한 한개의 코인을 가져오는 함수

        def upbit_coin(code, coin_englingsh_name, count):

            # 서버 접속 정보 확인
            ip = input('서버ip를 입력하세요:')
            id_s = input('서버접속 id를 입력하세요:')
            pw = getpass.getpass('서버접속 pw를 입력하세요:')

            # server 연결
            server = pymongo.MongoClient(f'mongodb://{id_s}:{pw}@{ip}:27017/')
            db = server.upbit_hour

            # 지정된 코인 정보 가져오기
            day = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
            response = requests.get(
                f'https://crix-api-cdn.upbit.com/v1/crix/candles/minutes/60?\
                code=CRIX.UPBIT.{code}&count={count}&to={day}T23:59:59Z&')
            datas = response.json()

            # database에 저장(mongodb)
            collection = db[coin_englingsh_name]
            idx = collection.insert(datas)
            print(code, len(idx), end=" ")

        # 지정한 한개의 코인을 가져오는 함수

        def upbit_coin(code, coin_englingsh_name, count):

            # 서버 접속 정보 확인
            ip = input('서버ip를 입력하세요:')
            id_s = input('서버접속 id를 입력하세요:')
            pw = getpass.getpass('서버접속 pw를 입력하세요:')

            # server 연결
            server = pymongo.MongoClient(f'mongodb://{id_s}:{pw}@{ip}:27017/')
            db = server.upbit_day

            # 지정된 코인 정보 가져오기
            response = requests.get(
                f'https://crix-api-cdn.upbit.com/v1/crix/candles/days?\
                code=CRIX.UPBIT.{code}&count={count}&ciqrandom=1582871221736')
            datas = response.json()

            # database에 저장(mongodb)
            collection = db[coin_englingsh_name]
            idx = collection.insert(datas)
            print(code, len(idx), end=" ")
        ```

    - 코드 설명
        
      - call_code() : 현재 거래중인 코인의 코드를 가져 옵니다.
        - <img src ='https://user-images.githubusercontent.com/60168331/76542275-1b270400-64c8-11ea-9a73-aa273cb9985a.PNG'>
      - upbit_all(count) : 현재 거래중인 모든 코인의 가격(한화)정보를 가져옵니다.
        - <img src = 'https://user-images.githubusercontent.com/60168331/76542702-ba4bfb80-64c8-11ea-8432-a0c144889fee.PNG'>
        - count : 입력값으로 가져올 데이터의 갯수를 입력합니다.
        - ip = input('서버ip를 입력하세요:') : 몽고디비에 연결할 서버ip를 입력합니다.
        - id_s = input('서버접속 id를 입력하세요:') : 서버접속의 id를 입력합니다. 
        - pw = getpass.getpass('서버접속 pw를 입력하세요:') : 서버접속의 password를 입력합니다.
      
      - upbit_coin(code, coin_englingsh_name, count) : 지정한 한개의 코인의 가격을 가져옵니다.
        - <img src = 'https://user-images.githubusercontent.com/60168331/76542703-bae49200-64c8-11ea-8544-515c11371013.PNG'>
        - code : call_code()의 market(code_name)을 지정하여 입력합니다.
        - coin_englingsh_name : call_code()의 englingsh_name을 지정하여 입력합니다.
        - count : 입력값으로 가져올 데이터의 갯수를 입력합니다.
        - ip = input('서버ip를 입력하세요:') : 몽고디비에 연결할 서버ip를 입력합니다.
        - id_s = input('서버접속 id를 입력하세요:') : 서버접속의 id를 입력합니다. 
        - pw = getpass.getpass('서버접속 pw를 입력하세요:') : 서버접속의 password를 입력합니다.
  
	- 서버에서 자동으로 활동하는 파일
    	- 시간단위 (upbit_hour_auto.py)
    	- ```python
                import pandas as pd # version - 0.25.1
                import numpy as np # version - 1.16.5
                import requests # version - 2.22.0
                import pymongo # version - 2.8.1
                from bs4 import BeautifulSoup # version - 4.8.0
                from datetime import datetime
                import getpass
                import time


                def upbit_all(count):

                    # server 연결
                    server = pymongo.MongoClient('mongodb://dss:dss@13.209.146.29:27017/')
                    db = server.upbit_hour_auto

                    # 마켓 코드 가져오기.
                    url = 'https://api.upbit.com/v1/market/all'
                    response = requests.get(url)
                    datas = response.json()

                    # 데이터 프레임으로 변경
                    df = pd.DataFrame(datas)

                    # market 기준 한화로 변경
                    coins_krw = df[df['market'].str.startswith(
                        'KRW')].reset_index(drop=True)

                    # 데이터프레임을 코드와 네임의 딕셔너리로 변경
                    a = coins_krw['market'].to_dict().values()
                    b = coins_krw['english_name'].to_dict().values()
                    coin_names = dict(zip(a, b))

                    # database에 저장(mongodb)
                    day = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
                    for code, name in coin_names.items():
                        collection = db[name]
                        response = requests.get(
                            f'https://crix-api-cdn.upbit.com/v1/crix/candles/minutes/60?\
                            code=CRIX.UPBIT.{code}&count={count}&to={day}T23:59:59Z&')
                        datas = response.json()
                        idx = collection.insert(datas)
                        # print(code, len(idx), end=" ")
                    
            upbit_all(1)
            ```
        - 일 단위(upbit_day_auto.py)
        - ```python
                import pandas as pd # version - 0.25.1
                import numpy as np # version - 1.16.5
                import requests # version - 2.22.0
                import pymongo # version - 2.8.1
                import getpass


                def upbit_all(count):
                    
                    # server 연결
                    server = pymongo.MongoClient('mongodb://dss:dss@13.209.146.29:27017/')
                    db = server.upbit_day_auto

                    # 마켓 코드 가져오기.
                    url = 'https://api.upbit.com/v1/market/all'
                    response = requests.get(url)
                    datas = response.json()

                    # 데이터 프레임으로 변경
                    df = pd.DataFrame(datas)

                    # market 기준 한화로 변경
                    coins_krw = df[df['market'].str.startswith(
                        'KRW')].reset_index(drop=True)

                    # 데이터프레임을 코드와 네임의 딕셔너리로 변경
                    a = coins_krw['market'].to_dict().values()
                    b = coins_krw['english_name'].to_dict().values()
                    coin_names = dict(zip(a, b))

                    # database에 저장(mongodb)
                    for code, name in coin_names.items():
                        collection = db[name]
                        response = requests.get(
                            f'https://crix-api-cdn.upbit.com/v1/crix/candles/days?\
                            code=CRIX.UPBIT.{code}&count={count}&ciqrandom=1582871221736')
                        datas = response.json()
                        idx = collection.insert(datas)
                        print(code, len(idx), end=" ")
                        
                upbit_all(1)
            ```

    - 코드 설명
      - upbit_all(count) : 함수가 실행되면 업비트의 모든 코인의 가격정보를 가져옴
      - <img src='https://user-images.githubusercontent.com/60168331/76542306-2417d580-64c8-11ea-92ee-d21e9bc55bc9.PNG'>
        - count : 가져올 데이터의 갯수를 입력

    - columns 설명
      - 'code' : 화폐의 코드
      - 'candleDateTime' : 국제 표준시
      - 'candleDateTimeKst' : 한국시
      - 'openingPrice' : 시가
      - 'highPrice' : 고가
      - 'lowPrice' : 저가
      - 'tradePrice' : 현재가격정보
      - 'candleAccTradeVolume' : 누적체결량
      - 'candleAccTradePrice' :  누적체결대금
      - 'timestamp' : Unix 타임스탬프, 1970년 1월1일부터 얼마나 지났는지에 대한것
      - 'prevClosingPrice' : 전일 종가 (UTC 0기준)
      - 'change' : 전일 종가 대비 변화금액의 여부 (RISE 오름, EVEN 변화없음, FALL떨어짐)
      - 'changePrice' : : 전일 종가 대비 변화금액 (절대값)
      - 'signedChangePrice' : 부호가 있는 변화금액
      - 'changeRate' : 전일 종가 대비 변화량 (절대값)
      - 'signedChangeRate' : 부호가 있는 변화량
      - <img src ='https://user-images.githubusercontent.com/60168331/76542304-2417d580-64c8-11ea-87ab-c636ebf9bc87.PNG'>
  
    - 데이터 저장
      - upbit_day.py : mongodb의 upbit_day에 저장됨
      - upbit_day_auto.py : mongodb의 upbit_day_auto에 저장됨
      - upbit_hour.py : mongodb의 upbit_hour에 저장됨
      - upbit_hour_auto.py : mongodb의 upbit_hour_auto에 저장됨
      - <img src='https://user-images.githubusercontent.com/60168331/76542310-24b06c00-64c8-11ea-8024-00ac586d67d8.PNG'>
    
    - EDA
      - 일일 최고점을 보았을때 최근 하향세인것을 확인할수 있었음
      - <img src ='https://user-images.githubusercontent.com/60168331/76542296-22e6a880-64c8-11ea-8059-e5969705e589.PNG'>

- 프로젝트 회고
  - 업비트에서 respone해주는 형식이 json형식이라 어렵지 않게 한것 같다.
  - 추후, 업비트말고 다른 가상화폐 사이트를 scrapy로 크롤링해야겠다. 