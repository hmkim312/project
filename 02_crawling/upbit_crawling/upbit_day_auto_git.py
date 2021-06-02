
import pandas as pd # version - 0.25.1
import numpy as np # version - 1.16.5
import requests # version - 2.22.0
import pymongo # version - 2.8.1
import getpass


def upbit_all(count):
    
    # server 연결
    # id, pw, ip는 수정하여 사용
    server = pymongo.MongoClient('mongodb://id:pw@ip:27017/')
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
            f'https://crix-api-cdn.upbit.com/v1/crix/candles/days?code=CRIX.UPBIT.{code}&count={count}&ciqrandom=1582871221736')
        datas = response.json()
        idx = collection.insert(datas)
        print(code, len(idx), end=" ")
        
upbit_all(1)
