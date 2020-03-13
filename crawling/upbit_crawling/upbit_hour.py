
import pandas as pd # version - 0.25.1
import numpy as np # version - 1.16.5
import requests # version - 2.22.0
import pymongo # version - 2.8.1
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
    ip = input('서버ip를 입력하세요:' )
    id_s = input('서버접속 id를 입력하세요:' )
    pw = getpass.getpass('서버접속 pw를 입력하세요:' )

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
              f'https://crix-api-cdn.upbit.com/v1/crix/candles/minutes/60?code=CRIX.UPBIT.{code}&count={count}&to={day}T00:00:52Z&')
        datas = response.json()
        idx = collection.insert(datas)
        print(code, len(idx), end=" ")

# 지정한 한개의 코인을 가져오는 함수
def upbit_coin(code, coin_englingsh_name, count):
    
    # 서버 접속 정보 확인
    ip = input('서버ip를 입력하세요:' )
    id_s = input('서버접속 id를 입력하세요:' )
    pw = getpass.getpass('서버접속 pw를 입력하세요:' )

    # server 연결
    server = pymongo.MongoClient(f'mongodb://{id_s}:{pw}@{ip}:27017/')
    db = server.upbit_hour

    # 지정된 코인 정보 가져오기
    day = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
    response = requests.get(
         f'https://crix-api-cdn.upbit.com/v1/crix/candles/minutes/60?code=CRIX.UPBIT.{code}&count={count}&to={day}T00:00:52Z&')
    datas = response.json()

    # database에 저장(mongodb)
    collection = db[coin_englingsh_name]
    idx = collection.insert(datas)
    print(code, len(idx), end=" ")
