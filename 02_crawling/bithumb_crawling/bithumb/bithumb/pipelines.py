
# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


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
