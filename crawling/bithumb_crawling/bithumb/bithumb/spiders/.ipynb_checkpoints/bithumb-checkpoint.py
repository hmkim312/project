
# python version - 3.7.4
# jupyter lab version - 1.1.4
# os - windows 10 home 1909 version
import pandas as pd # version - 0.25.1
import numpy as np # version - 1.16.5
import requests # version - 2.22.0
import pymongo # version - 2.8.1
from bs4 import BeautifulSoup # version - 4.8.0
from datetime import datetime
import getpass
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
    
    def __init__(self, highprice, lowprice, name, **kwargs):
        self.webhook_url = 'https://hooks.slack.com/services/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'
        self.highprice = highprice
        self.lowprice = lowprice
        self.name = name
        super().__init__(**kwargs)
    
    def parse(self,response):
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
        for item in zip(date, coin_names, coin_codes,coin_prices,price_changes,\
                        transaction_volumes,market_capitalizations):
            item = {
                'date' : date,
                'coin_names' : item[1].strip(),
                'coin_codes' : item[2].strip(),
                'coin_prices' : item[3].replace('원','').replace(',','').strip(),
                'price_changes' : item[4].replace('원','').replace(',','').strip(),
                'transaction_volumes' : item[5].replace('원','').replace('≈','').replace(',','').strip()[:-6],
                'market_capitalizations' : item[6].replace('조','').replace('억','').replace(' ',''),
            }
            yield item
