# upbit_crwaling
### All coins by hour or day, Crawling designated coins and saving them in MongoDB

#### upbit_day.py : Daily coin price crawl
  - call_code()
    - crwaling all coins market(code), korean_name, english_name
  - upbit_all(count)
    - count : Number of data to crawl
    - input server ip, id, password to be stored in MongoDB
  - upbit_coin(code, coin_englinsh_name, count)
    - code : Code name obtained using call_code
    - coin_english_name : coin english name obtained using call_code
    - count : Number of data to crawl

#### upbit_hour.py : Hourly coin price crawl
  - call_code()
      - crwaling all coins market(code), korean_name, english_name
    - upbit_all(count)
      - count : Number of data to crawl
      - input server ip, id, password to be stored in MongoDB
    - upbit_coin(code, coin_englinsh_name, count)
      - code : Code name obtained using call_code
      - coin_english_name : coin english name obtained using call_code
      - count : Number of data to crawl

#### upbit_day_auto.py : Automatically crawl daily coin prices
  - When executed, today's cryptocurrency price information is fetched and stored in MongoDB.
  
#### upbit_hour_auto.py : Automatically crawl hourly coin prices
  - When executed, the price information of the currency is fetched at the time and stored in MongoDB.
