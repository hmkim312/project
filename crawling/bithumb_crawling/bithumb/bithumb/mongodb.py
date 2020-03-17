import pymongo

client = pymongo.MongoClient('mongodb://{id}:{pw}@{ip}:27017')
# db 생성
db = client.bithumb

# 컬렉션 생성
collection = db.coins
