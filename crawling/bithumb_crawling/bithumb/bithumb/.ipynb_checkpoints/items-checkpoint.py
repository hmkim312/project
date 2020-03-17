import scrapy

class BithumbItem(scrapy.Item):
    date = scrapy.Field()
    coin_names = scrapy.Field()
    coin_codes = scrapy.Field()
    coin_prices = scrapy.Field()
    price_changes = scrapy.Field()
    transaction_volumes = scrapy.Field()
    market_capitalizations = scrapy.Field()
