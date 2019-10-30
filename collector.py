# - coding: utf-8 -
"""
Premium APIを使ったツイート収集
tweepyはPremium APIに対応していないため、代替としてTwitterAPIを使っている

"""

import time
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterError
import csv
from pymongo import MongoClient

"""
MongoDB呼び出し
"""
client = MongoClient('localhost', 27017)
# DB呼び出し
db = client.tweet_db
# コレクション呼び出し
collection = db.collection3


"""
Twitter API
"""
SEARCH_TERM = 'iPhone11 lang:ja'
PRODUCT = '30day'
LABEL = 'dev'

api = TwitterAPI("CdJ5dMMwhxqxcSp9hAKSBmYBK",
             "7sy47R4K3yXyQtjHhHWbA0S3YHBV1yf5Qf2JXll5GqPIWqX0ZQ",
             "1179291145458704384-BJxih9B41KvDZf2lUOFKsRiO7WPilW",
             "z4mBuRCxEOrbNp1Z8bH3iGXoweQ4wrmiaHNJviplwsBJ5")

# csvFile = open('Data/Twitter/10008.csv', 'w', encoding="utf-8")
# csvWriter = csv.writer(csvFile)

next = ''
results = []
while True:
    r = api.request('tweets/search/%s/:%s' % (PRODUCT, LABEL),
                {'query':SEARCH_TERM,
                'fromDate':'201909300000',
                'toDate':'201910020000'
                })
    if r.status_code != 200:    # レスポンスが正常でないとき
        time.sleep(60*15)
        r = api.request('tweets/search/%s/:%s' % (PRODUCT, LABEL),
                        {'query': SEARCH_TERM,
                         'fromDate': '201909190000',
                         'toDate': '201910020000'
                         })
        if r.status_code != 200: break

    for item in r:
        print(item['text'])
        # csvWriter.writerow([item['text'] if 'text' in item else item, item['id_str'], item['created_at']])
        # post = {"text": item['text']  if 'text' in item else item,
        #         "id": item['id_str'],
        #         "created_at": item['created_at']
        # }
        # column = collection.insert_one(post)

    json = r.json()
    if 'next' not in json:      # nextidがないとき
        break
    next = json['next']
