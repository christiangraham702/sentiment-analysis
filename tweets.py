import snscrape.modules.twitter as sntwitter
import pandas as pd

query = '(@Wendys) until:2022-11-14 since:2022-01-01'
tweets = []
limit = 100

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
        print('Tweet scraped from: {}'.format(tweet.username))

df = pd.DataFrame(tweets, columns=['Datetime', 'Username', 'Text'])
df.to_json('data/target_tweets_split.json', orient='split')
