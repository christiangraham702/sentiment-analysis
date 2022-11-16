from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
import json


tweeter_test = "@Target \nI  spoke to a https:\\/\\/t.co\\/vkCuMPhxgx Customer Service in Philippines. They're not allowed to give their 1st name on the phone. But they're assigned a number to identify them like a prisoner. Very degrading. This's a BAD policy. I don't see the logic. Help me understand."

# pre process tweets


def preprocess(tweet):
    tweet = str(tweet).replace('\\/', '/')
    tweet = str(tweet).replace('\n', ' ')
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    return " ".join(tweet_words)


# load model
model = TFAutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment")

lables = ['Negative', 'Neutral', 'Positive']

with open('data/target_tweets_split.json') as f:
    data = json.load(f)
    tweets = []
    for tweet in data['data']:
        tweets.append(tweet[2])

    count = 0
    for tweet in tweets:
        if count == 100:
            break
        else:
            tweet = preprocess(tweet)
            inputs = tokenizer(tweet, return_tensors="tf")
            outputs = model(**inputs)
            scores = outputs[0][0].numpy()
            scores = softmax(scores)
            print(f"Tweet: {tweet}")
            print(f"Negative: {scores[0]}")
            print(f"Neutral: {scores[1]}")
            print(f"Positive: {scores[2]}")
            print(f"Sentiment: {lables[outputs.logits.numpy()[0].argmax()]}")
            print('------------------------')
            count += 1
# sentiment analysis
# encoded_tweet = tokenizer(tweet_proc, return_tensors='tf')

# output = model(**encoded_tweet)

# scores = output[0][0].numpy()

# scores = softmax(scores)

# for i in range(len(scores)):
#     print(f"{lables[i]}: {scores[i]:.3f}")
