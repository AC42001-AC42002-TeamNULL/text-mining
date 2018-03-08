from tweepy import OAuthHandler, API

# Connect to twitter API
auth = OAuthHandler('', '')
auth.set_access_token('', '')

api = API(auth)

user_timeline = api.user_timeline('realdonaldtrump', count=50, tweet_mode='extended')

tweets_a = []

for tweet in user_timeline:
    tweets_a.append(tweet.full_text)

with open('trump.txt', 'wb') as tweets:
    for tweet in tweets_a:
        tweets.write(tweet.encode("UTF-8"))
        tweets.write(b'\n')
