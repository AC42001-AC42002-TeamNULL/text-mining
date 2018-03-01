from tweepy import OAuthHandler, API
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer

auth = OAuthHandler('', '')
auth.set_access_token('', '')

api = API(auth)

user_timeline = api.user_timeline('stephenfry', count=100)

sentim_analyzer = SentimentAnalyzer()
sid = SentimentIntensityAnalyzer()

for tweet in user_timeline:
    ss = sid.polarity_scores(tweet.text)
    print(tweet.text)
    print(ss, '\n')
