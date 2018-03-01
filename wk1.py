from tweepy import OAuthHandler, API
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
import string
from itertools import chain
from nltk.corpus import movie_reviews as mr
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier as nbc
from nltk import word_tokenize, classify

# Modified version of
# https://stackoverflow.com/questions/21107075/classification-using-movie-review-corpus-in-nltk-python/21126594#21126594
# Required changes due to using python3

stop = stopwords.words('english')
documents = [
    ([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0]) for i
    in mr.fileids()]

word_features = FreqDist(chain(*[i for i, j in documents]))
word_features = list(word_features.keys())[:100]

numtrain = int(len(documents) * 90 / 100)
mr_train = [({i: (i in tokens) for i in word_features}, tag) for tokens, tag in documents[:numtrain]]
mr_test = [({i: (i in tokens) for i in word_features}, tag) for tokens, tag in documents[numtrain:]]

mr_classifier = nbc.train(mr_train)

print('Accuracy against movie review training set:')
print(classify.accuracy(mr_classifier, mr_test), '\n')

auth = OAuthHandler('', '')
auth.set_access_token('', '')

api = API(auth)

user_timeline = api.user_timeline('stephenfry', count=1000)

sentim_analyzer = SentimentAnalyzer()
sid = SentimentIntensityAnalyzer()

tweets = []

for tweet in user_timeline:
    ss = sid.polarity_scores(tweet.text)
    label = ''

    if ss['compound'] >= .5:
        label = 'pos'

    if ss['compound'] <= -.5:
        label = 'neg'

    if -.5 < ss['compound'] < .5:
        label = 'neu'

    tweets.append([tweet.text, label])

tweet_set = [(word_tokenize(sent), lab) for sent, lab in tweets]

tweet_features = FreqDist(chain(*[f for f, g in tweet_set]))
tweet_features = list(tweet_features.keys())[:100]

tweet_train = [({i: (i in tokens) for i in tweet_features}, tag) for tokens, tag in tweet_set[:100]]
tweet_test = [({i: (i in tokens) for i in tweet_features}, tag) for tokens, tag in tweet_set[100:]]

tweet_classifier = nbc.train(tweet_train)

print('Accuracy against twitter data set:')
print(classify.accuracy(tweet_classifier, tweet_test), '\n')

print('Cross classifier accuracy:')
print('Tweet classifier on mr set', classify.accuracy(tweet_classifier, mr_test))
print('Mr classifier on tweet set', classify.accuracy(mr_classifier, tweet_test))
