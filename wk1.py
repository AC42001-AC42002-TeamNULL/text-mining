from tweepy import OAuthHandler, API
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer
from nltk.probability import FreqDist
from itertools import chain
from nltk.classify import NaiveBayesClassifier, accuracy
import collections
from nltk.metrics.scores import precision, f_measure, recall
from nltk import word_tokenize

#######
# SECTION 1
#######

# Connect to twitter API
auth = OAuthHandler('', '')
auth.set_access_token('', '')

api = API(auth)

# Fetch stephenfry's public timeline of tweets == count amount
user_timeline = api.user_timeline('stephenfry', count=1000)

# Build a sentiment analyser and assign tweets to either pos or neg sentiment
sentim_analyzer = SentimentAnalyzer()
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis and construct training and test sets for later classification
tweets = []

# Automatically annotate and label tweets with .2 weighting - found to be best in quick tests

for tweet in user_timeline:
    ss = sid.polarity_scores(tweet.text)
    label = ''

    if ss['compound'] >= .2:
        label = 'pos'

    if ss['compound'] <= -.2:
        label = 'neg'

    if -.2 < ss['compound'] < .2:
        label = 'neu'

    tweets.append([tweet.text, label])

tweet_set = [(word_tokenize(sent), lab) for sent, lab in tweets]

tweet_features = FreqDist(chain(*[f for f, g in tweet_set]))
tweet_features = list(tweet_features.keys())[:100]

# 40 Chosen as only get 139 results back
tweet_train = [({i: (i in tokens) for i in tweet_features}, tag) for tokens, tag in tweet_set[:40]]
tweet_test = [({i: (i in tokens) for i in tweet_features}, tag) for tokens, tag in tweet_set[40:]]

#######
# SECTION 2
#######

# Train classifier with NaiveBayesClassifier
tweet_classifier = NaiveBayesClassifier.train(tweet_train)

tw_refsets = collections.defaultdict(set)
tw_testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(tweet_test):
    tw_refsets[label].add(i)
    observed = tweet_classifier.classify(feats)
    tw_testsets[observed].add(i)

print('Classifier accuracy against twitter data set:', accuracy(tweet_classifier, tweet_test), '\n')

# Compute and print the precision, recall and f-measure(f-score) of both pos and neg precision
print('Classifier pos precision:', precision(tw_refsets['pos'], tw_testsets['pos']))
print('Classifier pos recall:', recall(tw_refsets['pos'], tw_testsets['pos']))
print('Classifier pos F-measure:', f_measure(tw_refsets['pos'], tw_testsets['pos']), '\n')

print('Classifier neg precision:', precision(tw_refsets['neg'], tw_testsets['neg']))
print('Classifier neg recall:', recall(tw_refsets['neg'], tw_testsets['neg']))
print('Classifier neg F-measure:', f_measure(tw_refsets['neg'], tw_testsets['neg']), '\n')

print('Classifier neu precision:', precision(tw_refsets['neu'], tw_testsets['neu']))
print('Classifier neu recall:', recall(tw_refsets['neu'], tw_testsets['neu']))
print('Classifier neu F-measure:', f_measure(tw_refsets['neu'], tw_testsets['neu']), '\n')

#######
# SECTION 3
#######

# ref: https://opensourceforu.com/2016/12/analysing-sentiments-nltk/

man_test = [("Great place to be when you are in Bangalore.", "pos"),
            ("The place was being renovated when I visited so the seating was limited.", "neg"),
            ("Loved the ambiance, loved the food", "pos"),
            ("The food is delicious but not over the top.", "neg"),
            ("Service - Little slow, probably because too many people.", "neg"),
            ("The place is not easy to locate", "neg"),
            ("It was okay, nothing amazing", "neu"),
            ("Horrible experience overall, would never dine again", "neg"),
            ("Simple amazing", "pos"),
            ("Bland but okay", "neu"),
            ("Horrid and dram overall", "neg"),
            ("Mediocre experience", "neu"),
            ("To die for", "pos"),
            ]

# Build our reviews into a testable feature set for our classifier

reviews = []

# Automatically annotate and label tweets with .2 weighting - found to be best in quick tests

for review in man_test:
    ss = sid.polarity_scores(review[0])
    label = ''

    if ss['compound'] >= .2:
        label = 'pos'

    if ss['compound'] <= -.2:
        label = 'neg'

    if -.2 < ss['compound'] < .2:
        label = 'neu'

    reviews.append([review[0], label])

man_set = [(word_tokenize(sent), lab) for sent, lab in man_test]

man_features = FreqDist(chain(*[p for p, o in man_set]))
man_features = list(man_features.keys())[:100]

man_test_set = [({i: (i in tokens) for i in man_features}, tag) for tokens, tag in man_set]

# Test our classifier against our manual data set
print('Classifier accuracy against manual data set:', accuracy(tweet_classifier, man_test_set), '\n')

# Test our manual dataset against the sentiment analyser

correct_sent = 0

for i in range(0, int(len(man_test))):
    if man_test[i][1] == reviews[i][1]:
        correct_sent += 1

print('Sentiment analyser correct guess percentage:', 100 * float(correct_sent) / float(len(man_test)))
