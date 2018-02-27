import requests
import bs4
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer

sentim_analyzer = SentimentAnalyzer()

trainer = NaiveBayesClassifier.train

res = requests.get(
    'https://www.amazon.com/JBL-Everest-700-Around-Ear-Refurbished/product-reviews/B079DFZZ6C/ref=cm_cr_getr_d_paging_btm_2?ie=UTF8&reviewerType=all_reviews&pageNumber=1')
res.raise_for_status()
noStarchSoup = bs4.BeautifulSoup(res.text, 'lxml')

linkElems = noStarchSoup.select('.review-text')

sid = SentimentIntensityAnalyzer()

length = len(linkElems)

imems = []

for i in range(0, length):
    ss = sid.polarity_scores(linkElems[i].getText())
    print(linkElems[i].getText())
    imems.append(linkElems[i].getText())
    print(ss)

# classifier = sentim_analyzer.train(trainer, imems)
#
# for key, value in sorted(sentim_analyzer.evaluate(imems).items()):
#     print('{0}: {1}'.format(key, value))

train = [("Great place to be when you are in Bangalore.", "pos"),
         ("The place was being renovated when I visited so the seating was limited.", "neg"),
         ("Loved the ambience, loved the food", "pos"),
         ("The food is delicious but not over the top.", "neg"),
         ("Service - Little slow, probably because too many people.", "neg"),
         ("The place is not easy to locate", "neg"),
         ("Mushroom fried rice was spicy", "pos"),
         ]
