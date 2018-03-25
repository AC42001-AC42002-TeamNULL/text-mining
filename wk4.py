import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.metrics.scores import *
from nltk.metrics import (accuracy as eval_accuracy, precision as eval_precision,
        recall as eval_recall, f_measure as eval_f_measure)

# 3644 / 911 / 2733
'''
trainData = pd.read_csv('./HansardData.csv',
                        delimiter=',',
                        encoding='latin1',
                        header=0,
                        )
pds = trainData.to_dict()
# print(pds['IsEthotic'])

print(pds)
'''

sid = SentimentIntensityAnalyzer()

# Get training data
trainData = pd.read_csv('./train.csv',
                        delimiter=',',
                        encoding='latin1',
                        header=0,
                        )
trainDict = trainData.to_dict()

# Get test data
testData = pd.read_csv('./test.csv',
                        delimiter=',',
                        encoding='latin1',
                        header=0,
                        )
testDict = testData.to_dict()

pipeline = Pipeline([('BagOfWords', CountVectorizer()), ('classifier', MultinomialNB())])
model = pipeline.fit(trainData['Text'], trainData['IsEthotic'])
predictions = model.predict(testData['Text']) # TODO SPLIT DATA

target_names = ['0', '1']

print('Ethotic Statement Classifier Confusion Matrix')
print(confusion_matrix(testData['IsEthotic'], predictions))
print('Ethotic Statement Classifier Accuracy')
print(accuracy_score(testData['IsEthotic'], predictions) * 100, '%\n')
print('Ethotic sentences with VADER sentiment classifier')

reference_set = []
test_set = []
for index, item in enumerate(testData['Text']):
    if predictions[np.where(item == testData['Text'])].any() == 1:
        print(item)
        reference_set.append(testData['SentimentPolarity'][index])
        #print(testData['SentimentPolarity'].keys()[index])
        polarity_score = sid.polarity_scores(item)
        for j in sorted(polarity_score):
            print('{0}: {1}, \n'.format(j, polarity_score[j]), end='')
        if polarity_score['compound'] > 0:
            test_set.append(1)
        else:
            test_set.append(2)

accuracy_score = eval_accuracy(reference_set, test_set)
reference_set = set(reference_set)
test_set = set(test_set)
precision_score = eval_precision(reference_set, test_set)
recall_score = eval_recall(reference_set, test_set)
f_measure_score = eval_f_measure(reference_set, test_set)
print('VADER Classification Statistics')
print('Accuracy: ',accuracy_score * 100, '%')
print('Precision: ', precision_score)
print('Recall: ', recall_score)
print('F-measure', f_measure_score)
