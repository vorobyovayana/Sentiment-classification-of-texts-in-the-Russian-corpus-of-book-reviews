
# coding: utf-8

# In[50]:

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import os
import pandas as pd
import io



def get_data(dir_list):
    """
    This method allows us to write our corpus into a dictionary.
    @param a list of data directories.
    @return dictionaries with train and test reviews.
    """
    with open('../swl.txt', 'r', encoding = 'utf-8') as f:
        stop_words = f.read()   
    train_reviews = {'review': [], 'label': []}
    test_reviews = {'review': [], 'label': []}
    for directory in dir_list:
        for review in os.listdir(directory):
            clean_review = ''
            if review != '.ipynb_checkpoints':
                review_body = open(directory + '/' + review, 'r', encoding = 'utf-8').read()
                #print(type(review_body))
                if 'bad' in review:
                    label = '-1'
                elif 'good' in review:
                    label = '1'                  
                else:
                    print('error!')
                # Pick out the stop words from the review.
                for word in review_body.split():                    
                    if word not in stop_words:
                        clean_review += word + ' '
                    
            if 'train' in directory:
                train_reviews['review'].append(clean_review)
                train_reviews['label'].append(label)
            if 'test' in directory:
                test_reviews['review'].append(clean_review)
                test_reviews['label'].append(label)       
    return (train_reviews, test_reviews)


if __name__ == '__main__':
    
    # Data directory
    dir_list = ['binary_twice_merged_corpus/train', 'binary_twice_merged_corpus/test']
    dir_list2 = ['./binary_full_corpus/train', './binary_full_corpus/test']
    #dir_list2 = ['lemmatized_corpus_no_punct/train', 'lemmatized_corpus_no_punct/test']
    #dir_list2 = ['./relevant_sentences_corpus/train', './relevant_sentences_corpus/test']
    data1 = get_data(dir_list)    
    data2 = get_data(dir_list2)
    


# In[51]:


train_reviews = data1[0]
test_reviews =  data1[1]

train_reviews2 = data2[0]
test_reviews2 =  data2[1]

X_train = train_reviews['review']
X_test = test_reviews['review']
y_train = train_reviews['label']
y_test = test_reviews['label']


x_train2 = train_reviews2['review']
X_test2 = test_reviews2['review']
y_train2 = train_reviews2['label']
y_test2 = test_reviews2['label']

# Train and test SVM classificator
print('SVM')
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test2)

print(metrics.classification_report(y_test2, predicted))


   
# Train and test Naive Bayes classificator
print('MultinomialNB')
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test2)

print(metrics.classification_report(y_test2, predicted))

# Train and test K-nearest neighbors classificator
print('KNeighbor')
from sklearn.neighbors import KNeighborsClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test2)

print(metrics.classification_report(y_test2, predicted))

# Train and test Nearest Centroid classificator
print('Nearest Centroid')
from sklearn.neighbors.nearest_centroid import NearestCentroid

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', NearestCentroid()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))


    





# In[ ]:




# In[ ]:



