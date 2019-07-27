#import useful liabrary
import numpy as np 
import pandas as pd 
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split 
#import dataset
dataset = pd.read_csv('spam.csv',encoding='latin-1')
#drop unuse column and rename previous column
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"label", "v2":"sms"})
#now we change spam to 1 and ham to 0
dataset.loc[:,'label']=dataset["label"].replace("ham",0)
dataset.loc[:,'label']=dataset["label"].replace("spam",1)
#now we apply bag of word approach to find feature matrix
corpus=[] 
#apply fn to remove stop word
for i in range(0, 5572):
    clean_data = re.sub('[!]', ' ', dataset['sms'][i])
    clean_data = clean_data.lower()
    clean_data = clean_data.split()
    clean_data = [word for word in clean_data if not word in set(stopwords.words('english'))]
    clean_data = ' '.join(clean_data)
    corpus.append(clean_data) 
#import countvectoriser
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
features = cv.fit_transform(corpus).toarray()
labels = dataset.iloc[:, 1].values
#split the data training ,testing
X_train, X_test, y_train, y_test = train_test_split(features, dataset['label'], test_size=0.25, random_state=0)
 #import multinomial model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.2)
mnb.fit(X_train,y_train)
predictions = mnb.predict(X_test)
#accuracy is 98%       
#joblib and pickle have same fuctionality,but when y have more numpy array then joblib is best.
from sklearn.externals import joblib
joblib.dump(mnb, 'best.pkl')
joblib.dump(cv, 'best1.pkl')
best1 = joblib.load('best1.pkl')
best = joblib.load('best.pkl')

def find(p):
    if p == 1:
        print ("Message is SPAM")
    else:
        print ("Message is NOT Spam")
text4 = ["WINNER!! You just won a free ticket to Bahamas. Send your Details"]
I = best1.transform(text4)
p = best.predict(I)
find(p)