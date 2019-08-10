import numpy as np
import pandas as pd


dataset = pd.read_csv('stock.csv',encoding="ISO-8859-1")

#print(dataset)

train = dataset[dataset['Date']<'20150101']


test = dataset[dataset['Date']>'20150101']

dataset1 = train.iloc[:,2:27]

#print(type(data.columns))


dataset1.replace("[^a-zA-Z]"," ",regex = True,inplace=True)

list1 = [i for i in range(25)]

new_index = [str(i) for i in list1]

dataset1.columns = new_index

#print(dataset1.head(5))


#print(type(new_index))


for index in new_index:

    dataset1[index] = dataset1[index].str.lower()



#print(dataset1.head(5))


headlines = []

for row in range(0,len(dataset1.index)):

    headlines.append(' '.join(str(x) for x in dataset1.iloc[row,0:25]))

#print(headlines[0])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)



