import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

from Clean_data import clean_data

split = .8

estimators = 200
depth = 4
bstrap = True
state = 12345
jobs = 4

print("Loading data...")
data1,g = clean_data("C:\Development\Python\TNAP\DATA_NEG",750)
data2,g = clean_data("C:\Development\Python\TNAP\DATA_POS",750)


label1 = np.zeros(5000)
label2 = np.ones(5000)

data = np.concatenate((data1,data2))
labels = np.concatenate((label1,label2))

print("Permutating data...")
p = np.random.permutation(data.shape[0])
data = data[p]
labels = labels[p]

print("Splitting train and test data...")
train_data = data[0:int(data.shape[0]*split)]
train_labels = labels[0:int(labels.shape[0]*split)]

val_data = data[int(data.shape[0]*split):]
val_labels = labels[int(labels.shape[0]*split):]

print("Fitting classifier...")
start = time.time()
RForest = RandomForestClassifier(n_estimators=estimators,max_depth=depth,bootstrap=bstrap,random_state=state)
RForest.fit(train_data,train_labels)
print(time.time()-start)

print("Saving model")
pickle.dump(RForest, open('trained_model.sav','wb'))

print("Predicting...")
preds = RForest.predict(val_data)

print("Evaluating predictions...")
num_preds = 0
num_correct = 0
for i in range(val_data.shape[0]):
    if(preds[i] == val_labels[i]):
        num_correct+=1
    num_preds+=1

print(num_correct/num_preds)