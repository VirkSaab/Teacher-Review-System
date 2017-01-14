import os
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
#np.set_printoptions(threshold=np.nan)  # For full array printing


# IMPORT DATA_____________________________________________________________:
data_read_location = "./data_per_lec/"
data_file_extension = ".txt"
filenames = []
for unused1, unused2, files in os.walk(data_read_location):
    for file in files:
        name = file.split(".")
        if name[0] not in filenames:
            filenames.append(name[0])
dataset = []
for i, filename in enumerate(filenames):
    dataset.append(np.loadtxt(data_read_location+filename+data_file_extension, delimiter=","))
dataset = np.reshape(dataset, (-1,16))
print (dataset.shape)

# Separate features and output
X = dataset[:,:15]
y = np.ravel(dataset[:,15:])


# SPLIT TRAINING AND TESTING DATA_________________________________________:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# APPLY SVM ON TRAINING SET_______________________________________________:
print("Training...")
SVM = svm.SVC()
SVM.fit(X_train,y_train)


# PREDICT WITH TESTING DATA_______________________________________________:
print ("predicting...")
y_pred = SVM.predict(X_test)


# CHECK THE ACCURACY OF YOUR MODEL________________________________________:
accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy =", accuracy)


print("**"*10,"END OF PROGRAM","**"*10)