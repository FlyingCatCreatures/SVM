# Importing the necessary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import time
import joblib
#Time entire program
TrueStart=time.time()

#Load data
start = time.time()
print("Loading Data")
dataset = pd.read_csv("Short.csv") #Not included in github repository as filesize is over 100MB.
end = -1*(start-time.time())
print("Data loaded in " + str(end) + " seconds\n")

#Separate labels
data_labels = np.asarray(dataset.isFraud)

#Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(data_labels)


#Apply encoding to labels
data_labels = label_encoder.transform(data_labels)


#Drop labels from input data
data_selectedColums = dataset.drop(['isFraud', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest' ], axis=1)

#Extract features
start = time.time()
print("Extracting features from dataset")
data_features = data_selectedColums.to_dict(orient='records')
vectorizer = DictVectorizer()
data_features = vectorizer.fit_transform(data_features).toarray()
end =  -1*(start-time.time())
print("Features extracted from dataset in " + str(end) + " seconds\n")

#Split data
features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels, test_size=0.10, random_state=507) #Random_state is just a seed value. Can be any number

# Making the SVM Classifer
Classifier = SVC(kernel="sigmoid", verbose=True, degree=10, cache_size=4096)
#Classifier = SVC(kernel="sigmoid", class_weight="balanced", verbose=True, degree=10, cache_size=4096)

# Training the model 
start = time.time()
print("Fitting SVM")
Classifier.fit(features_train, labels_train)
end =  -1*(start-time.time())
print("SVM fit in " + str(end) + " seconds\n")

# Using the model to predict the labels of the test data
prediction = Classifier.predict(features_test)

# Evaluating the accuracy of the model using the sklearn functions
accuracy = accuracy_score(labels_test,prediction)*100
precision = precision_score(labels_test, prediction)
recall = recall_score(labels_test, prediction)
#confusion_mat = confusion_matrix(labels_test,labels_pred)

# Printing the results
print("----------------------------------------------------------------------------------------------------")
print("Test Accuracy:", accuracy)
print ("Precision:", precision) 
print ("Recall:", recall) 
print("----------------------------------------------------------------------------------------------------")

TrueEndSeconds = -1*(TrueStart-time.time())
TrueEndMinutes = (TrueEndSeconds - TrueEndSeconds%60)/60
print("\n\nTotal runtime: " + str(int(TrueEndMinutes)) + ":" + str(TrueEndSeconds%60))

#Save the classifier
joblib.dump(Classifier, "./Classifier.joblib")

#Example for how to load classifier
#loaded_classifier: SVC = joblib.load("./Classifier.joblib")
