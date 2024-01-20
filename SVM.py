# Importing the necessary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import time
import joblib


TRAINING_SIZE = 300000

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
#features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels, test_size=1-(TRAINING_SIZE/6362599), random_state=507, shuffle=True) #Random_state is just a seed value. Can be any number
features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels, test_size=0.2, random_state=507, shuffle=True) #Random_state is just a seed value. Can be any number

i=0
array=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
while i<26:
    CLASS_WEIGHTS = {0:1.0, 1:0.25*i + 53}

    # Making the SVM Classifer
    Classifier = SVC(kernel="rbf", verbose=False, cache_size=4096, class_weight=CLASS_WEIGHTS)
    #Classifier = SVC(kernel="sigmoid", class_weight="balanced", verbose=True, degree=10, cache_size=4096)

    # Training the model 
    start = time.time()
    print("Fitting SVM")
    Classifier.fit(features_train, labels_train)
    end =  -1*(start-time.time())
    print("SVM fit in " + str(end) + " seconds on i=" + str(i))

    #start = time.time()
    #print("Calculating metrics")
    # Using the model to predict the labels of the test data
    prediction = Classifier.predict(features_test)

    # Evaluating the accuracy of the model using the sklearn functions
    accuracy = accuracy_score(labels_test,prediction)*100
    precision = precision_score(labels_test,prediction, zero_division=0)*100
    recall = recall_score(labels_test,prediction)*100
    #precision, recall, f_score = precision_recall_fscore_support(labels_test, prediction, average='weighted')
    #end =  -1*(start-time.time())
    #print("Metrics calculated in " + str(end) + " seconds")

    # Printing the results
    #print("----------------------------------------------------------------------------------------------------")
    #print("Test Accuracy: ", accuracy)
    #print ("Precision: ", precision) 
    #print ("Recall: ", recall) 
    #print("F-score: ", f_score)
    #print("----------------------------------------------------------------------------------------------------")

    #Save the classifier
    #joblib.dump(Classifier, "./Classifier.joblib")

    i+=1

    #Example for how to load classifier
    #loaded_classifier: SVC = joblib.load("./Classifier.joblib")

    array[i] = (accuracy, precision, recall)
    del Classifier, prediction, accuracy, recall, precision
    
print("\n\n")
i=0
while i<26:
    inte = 0.25*i + 53
    print("Test accuracy, precision, recall, with ratio 0:1, 1: " + str(inte) + " : ", array[i][0], "   ", array[i][1], "    ", array[i][2])
    i+=1

TrueEndSeconds = -1*(TrueStart-time.time())
TrueEndMinutes = (TrueEndSeconds - TrueEndSeconds%60)/60
print("Total runtime: " + str(int(TrueEndMinutes)) + ":" + str(TrueEndSeconds%60))