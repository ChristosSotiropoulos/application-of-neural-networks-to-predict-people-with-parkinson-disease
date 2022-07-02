# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import keras
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score,recall_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from ann_visualizer.visualize import ann_viz

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


#Data preprocesing

# Importing the dataset
dataset = pd.read_csv('parkinsons.csv')

# Checking for missing values
dataset.isnull().any().sum()


X = dataset.iloc[:, 1:23].values
y = dataset.iloc[:, 23].values



# Splitting the dataset into the Training set and Test set (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Splitting the dataset into the Training set and Test set (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X = sc.fit_transform(X)

#Min max scaling
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
X = mms.fit_transform(X)

#Dimensionality reduction

#PCA
pca = PCA(n_components= None)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
X = pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_

#LDA
lda = LDA(n_components= 3)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.fit_transform(X_test,y_test)
X = lda.fit_transform(X,y)


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim  = 26, init = 'uniform', activation = 'relu', input_dim = 22))
classifier.add(Dropout(0.1))

# Adding the second hidden layer and the third layer
classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

#classifier.add(Dropout(dropout_rate))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Fix the optimize values
optimizer = keras.optimizers.Adam(lr  = 0.002)

# Compiling the ANN
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy','precision', 'recall' ])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 150)


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Making the Classification Report
cr = classification_report(y_test,y_pred)
print(cr)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)

# Part 4 - Evaluating, improving and Turning the ANN

# Evaluating the Ann

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 26, init = 'uniform', activation = 'relu', input_dim = 22))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    optimizer = keras.optimizers.Adam(lr  = 0.002)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy','precision', 'recall' ])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier,batch_size = 20,nb_epoch=150)

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score)}

results = cross_validate(estimator=classifier,X=X, y=y,cv=10,scoring=scoring)

results = cross_validate(estimator=classifier,X=X, y=y,cv=5,scoring=scoring)

print('10 Fold results of training set')
print('Training Accuracy mean:',results['train_accuracy'].mean())
print('Training Precision mean:',results['train_precision'].mean())
print('Training Recall mean:',results['train_recall'].mean())


print('10 Fold results of test set')
print('Accuracy mean:',results['test_accuracy'].mean())
print('Precision mean:',results['test_precision'].mean())
print('Recall mean:',results['test_recall'].mean())



# Improving the Ann
#Dropout Regularization to reduce overfitting if need

#Tunning the ANN
def build_classifier(dropout_rate1,dropout_rate2):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 26, init = 'uniform', activation = 'relu', input_dim = 22))
    classifier.add(Dropout(dropout_rate1))
    classifier.add(Dense(output_dim = 22, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(dropout_rate2))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    optimizer = keras.optimizers.Adam(lr  = 0.002)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy','precision', 'recall' ])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,batch_size = 20,nb_epoch=150)

#Tunning batch and epochs

parameters = {'batch_size' : [20,22,25],
              'nb_epoch' : [150,180 ,200],
             }

#Tunning activation function
parameters = {'batch_size' : [25],
              'nb_epoch' : [500],
              'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}

#Tunning SGD
parameters = {'learn_rate' : [0.001, 0.002, 0.003, 0.004, 0.0005]}


#Number of Neurons
parameters ={'neurons1':[20,22,26,30,32],
             'neurons2':[20,22,26,30,32]}

#Νumber of Dropout
parameters ={'dropout_rate1' : [ 0.1,0.2,0.3,0.5],
             'dropout_rate2' : [ 0.1,0.2,0.3, 0.5]}

grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_

# summarize results
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



#Graphs and plots


print(history.history['recall'])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])


#Plot Graphs
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['precision'])
plt.title('Model precision')
plt.ylabel('precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.title('Model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

