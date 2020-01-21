import sys
print("python: {}" .format(sys.version))
import scipy
print("scipy : {}".format(scipy.__version__))
import numpy
print("numpy: {}".format(numpy.__version__))
import matplotlib
print("matplotlib: {}".format(matplotlib.__version__))
import pandas
print("pandas: {}".format(pandas.__version__))
import sklearn
print("sklearn: {}".format(sklearn.__version__))

#load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load dataset
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

#bivariate plots
# box and whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()

#histograms
dataset.hist()
pyplot.show()

#multivariate plots
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

#split-out validation dataset
array=dataset.values
X=array[:,0:4]
y=array[:,4]
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=0.20,random_state=1)



#spot check algorithms
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
#evaluate each model in turn
results=[]
names=[]
for name ,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' %(name,cv_results.mean(),cv_results.std()))

#compare algorithms
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparison');
pyplot.show()

#make predictions on validation dataset
model=SVC(gamma='auto')
model.fit(X_train,y_train)
predictions=model.predict(X_validation)

#evaluate predictions
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))