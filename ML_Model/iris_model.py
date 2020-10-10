# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
dataset.head()

dataset.shape

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.plot()
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

array = dataset.values
X = array[:,0:4]
Y=array[:,4]
validation_size = 0.20
seed = 776
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(DecisionTreeClassifier(), X_train, Y_train, cv=kfold, scoring=scoring)
results.append(cv_results)
names.append('DecisionTreeClassifier')
msg = "%s: %f (%f)" % ('DecisionTreeClassifier', cv_results.mean(), cv_results.std())
print(msg)

# Make predictions on validation dataset
dtc = DecisionTreeClassifier()
iris_train = dtc.fit(X_train, Y_train)
predictions = dtc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))

accuracy_score(Y_validation, predictions)

iris_train.predict([[4.7,3.2,1.3,0.2]])

