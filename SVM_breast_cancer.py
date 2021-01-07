import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# Loading some sklearn data set
cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Adding kernel 'linear' increases accuracy
# with same compilation time
# and it works best for this dataset
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_prediction)

# Print predictions
classes = ['malignant', 'benign']
print("\nComparison:")
for i in range(len(y_prediction)):
	print(f"{i + 1}: {classes[y_prediction[i]]} | {classes[y_test[i]]}")
print(f"Accuracy = {acc:.2%}")
