import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd


# DATA MANIPULATION
data = pd.read_csv("data/car.data")
# Must be encoded to numeric values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_ = le.fit_transform(list(data["class"]))


# TRAIN AND PREDICT
predict = "class"
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(class_)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=9)  # should find OPTIMAL count of neighbours
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"\nAccuracy = {acc:.2%}")
predicted = model.predict(x_test)


# GET RESULTS
names = ["unacc", "acc", "good", "vgood"]  # Decode numeric results
errs = {}
print("\nComparison:")
for i in range(len(predicted)):
	print(f"{i + 1}: {names[predicted[i]]} | {names[y_test[i]]}")
	if not names[predicted[i]] == names[y_test[i]]:
		errs[i] = {"pred": names[predicted[i]], "act": names[y_test[i]]}

print("\nErrors:")
for n, value in enumerate(errs.values()):
	print(f"Err#{n}: {value['pred']} | {value['act']}")
print(f"\nAccuracy = {acc:.2%}")
