import pandas as pd
import numpy as np
import sklearn as skl
# from sklearn import linear_model
# from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("./data/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.1)

# best = 0
# for _ in range(10000):
# 	x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.1)
#
# 	linear = linear_model.LinearRegression()
#
# 	linear.fit(x_train, y_train)
#
# 	acc = linear.score(x_test, y_test)
#
# 	if acc > best:
# 		best = acc
# 		print(best)
# 		with open("student_model.pickle", "wb") as f:
# 			pickle.dump(linear, f)

pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

# print("Co:", linear.coef_)
# print("Intercept:", linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
	print(predictions[i], x_test[i], y_test[i])

p = "absences"  # "G1", "G2", "studytime", "failures", "absences"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
