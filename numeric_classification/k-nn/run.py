import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# My implementation of k-nn
from KNeighborsClassifier import KNeighborsClassifier

def main():
	ds = load_iris()
	train = pd.DataFrame(ds.data)
	target = pd.Series(ds.target)
	train.columns = ['A','B','C','D']

	X_train, X_test, y_train, y_test = train_test_split(
	    train,
	    target,
	    test_size=0.25)

	clf = KNeighborsClassifier(num_neighbors=3)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	print("accuracy: "+str(accuracy_score(preds, y_test)))

if __name__ == "__main__":
    main()