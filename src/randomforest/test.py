from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
accuracy = sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

print("First 5 predictions:", predictions[:5])
print("First 5 actual:     ", list(y_test[:5]))
feature, threshold = tree._best_split(X_train, y_train)
print("Best feature:", feature)
print("Best threshold:", threshold)
