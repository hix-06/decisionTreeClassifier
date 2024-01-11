import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from six import StringIO  
from IPython.display import Image  
import pydotplus


path = r"C:\Users\Ahmed\Downloads\Iris.csv" 
'''
I downloaded the same dataset provided in the task on my PC and uploaded it
from my device due to the shortened link non-compliance.
'''
data = pd.read_csv(path)

print(data.head())

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=X.columns, class_names=y.unique())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

