import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

features = ['Constituency ∇', 'Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']
target = 'Education'

combined_data = pd.concat([data_train[features], data_test[features]])

combined_data = pd.get_dummies(combined_data)

X_train = combined_data[:len(data_train)]
X_test = combined_data[len(data_train):]

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

y_train = data_train[target]
random_forest_classifier.fit(X_train, y_train)

predictions = random_forest_classifier.predict(X_test)

submission_data = pd.DataFrame({'ID': data_test['ID'], 'Education': predictions})
submission_data.to_csv('submission_random_forest.csv', index=False)
