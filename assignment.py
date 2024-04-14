import pandas as pd
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

features = ['Constituency ∇', 'Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']
target = 'Education'

combined = pd.concat([df[features], df_test[features]])

combined = pd.get_dummies(combined)

X_train = combined[:len(df)]
X_test = combined[len(df):]

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

y_train = df[target]
rf_classifier.fit(X_train, y_train)

predictions = rf_classifier.predict(X_test)

submission_df = pd.DataFrame({'ID': df_test['ID'], 'Education': predictions})
submission_df.to_csv('submission_random_forest.csv', index=False)
