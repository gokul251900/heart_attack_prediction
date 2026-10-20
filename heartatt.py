
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('heart.csv')

for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(le.classes_)
df.to_csv('heartatt.csv', index=False)
print(df.head())

x = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

pred_y = model.predict(x_test)
joblib.dump(model, 'heart_model.pkl')

rep = classification_report(y_test, pred_y)
print("Classification Report:\n", rep)
print("Accuracy Score:", accuracy_score(y_test, pred_y))
print("Predicted values:", pred_y.tolist())