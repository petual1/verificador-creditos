import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('credit_data.csv')

X = data.drop('approved', axis=1)
y = data['approved']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


predictions = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia do modelo: {accuracy:.2f}")

classification_rep = classification_report(y_test, predictions)
print("Relatório de Classificação:\n", classification_rep)


for i, test_sample in enumerate(X_test_scaled):
    prediction = model.predict([test_sample])
    if prediction == 1:
        print(f"Cliente {i+1}: Pode receber crédito.")
    else:
        print(f"Cliente {i+1}: Não pode receber crédito.")
