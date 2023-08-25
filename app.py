from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

@app.route('/')
def index():
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
    classification_rep = classification_report(y_test, predictions)

    client_predictions = []
    for i, test_sample in enumerate(X_test_scaled):
        prediction = model.predict([test_sample])
        if prediction == 1:
            client_predictions.append(f"Cliente {i+1}: Pode receber crédito.")
        else:
            client_predictions.append(f"Cliente {i+1}: Não pode receber crédito.")

    return render_template('index.html', accuracy=accuracy, classification_report=classification_rep, client_predictions=client_predictions)

if __name__ == '__main__':
    app.run(debug=True)
