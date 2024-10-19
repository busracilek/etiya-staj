import logging
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from logging.handlers import RotatingFileHandler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

app.config['JWT_SECRET_KEY'] = 'super-secret-key'  
jwt = JWTManager(app)


if __name__ != "__main__":
    handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.ERROR)
    app.logger.addHandler(handler)

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Bir hata oluştu: {e}")
    return jsonify(error=str(e)), 500

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    if username != 'admin' or password != 'password':
        return jsonify({"msg": "Kullanıcı adı veya şifre hatalı"}), 401
    
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/')
def home():
    return 'Merhaba'


@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    return jsonify(message="Bu rota korumalı ve JWT ile erişildi!")

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_route():
    try:
        data = request.get_json()
        account_id = data['accountId']  
        transactions = data['transactions'] 
        
        predictions = predict(transactions)  
        
        return jsonify({'predictions': predictions})  
    except KeyError as e:
        return jsonify(error=f"Gerekli veri eksik: {str(e)}"), 400
    except Exception as e:
        return jsonify(error=str(e)), 500



def generate_data(num_entries):
    timestamps = [datetime.now() - timedelta(minutes=30*i) for i in range(num_entries)]
    amounts = [round(100 * i, 2) for i in range(num_entries)]
    types = ['debit' if i % 2 == 0 else 'credit' for i in range(num_entries)]
    
    data = {
        'timestamp': [ts.isoformat() for ts in timestamps],
        'amount': amounts,
        'type': types
    }
    
    df = pd.DataFrame(data)
    df.to_csv('transactions.csv', index=False)
    print("transactions.csv dosyası başarıyla oluşturuldu.")


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print("Orijinal veri türleri:")
    print(df.dtypes)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9  
    
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'])
    
    bool_cols = df.select_dtypes(include=['bool']).columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)
    
    if df.isnull().sum().any():
        print("Eksik veriler bulundu. Eksik verileri dolduruyoruz.")
        df = df.fillna(0)
    
    print("Dönüştürülmüş veri türleri:")
    print(df.dtypes)
    
    return df


def train_model(df):
    if 'prediction' in df.columns:
        X = df.drop(['prediction'], axis=1)
        y = df['prediction']
    else:
        raise KeyError("DataFrame'de 'prediction' sütunu bulunamadı.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'model.pkl')
    return model

def predict(data):
    
    model = joblib.load('model.pkl')
    
    df = pd.DataFrame(data['transactions'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9  
    df = pd.get_dummies(df, columns=['type'])
    
    
    if 'type_credit' not in df.columns:
        df['type_credit'] = 0
    if 'type_debit' not in df.columns:
        df['type_debit'] = 0
    
    predictions = model.predict(df)
    
    df['prediction'] = predictions
    
    response = []
    for i, row in df.iterrows():
        result = {
            "timestamp": datetime.fromtimestamp(row['timestamp']).isoformat(),
            "amount": row['amount'],
            "type": "debit" if row['type_debit'] else "credit",
            "prediction": "high" if row['prediction'] == 1 else "low"
        }
        response.append(result)
    
    return response

@app.route('/train', methods=['POST'])
@jwt_required()
def train():
    try:

        generate_data(100)
        
        df = preprocess_data(r'C:\Users\ymö\Desktop\project-root\transactions.csv')
        
        df['prediction'] = df['amount'].apply(lambda x: 1 if x > 1000 else 0)
        
        model = train_model(df)
        
        return jsonify({"message": "Model başarıyla eğitildi ve kaydedildi."})
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
