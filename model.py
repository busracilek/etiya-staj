import pandas as pd

def predict(data):
    
    transactions = data.get('transactions', [])
    
    df = pd.DataFrame(transactions)
    
    if 'amount' not in df.columns:
        raise KeyError("DataFrame'de 'amount' sütunu bulunmuyor.")
    
    df['prediction'] = df['amount'].apply(lambda x: 'high' if x > 100 else 'low')

    return df.to_dict(orient='records')


if __name__ == "__main__":
    test_data = [
        {'transaction_amount': 250},
        {'transaction_amount': 50},
        {'transaction_amount': 400},
        {'transaction_amount': None}  
    ]

    predictions = predict(test_data)
    print(predictions)  
