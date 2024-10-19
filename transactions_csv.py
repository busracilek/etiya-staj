import pandas as pd
from datetime import datetime, timedelta

# Veri oluşturma fonksiyonu
def generate_data(num_entries):
    # Veriler için gerekli alanları oluşturun
    timestamps = [datetime.now() - timedelta(minutes=30*i) for i in range(num_entries)]
    amounts = [round(100 * i, 2) for i in range(num_entries)]
    types = ['debit' if i % 2 == 0 else 'credit' for i in range(num_entries)]
    
    # Veriyi DataFrame formatında oluşturun
    data = {
        'timestamp': [ts.isoformat() for ts in timestamps],
        'amount': amounts,
        'type': types
    }
    
    df = pd.DataFrame(data)
    return df

# Veri setini oluştur
df = generate_data(10)  # 10 örnek veri

# DataFrame'i CSV dosyasına kaydedin
df.to_csv('transactions.csv', index=False)

print("transactions.csv dosyası başarıyla oluşturuldu.")
