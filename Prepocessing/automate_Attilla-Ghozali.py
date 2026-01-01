import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def preprocess_and_train():
    # Gunakan r"..." agar path tidak merah/error
    input_path = r"C:\Users\USER\Desktop\Eksperimen_SML_Attilla-Ghozali\Dataset_Mobile_Price_Prediction\train.csv"
    output_path = "train_preprocessed.csv"

    # 1. Preprocessing
    df = pd.read_csv(input_path)
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed['price_range'] = y.values
    df_processed.to_csv(output_path, index=False)

    # 2. Training
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Model Berhasil Dilatih!")
    print(f"Akurasi: {accuracy_score(y_test, predictions)}")

if __name__ == "__main__":
    preprocess_and_train()