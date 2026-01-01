import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(
    input_path: str,
    output_path: str
):
    df = pd.read_csv(input_path)

    X = df.drop('price_range', axis=1)
    y = df['price_range']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )
    df_processed['price_range'] = y.values

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_processed.to_csv(output_path, index=False)

    return df_processed


if __name__ == "__main__":
    preprocess_data(
        input_path=r"C:\Users\USER\Desktop\Eksperimen_SML_Attilla-Ghozali\Dataset_Mobile_Price_Prediction\train.csv",
        output_path="train_preprocessed.csv"
    )