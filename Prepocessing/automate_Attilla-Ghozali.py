import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    X = df.drop('price_range', axis=1)
    y = df['price_range']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed['price_range'] = y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    return df_processed


if __name__ == "__main__":
    # Lokasi file automate ini
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path dataset (RELATIVE, bukan Windows path)
    input_csv = os.path.join(
        BASE_DIR,
        "..",
        "Dataset_Mobile_Price_Prediction",
        "train.csv"
    )

    # Output hasil preprocessing
    output_csv = os.path.join(
        BASE_DIR,
        "mobile_price_preprocessing",
        "train_preprocessed.csv"
    )

    preprocess_data(input_csv, output_csv)
