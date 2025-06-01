# 212952080 資料分析與機器學習_期末報告
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Multiply, Permute, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# --- 技術指標函數 ---
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_MACD(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# --- Attention Layer ---
def attention(inputs):
    # inputs.shape = (batch_size, time_steps, features)
    time_steps = K.int_shape(inputs)[1]
    features = K.int_shape(inputs)[2]

    # 計算每個時間步的注意力權重 (對 time_steps 做 softmax)
    a = Dense(time_steps, activation='softmax')(inputs)  # shape: (batch_size, time_steps, time_steps)

    # 取每個時間步的 attention 權重，對 features 軸做 Permute 來匹配 inputs
    a_probs = Lambda(lambda x: K.mean(x, axis=2), output_shape=(time_steps,))(a)  # shape: (batch_size, time_steps)
    a_probs = Lambda(lambda x: K.expand_dims(x, axis=-1))(a_probs)  # shape: (batch_size, time_steps, 1)

    # 對 inputs 加權
    output_attention_mul = Multiply()([inputs, a_probs])  # shape: (batch_size, time_steps, features)
    return output_attention_mul


# 分類評估
def evaluate_classification(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 迴歸評估
def evaluate_regression(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name} Regression Performance:\nRMSE: {rmse:.6f}, R²: {r2:.4f}")
    plt.figure(figsize=(8,4))
    plt.plot(y_test[:100], label='True')
    plt.plot(preds[:100], label='Predicted')
    plt.title(f"{name} - True vs Predicted")
    plt.legend()
    plt.show()

# 聚類評估
def evaluate_clustering(X, labels, model_name):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(f"{model_name} Clustering Result (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def clustering_analysis(X_scaled):
    print("\n=== Clustering Models ===")
    models = [
        ("KMeans", KMeans(n_clusters=2, random_state=42)),
        ("DBSCAN", DBSCAN(eps=1.5, min_samples=5)),
        ("Agglomerative", AgglomerativeClustering(n_clusters=2))
    ]

    for name, model in models:
        labels = model.fit_predict(X_scaled)
        print(f"{name} Unique Clusters:", np.unique(labels))
        evaluate_clustering(X_scaled, labels, name)

# 建立序列資料
def create_seq_data(df, features, target_cls, target_reg, seq_len=10):
    data = df[features].values
    targets_cls = df[target_cls].values
    targets_reg = df[target_reg].values
    X_seq, y_seq_cls, y_seq_reg = [], [], []
    for i in range(len(df) - seq_len):
        X_seq.append(data[i:i+seq_len])
        y_seq_cls.append(targets_cls[i+seq_len])
        y_seq_reg.append(targets_reg[i+seq_len])
    return np.array(X_seq), np.array(y_seq_cls), np.array(y_seq_reg)

def main():
    print("Downloading S&P 500 data...")
    data = yf.download("^GSPC", start="2010-01-01", end="2023-01-01")

    # 技術指標與目標
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Return'] = data['Close'].pct_change()
    data['RSI'] = compute_RSI(data['Close'])
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = compute_MACD(data['Close'])

    N = 5
    data['Return_N'] = data['Close'].pct_change(periods=N).shift(-N)
    data['Target_N'] = (data['Return_N'] > 0).astype(int)
    data = data.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
    X = data[features]
    y_class = data['Target_N']
    y_reg = data['Return_N']

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 分類 ---
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    print("\n=== Classification Models ===")
    evaluate_classification(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest", X_train, y_train_cls, X_test, y_test_cls)
    evaluate_classification(LogisticRegression(max_iter=1000), "Logistic Regression", X_train, y_train_cls, X_test, y_test_cls)
    evaluate_classification(KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors", X_train, y_train_cls, X_test, y_test_cls)

    # --- 迴歸 ---
    print("\n=== Regression Models ===")
    evaluate_regression(RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest Regressor", X_train, y_train_reg, X_test, y_test_reg)
    evaluate_regression(LinearRegression(), "Linear Regression", X_train, y_train_reg, X_test, y_test_reg)
    evaluate_regression(SVR(), "Support Vector Regressor", X_train, y_train_reg, X_test, y_test_reg)

    # --- 聚類 ---
    clustering_analysis(X_scaled)

    # --- 時序模型 ---
    seq_len = 10
    X_seq, y_seq_cls, y_seq_reg = create_seq_data(data.reset_index(drop=True), features, 'Target_N', 'Return_N', seq_len)
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq_reg, y_test_seq_reg = y_seq_reg[:split_idx], y_seq_reg[split_idx:]

    # LSTM
    model_lstm = Sequential([
        LSTM(64, input_shape=(seq_len, len(features))),
        Dense(1)
    ])
    model_lstm.compile(optimizer=Adam(0.001), loss='mse')
    print("\nTraining LSTM model...")
    model_lstm.fit(X_train_seq, y_train_seq_reg, epochs=10, batch_size=32, verbose=1)

    preds_lstm = model_lstm.predict(X_test_seq)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_seq_reg, preds_lstm))
    print(f"LSTM Regression RMSE: {rmse_lstm:.4f}")

    plt.figure(figsize=(10,4))
    plt.plot(y_test_seq_reg[:100], label='True')
    plt.plot(preds_lstm[:100], label='Predicted')
    plt.title("LSTM Prediction (Return_N)")
    plt.legend()
    plt.show()

    # GRU + Attention
    input_layer = Input(shape=(seq_len, len(features)))
    gru_out = GRU(64, return_sequences=True)(input_layer)
    att_out = attention(gru_out)
    flat = Flatten()(att_out)
    output = Dense(1)(flat)

    model_gru_att = Model(inputs=input_layer, outputs=output)
    model_gru_att.compile(optimizer=Adam(0.001), loss='mse')

    print("\nTraining GRU + Attention model...")
    model_gru_att.fit(X_train_seq, y_train_seq_reg, epochs=10, batch_size=32, verbose=1)

    preds_gru_att = model_gru_att.predict(X_test_seq)
    rmse_gru_att = np.sqrt(mean_squared_error(y_test_seq_reg, preds_gru_att))
    print(f"GRU + Attention Regression RMSE: {rmse_gru_att:.4f}")

    plt.figure(figsize=(10,4))
    plt.plot(y_test_seq_reg[:100], label='True')
    plt.plot(preds_gru_att[:100], label='Predicted')
    plt.title("GRU + Attention Prediction (Return_N)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
