import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def build_training_data(ratings: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
    df = ratings.merge(user_features, on="user_id")
    df["liked"] = (df["rating"] >= 4).astype(int)

    item_stats = ratings.groupby("item_id").agg(
        item_avg_rating=("rating", "mean"),
        item_num_ratings=("rating", "count")
    ).reset_index()

    df = df.merge(item_stats, on="item_id")
    return df

def train_logistic_model(df: pd.DataFrame):
    feature_cols = ["avg_rating", "num_ratings", "rating_std", "cluster_id",
                     "item_avg_rating", "item_num_ratings"]

    X = df[feature_cols]
    y = df["liked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model, scaler, feature_cols

def train_neural_model(df: pd.DataFrame):
    feature_cols = ["avg_rating", "num_ratings", "rating_std", "cluster_id",
                     "item_avg_rating", "item_num_ratings"]

    X = df[feature_cols]
    y = df["liked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("Neural Network Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model, scaler, feature_cols