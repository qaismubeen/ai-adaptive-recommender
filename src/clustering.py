import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def build_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    features = ratings.groupby("user_id").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
        rating_std=("rating", "std")
    ).reset_index()
    features["rating_std"] = features["rating_std"].fillna(0)
    return features

def find_optimal_k(scaled_features, k_range=range(2, 10)):
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_features)
        inertia.append(km.inertia_)
    return list(k_range), inertia

def cluster_users(features: pd.DataFrame, n_clusters: int = 5):
    cols = ["avg_rating", "num_ratings", "rating_std"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features["cluster_id"] = kmeans.fit_predict(scaled)

    return features, kmeans, scaler