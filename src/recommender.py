import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)

def train_svd_model(user_item_matrix: pd.DataFrame, n_components: int = 20):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_

    reconstructed = np.dot(user_factors, item_factors)
    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    return reconstructed_df, svd

def recommend_top_n(user_id: int, reconstructed_df: pd.DataFrame,
                     original_matrix: pd.DataFrame, items: pd.DataFrame, n: int = 10):
    user_predictions = reconstructed_df.loc[user_id]
    already_rated = original_matrix.loc[user_id]
    unrated_items = user_predictions[already_rated == 0]

    top_items = unrated_items.sort_values(ascending=False).head(n)
    result = items[items["item_id"].isin(top_items.index)].copy()
    result["predicted_score"] = result["item_id"].map(top_items)
    return result.sort_values("predicted_score", ascending=False)

def recommend_for_new_user(preferred_item_ids: list, ratings: pd.DataFrame,
                            items: pd.DataFrame, n: int = 10):
    similar_raters = ratings[ratings["item_id"].isin(preferred_item_ids)]["user_id"].unique()

    candidate_ratings = ratings[ratings["user_id"].isin(similar_raters)]
    popular_items = candidate_ratings.groupby("item_id").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()

    popular_items = popular_items[~popular_items["item_id"].isin(preferred_item_ids)]
    popular_items = popular_items[popular_items["num_ratings"] >= 5]
    top = popular_items.sort_values(["avg_rating", "num_ratings"], ascending=False).head(n)

    result = items[items["item_id"].isin(top["item_id"])]
    return result
def train_svd_model_centered(user_item_matrix: pd.DataFrame, n_components: int = 20):
    mask = user_item_matrix != 0
    user_means = user_item_matrix.replace(0, np.nan).mean(axis=1)
    user_means = user_means.fillna(0)

    centered_matrix = user_item_matrix.sub(user_means, axis=0)
    centered_matrix = centered_matrix * mask  # keep zeros where data was actually missing

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(centered_matrix)
    item_factors = svd.components_

    reconstructed = np.dot(user_factors, item_factors)
    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    reconstructed_df = reconstructed_df.add(user_means, axis=0)

    return reconstructed_df, svd, user_means