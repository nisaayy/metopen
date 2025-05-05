# === IMPORT MODULE ===
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# === DATASET (langsung di dalam kode) ===
ratings_data = {
    'userId': [1, 1, 2, 2, 3],
    'itemId': [101, 102, 101, 103, 102],
    'rating': [4, 5, 3, 4, 2]
}
items_data = {
    'itemId': [101, 102, 103],
    'title': ['Movie A', 'Movie B', 'Movie C'],
    'genre': ['Action', 'Comedy', 'Action']
}

ratings = pd.DataFrame(ratings_data)
items = pd.DataFrame(items_data)

# === CONTENT-BASED FILTERING ===
def content_based(user_id):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(items['genre'])

    item_sim = cosine_similarity(tfidf_matrix)

    user_rated = ratings[ratings['userId'] == user_id]
    user_scores = {}

    for _, row in user_rated.iterrows():
        item_index = items[items['itemId'] == row['itemId']].index[0]
        sim_scores = list(enumerate(item_sim[item_index]))
        for idx, score in sim_scores:
            if items.loc[idx, 'itemId'] not in user_rated['itemId'].values:
                user_scores[items.loc[idx, 'itemId']] = user_scores.get(items.loc[idx, 'itemId'], 0) + score * row['rating']

    ranked = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in ranked]

# === COLLABORATIVE FILTERING (tanpa surprise) ===
def collaborative_filtering(user_id):
    # Buat matrix user-item
    user_item_matrix = ratings.pivot_table(index='userId', columns='itemId', values='rating').fillna(0)

    # Hitung kemiripan antar user
    similarity = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Ambil user yang mirip
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:]

    # Prediksi rating item yang belum dirating
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    pred_ratings = {}

    for item in unrated_items:
        weighted_sum = 0
        sim_sum = 0
        for other_user, sim_score in similar_users.items():
            other_rating = user_item_matrix.loc[other_user, item]
            if other_rating > 0:
                weighted_sum += sim_score * other_rating
                sim_sum += sim_score
        if sim_sum > 0:
            pred_ratings[item] = weighted_sum / sim_sum

    ranked = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in ranked]

# === HYBRID FILTERING ===
def hybrid_recommendation(user_id, alpha=0.5):
    content = content_based(user_id)
    collab = collaborative_filtering(user_id)

    scores = {}
    for rank, item_id in enumerate(content):
        scores[item_id] = scores.get(item_id, 0) + (1 - alpha) * (len(content) - rank)
    for rank, item_id in enumerate(collab):
        scores[item_id] = scores.get(item_id, 0) + alpha * (len(collab) - rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in ranked]

# === TEST OUTPUT ===
user_test = 1
print("📌 Content-Based Recommendation for User", user_test, ":", content_based(user_test))
print("📌 Collaborative Filtering for User", user_test, ":", collaborative_filtering(user_test))
print("📌 Hybrid Recommendation for User", user_test, ":", hybrid_recommendation(user_test))
