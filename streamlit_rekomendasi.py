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
    'title': ['tung tung tung sahoor', 'balerina cappucina', 'bombardilo crocodilo'],
    'genre': ['comedy', 'romance', 'Action']
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
    user_item_matrix = ratings.pivot_table(index='userId', columns='itemId', values='rating').fillna(0)
    similarity = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:]
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

# === FUNCTION TO GET USER RATING INPUT ===
def get_user_ratings():
    print("\nPlease rate the recommended movies (from 1 to 5):")
    user_ratings = {}
    for idx, item_id in enumerate(recommended_items, 1):
        title = items[items['itemId'] == item_id]['title'].values[0]
        while True:
            try:
                rating = int(input(f"Rate Movie {title} (1-5): "))
                if 1 <= rating <= 5:
                    user_ratings[item_id] = rating
                    break
                else:
                    print("Rating should be between 1 and 5. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5.")
    return user_ratings

# === TEST OUTPUT ===
user_test = 1
recommended_items = hybrid_recommendation(user_test)

# Menampilkan hasil dengan gaya yang lebih menarik
print("\n🎬 Recommender System Output for User", user_test)

# Hybrid Recommendation
print("\n📌 Hybrid Recommendation:")
for idx, item_id in enumerate(recommended_items, 1):
    title = items[items['itemId'] == item_id]['title'].values[0]
    genre = items[items['itemId'] == item_id]['genre'].values[0]
    print(f"{idx}. {title} - Genre: {genre}")

# Mengambil input rating dari pengguna
user_ratings = get_user_ratings()

# Menampilkan rating yang diberikan oleh pengguna
print("\n🎯 Your ratings for the recommended movies:")
for item_id, rating in user_ratings.items():
    title = items[items['itemId'] == item_id]['title'].values[0]
    print(f"Movie: {title}, Your Rating: {rating}")
