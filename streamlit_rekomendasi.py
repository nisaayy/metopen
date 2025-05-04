
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# Content-Based Filtering
@st.cache_data
def compute_content_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_content_similarity()

def recommend_content(title, top_n=5):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Collaborative Filtering
@st.cache_resource
def train_cf():
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)
    return algo

algo = train_cf()

def recommend_cf(user_id, top_n=5):
    movie_ids = movies['movieId'].unique()
    rated = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    predictions = []
    for movie_id in movie_ids:
        if movie_id not in rated:
            pred = algo.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [x[0] for x in predictions[:top_n]]
    return movies[movies['movieId'].isin(top_movie_ids)]['title']

# Streamlit UI
st.title("Sistem Rekomendasi: Content-Based vs Collaborative Filtering")

menu = st.sidebar.selectbox("Pilih Metode Rekomendasi", ["Content-Based", "Collaborative Filtering"])

if menu == "Content-Based":
    movie_title = st.selectbox("Pilih Judul Film", movies['title'].sort_values())
    if st.button("Rekomendasikan"):
        result = recommend_content(movie_title)
        st.write("Rekomendasi Film Berdasarkan Genre:")
        st.write(result)

elif menu == "Collaborative Filtering":
    user_id = st.number_input("Masukkan User ID", min_value=1, step=1)
    if st.button("Rekomendasikan"):
        result = recommend_cf(user_id)
        st.write("Rekomendasi Film Berdasarkan Preferensi Pengguna:")
        st.write(result)
