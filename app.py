import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load your cleaned data
df = pd.read_csv("ViewingActivity.csv")

# Prepare titles
titles = df[['Title']].drop_duplicates().copy()
titles['Title'] = titles['Title'].astype(str).str.strip().str.lower()

# TF-IDF on Titles (or use Genre if available)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(titles['Title'])

# Cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(title, top_n=5):
    title = title.strip().lower()
    if title not in titles['Title'].values:
        matches = get_close_matches(title, titles['Title'].tolist(), n=1, cutoff=0.6)
        if matches:
            title = matches[0]
        else:
            return f"'{title}' not found in your watch history.", []

    idx = titles[titles['Title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_titles = titles.iloc[[i[0] for i in sim_scores]]['Title'].str.title().tolist()
    return title.title(), recommended_titles

# Streamlit UI
st.title("🎬 Netflix Viewing Recommendation System")

user_input = st.text_input("Enter a movie you've watched:")

if user_input:
    matched_title, recommendations = recommend(user_input)
    
    if isinstance(recommendations, list):
        st.success(f"Because you watched **{matched_title}**, you might also like:")
        for rec in recommendations:
            st.write(f"✅ {rec}")
    else:
        st.error(recommendations)
