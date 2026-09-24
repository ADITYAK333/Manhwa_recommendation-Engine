#!/usr/bin/env python
# coding: utf-8

# 
# Manhwa Hybrid Recommendation Engine
# ------------------------------------
# Author: Adityak333
# 
# ---
# A fast Streamlit web app that recommends webtoons/manhwa by combining 
# Sentence-BERT plot embeddings with user-adjustable weights for rating and popularity.
# 
# Workflow:
# 1. Candidate Retrieval: Finds the top 50 semantic plot/genre matches using Sentence-BERT embeddings.
# 2. Live Re-Ranking: Re-ranks those candidates in real time based on sidebar slider weights 
#    (Plot Similarity, Community Rating, Popularity) and publication status filters.
# 
# Key Challenges & Fixes:
# - Slow Page Loads (BERT Latency):
#   Computing transformer embeddings on every load took minutes on CPU. 
#   -> Fixed by running embedding calculations offline (`precompute.py`) and saving the similarity 
#      matrix as a `cosine_sim.npy` binary. The app loads this instantly in <100ms.
# 
# - Unrelated Recommendations (TF-IDF Limits):
#   Keyword matching kept recommending romance titles for action queries due to generic word overlaps.
#   -> Fixed by upgrading to `all-MiniLM-L6-v2` embeddings, which match underlying story themes and synonyms.
# 
# - Unresponsive Sliders (Unscaled Math):
#   Raw ratings (0-100) completely overpowered similarity scores (0-1).
#   -> Fixed by scaling popularity and ratings to a uniform [0.0, 1.0] range using MinMaxScaler.
# 
# 
# End Result:
# A clean, fast UI that gives accurate plot recommendations, updates instantly when tweaking weights, 
# and loads without any live model compute delays.
# 

# In[ ]:


##Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


# In[ ]:


##data cleaning and preprocessing
@st.cache_data
def prepare_data(file): 
    df = pd.read_csv(file)
    df['chapters']=df['chapters'].fillna(0) 
    df['genres']=df['genres'].fillna('') 
    df['popularity']=df['popularity'].fillna(0) 
    df['score']=df['score'].fillna(0) 
    df['mean_score']=df['mean_score'].fillna(0)
    df['description']=df['description'].fillna('')
    df['soup']=df['genres']+ " "+df['description']
    scaler = MinMaxScaler()
    df[['popularity', 'score']] = scaler.fit_transform(df[['popularity', 'score']])
    return df


# In[ ]:


#load similarity matrix 
@st.cache_data
def load_similarity_matrix():
    return np.load('cosine_sim.npy')


# In[ ]:


#Recommendation system using weights and status filters 
def recommender(title, df, cosine_sim, w_sim, w_score, w_pop, status_filter):
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        return None

    target_idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[target_idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:100]     #getting top 100 matches

    candidate_keys = [pair[0] for pair in sorted_scores]
    sim_values = [pair[1] for pair in sorted_scores]
    candidate_df = df.iloc[candidate_keys].copy()
    candidate_df['sim_score'] = sim_values
    if status_filter != "All":      # Apply status filter 
        candidate_df = candidate_df[candidate_df['status'] == status_filter]
    candidate_df['final_score'] = (         #recommendation weights
        (w_sim * candidate_df['sim_score']) + 
        (w_score * candidate_df['score']) + 
        (w_pop * candidate_df['popularity'])
    )

    return candidate_df.sort_values(by='final_score', ascending=False).head(20)     #recommendation results


# In[ ]:


#Main UI 

st.set_page_config(page_title="Manhwa", layout="centered")
st.title("Manhwa Recommendation")#title
st.sidebar.header("Algorithm Controls")# Sidebar Controls
w_sim = st.sidebar.slider("Plot & Genre Similarity Weight", 0.0, 1.0, 0.5, 0.05)
w_score = st.sidebar.slider("Rating Weight", 0.0, 1.0, 0.3, 0.05)
w_pop = st.sidebar.slider("Popularity Weight", 0.0, 1.0, 0.2, 0.05)
st.sidebar.divider()
status_filter = st.sidebar.selectbox("Filter by Status", ["All", "FINISHED", "RELEASING"])

try :
    df = prepare_data('manhwa_industry_evolution_2026.csv')
    cosine_sim = load_similarity_matrix()
    if 'random_manhwa' not in st.session_state:
            top_rated_df = df[df['mean_score'] >= 75]
            st.session_state.random_manhwa = top_rated_df.sample(1).iloc[0]
    titles_list = sorted(df['title'].astype(str).unique().tolist())
    selected_title = st.selectbox("Select or search for a Manhwa title:",           # Selectbox for titles
        options=titles_list,
        index=None,
        placeholder="Solo Leveling ..."
    )
    if selected_title:
        results = recommender(selected_title, df, cosine_sim, w_sim, w_score, w_pop, status_filter)
        target_info = df[df['title'] == selected_title].iloc[0]
        st.info(f"**Selected:** {target_info['title']} | **Genres:** {target_info['genres']} | **Rating:** {target_info['mean_score']}/100")
        if results is not None and not results.empty: 
            st.subheader(f"Recommended titles for '{selected_title}':")
            for idx, row in results.iterrows():         #display the results
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"### {row['title']}")
                        st.caption(f"**Genres:** {row['genres']} | **Status:** {row['status']}")
                    with col2:
                        st.metric("Plot Match", f"{round(row['sim_score'] * 100, 1)}%")
                    with col3:
                        st.metric("Score", f"{int(row['mean_score'])}/100")

                    with st.expander("Read Synopsis"):
                        st.write(row['description'])
                    st.divider()
        else:
            st.error("No results found.")
    else:                                   #Provides a random manhwa title if user is unsure of the title 
            st.markdown("---")
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.subheader(" Featured Pick")
            with col_b:
                if st.button("Re-Roll"):
                    top_rated_df = df[df['mean_score'] >= 75]
                    st.session_state.random_manhwa = top_rated_df.sample(1).iloc[0]
                    st.rerun()
            featured = st.session_state.random_manhwa
            with st.container():
                st.markdown(f"## **{featured['title']}**")
                st.caption(f"**Genres:** {featured['genres']} | **Status:** {featured['status']}")
                st.metric("Community Rating", f"{int(featured['mean_score'])}/100")
                st.write(f"**Synopsis:** {featured['description']}")
                st.divider()
except FileNotFoundError:
    st.error("File not found. Please download the file and try again.")


# Future Enhancements : 
# 
# Vector Database Integration: Swap out cosine_sim.npy for FAISS or ChromaDB if scaling past 10,000 titles to keep memory usage low.
# 
# Strict Genre Filtering: Add multi-select genre tags so users can exclude entire genres (e.g., exclude Horror or Romance) before hybrid re-ranking.
# 
# Author/Artist Boosting: Give extra weight to titles created by the same author or artist.

# Author : ADITYK333
# AI usage : AI tools were used for minor formatting, editing, and troubleshooting in this project.
