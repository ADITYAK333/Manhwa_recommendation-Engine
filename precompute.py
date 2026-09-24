import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
df = pd.read_csv('manhwa_industry_evolution_2026.csv')

df['genres'] = df['genres'].fillna('')
df['description'] = df['description'].fillna('')
df['soup'] = df['genres'] + " " + df['description']

print(" Generating BERT Embeddings ")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['soup'].tolist(), show_progress_bar=True)

print(" Computing Cosine Similarity Matrix...")
cosine_sim = cosine_similarity(embeddings, embeddings)

print(" Saving similarity matrix to 'cosine_sim.npy'...")
np.save('cosine_sim.npy', cosine_sim)

print("Precompute completed")