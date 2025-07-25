from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load song dataset
csv_path = os.path.join(os.path.dirname(__file__), '../dataset/clustered_df.csv')
df = pd.read_csv(csv_path)

# Normalize features
features = ['acousticness', 'danceability', 'energy', 'valence', 'tempo']
df_features = (df[features] - df[features].mean()) / df[features].std()
similarity = cosine_similarity(df_features)
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(song_name, top_n=5):
    if song_name not in indices:
        return []
    idx = indices[song_name]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    return df.iloc[song_indices][['name', 'artists', 'year']].to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        song_name = request.form['song_name']
        recommendations = get_recommendations(song_name)

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
