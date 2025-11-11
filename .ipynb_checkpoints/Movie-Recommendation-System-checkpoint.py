import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import hnswlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import plotly.express as px
from metrics_implementation import *

# ===============================
# Page setup & styles
# ===============================
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
    .methodology-box {
        background-color: #f0f2f6;
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .math-formula {
        background-color: #fff;
        border-left: 4px solid #4ECDC4;
        padding: 12px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Session state
# ===============================
if "driver" not in st.session_state:
    st.session_state.driver = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"
if "ann_built" not in st.session_state:
    st.session_state.ann_built = False
if "content_index" not in st.session_state:
    st.session_state.content_index = None
if "cf_index" not in st.session_state:
    st.session_state.cf_index = None
if "movie_df" not in st.session_state:
    st.session_state.movie_df = None
if "ratings_df" not in st.session_state:
    st.session_state.ratings_df = None
if "genre_vocab" not in st.session_state:
    st.session_state.genre_vocab = []
if "movie_vectors_content" not in st.session_state:
    st.session_state.movie_vectors_content = None
if "movie_vectors_cf" not in st.session_state:
    st.session_state.movie_vectors_cf = None
if "movie_to_row" not in st.session_state:
    st.session_state.movie_to_row = {}
if "row_to_movie" not in st.session_state:
    st.session_state.row_to_movie = {}

# ===============================
# Neo4j helpers
# ===============================
def connect_to_neo4j(uri: str, user: str, password: str):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as s:
            s.run("RETURN 1")
        return driver, None
    except Exception as e:
        return None, str(e)

def disconnect_neo4j():
    if st.session_state.driver:
        st.session_state.driver.close()
    st.session_state.driver = None
    st.session_state.connected = False

# Pull minimal frames from Neo4j
def fetch_movies(driver) -> pd.DataFrame:
    with driver.session() as s:
        # Try Genre nodes, but always keep string genres if available
        res = s.run("""
            MATCH (m:Movie)
            RETURN m.movieId AS movieId, m.title AS title, m.genres AS genres
            ORDER BY toInteger(m.movieId)
        """)
        rows = [dict(r) for r in res]
    df = pd.DataFrame(rows)
    # normalize types
    if not df.empty and df["movieId"].dtype != int:
        df["movieId"] = df["movieId"].astype(int)
    return df

def fetch_users(driver) -> List[int]:
    with driver.session() as s:
        res = s.run("""
            MATCH (u:User) RETURN u.userId AS userId ORDER BY toInteger(u.userId)
        """)
        ids = [int(r["userId"]) for r in res]
    return ids

def fetch_ratings(driver) -> pd.DataFrame:
    with driver.session() as s:
        res = s.run("""
            MATCH (u:User)-[r:RATED]->(m:Movie)
            RETURN toInteger(u.userId) AS userId, toInteger(m.movieId) AS movieId, toFloat(r.rating) AS rating
        """)
        rows = [dict(r) for r in res]
    return pd.DataFrame(rows)

#=====================================
# Check if data exist
#=======================================
def get_database_stats(driver) -> Dict[str, int]:
    with driver.session() as s:
        users = s.run("MATCH (u:User) RETURN count(u) AS c").single()["c"]
        movies = s.run("MATCH (m:Movie) RETURN count(m) AS c").single()["c"]
        ratings = s.run("MATCH (:User)-[r:RATED]->(:Movie) RETURN count(r) AS c").single()["c"]
        # distinct genres parsed from m.genres string
        res = s.run("""
            MATCH (m:Movie) WHERE m.genres IS NOT NULL
            WITH m, [g IN split(m.genres,'|') WHERE trim(g)<>'' AND g<>'(no genres listed)'] AS gs
            UNWIND gs AS g
            RETURN count(DISTINCT g) AS c
        """).single()
        genres = res["c"] if res else 0
    return {"users": users, "movies": movies, "ratings": ratings, "genres": genres}

#=====================================
#DESCRIPTIVE STATISTIC
#=======================================
def get_descriptive_stats(driver):
    """Get comprehensive descriptive statistics from the database"""
    with driver.session() as session:
        stats = {}
        
        # Basic counts
        result = session.run("MATCH (u:User) RETURN count(u) AS count")
        stats['total_users'] = result.single()["count"]
        
        result = session.run("MATCH (m:Movie) RETURN count(m) AS count")
        stats['total_movies'] = result.single()["count"]
        
        result = session.run("MATCH ()-[r:RATED]->() RETURN count(r) AS count")
        stats['total_ratings'] = result.single()["count"]
        
        try:
            # result = session.run("MATCH (g:Genre) RETURN count(g) AS count")
            # stats['total_genres'] = result.single()["count"]
            # Genres (distinct genres across all nodes and relationships)
            # genres_query = """
            # CALL {
            #   MATCH (n)
            #   WHERE n.genres IS NOT NULL
            #   UNWIND n.genres AS genre
            #   RETURN genre
            #   UNION
            #   MATCH ()-[r]-()
            #   WHERE r.genres IS NOT NULL
            #   UNWIND r.genres AS genre
            #   RETURN genre
            # }
            # RETURN count(DISTINCT genre) AS c
            # """
            # genres = session.run(genres_query).single()["c"]
            # Choose DISTINCT or not, depending on what you want
            genres_query = """
            MATCH (m:Movie)
            WHERE m.genres IS NOT NULL
            UNWIND split(m.genres, '|') AS g
            WITH trim(g) AS genre
            WHERE genre <> '' AND genre <> '(no genres listed)'
            RETURN count(DISTINCT genre) AS c
            """
            genres = session.run(genres_query).single()["c"]
            stats['total_genres'] = genres
    
        except:
            stats['total_genres'] = 0
        
        # Rating statistics
        result = session.run("""
            MATCH ()-[r:RATED]->()
            RETURN avg(r.rating) AS avg_rating,
                   min(r.rating) AS min_rating,
                   max(r.rating) AS max_rating,
                   stdev(r.rating) AS std_rating
        """)
        record = result.single()
        stats['avg_rating'] = record['avg_rating']
        stats['min_rating'] = record['min_rating']
        stats['max_rating'] = record['max_rating']
        stats['std_rating'] = record['std_rating']
        
        # User activity statistics
        result = session.run("""
            MATCH (u:User)-[r:RATED]->()
            WITH u, count(r) AS num_ratings
            RETURN avg(num_ratings) AS avg_ratings_per_user,
                   min(num_ratings) AS min_ratings,
                   max(num_ratings) AS max_ratings
        """)
        record = result.single()
        stats['avg_ratings_per_user'] = record['avg_ratings_per_user']
        stats['min_ratings_per_user'] = record['min_ratings']
        stats['max_ratings_per_user'] = record['max_ratings']
        
        # Movie popularity statistics
        result = session.run("""
            MATCH ()-[r:RATED]->(m:Movie)
            WITH m, count(r) AS num_ratings
            RETURN avg(num_ratings) AS avg_ratings_per_movie,
                   min(num_ratings) AS min_ratings,
                   max(num_ratings) AS max_ratings
        """)
        record = result.single()
        stats['avg_ratings_per_movie'] = record['avg_ratings_per_movie']
        stats['min_ratings_per_movie'] = record['min_ratings']
        stats['max_ratings_per_movie'] = record['max_ratings']
        
        return stats
#================================================   
def get_all_users(driver):
    """Retrieve all users"""
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User)
            RETURN u.userId AS userId
            ORDER BY u.userId
        """)
        return [record["userId"] for record in result]

def get_user_ratings(driver, user_id, limit=10):
    """Retrieves films rated by a user"""
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
            RETURN m.title AS title, 
                   m.genres AS genres,
                   r.rating AS rating,
                   r.timestamp AS timestamp
            ORDER BY r.rating DESC
            LIMIT $limit
        """, userId=user_id, limit=limit)
        
        ratings = []
        for record in result:
            genres = []
            if record['genres']:
                genres = [g.strip() for g in record['genres'].split('|') if g.strip() and g.strip() != '(no genres listed)']
            
            ratings.append({
                'title': record['title'],
                'genres': ', '.join(genres) if genres else 'N/A',
                'rating': record['rating']
            })
        return ratings
#================================================   

        
def get_rating_distribution(driver):
    """Get rating distribution for visualization"""
    with driver.session() as session:
        result = session.run("""
            MATCH ()-[r:RATED]->()
            RETURN r.rating AS rating, count(*) AS count
            ORDER BY rating
        """)
        return pd.DataFrame([dict(record) for record in result])

def get_top_users(driver, limit=20):
    """Get most active users"""
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User)-[r:RATED]->()
            WITH u, count(r) AS num_ratings, avg(r.rating) AS avg_rating
            RETURN u.userId AS userId,
                   num_ratings,
                   round(avg_rating, 2) AS avg_rating
            ORDER BY num_ratings DESC
            LIMIT $limit
        """, limit=limit)
        return pd.DataFrame([dict(record) for record in result])

def get_genre_distribution(driver):
    """Get genre distribution"""
    with driver.session() as session:
        # Try with Genre nodes first
        result = session.run("""
            MATCH (g:Genre)<-[:HAS_GENRE]-(m:Movie)
            RETURN g.name AS genre, count(m) AS count
            ORDER BY count DESC
        """)
        df = pd.DataFrame([dict(record) for record in result])
        
        if df.empty:
            # Fallback: parse from genres string
            result = session.run("""
                MATCH (m:Movie)
                WHERE m.genres IS NOT NULL
                RETURN m.genres AS genres
            """)
            all_genres = []
            for record in result:
                genres = record['genres'].split('|')
                all_genres.extend([g.strip() for g in genres if g.strip() and g.strip() != '(no genres listed)'])
            
            genre_counts = pd.Series(all_genres).value_counts()
            df = pd.DataFrame({'genre': genre_counts.index, 'count': genre_counts.values})
        
        return df

        
def get_genre_ratings(driver):
    """Get average ratings by genre"""
    with driver.session() as session:
        result = session.run("""
            MATCH (g:Genre)<-[:HAS_GENRE]-(m:Movie)<-[r:RATED]-()
            WITH g.name AS genre, avg(r.rating) AS avg_rating, count(r) AS num_ratings
            WHERE num_ratings >= 100
            RETURN genre, 
                   round(avg_rating, 2) AS avg_rating,
                   num_ratings
            ORDER BY avg_rating DESC
        """)
        df = pd.DataFrame([dict(record) for record in result])
        
        if df.empty:
            # Fallback: parse from genres string
            result = session.run("""
                MATCH (m:Movie)<-[r:RATED]-()
                WHERE m.genres IS NOT NULL
                RETURN m.genres AS genres, r.rating AS rating
            """)
            
            genre_ratings = {}
            for record in result:
                genres = [g.strip() for g in record['genres'].split('|') if g.strip() and g.strip() != '(no genres listed)']
                rating = record['rating']
                for genre in genres:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    genre_ratings[genre].append(rating)
            
            data = []
            for genre, ratings in genre_ratings.items():
                if len(ratings) >= 100:
                    data.append({
                        'genre': genre,
                        'avg_rating': round(np.mean(ratings), 2),
                        'num_ratings': len(ratings)
                    })
            
            df = pd.DataFrame(data).sort_values('avg_rating', ascending=False)
        
        return df
        
def get_ratings_over_time(driver):
    """Get ratings over time"""
    with driver.session() as session:
        # Get timestamps and convert in Python to avoid Neo4j date errors
        result = session.run("""
            MATCH ()-[r:RATED]->()
            WHERE r.timestamp IS NOT NULL
            RETURN r.timestamp AS timestamp
        """)
        timestamps = [record['timestamp'] for record in result]
        
        if not timestamps:
            return pd.DataFrame(columns=['rating_date', 'count'])
        
        # Convert timestamps to dates in Python
        from datetime import datetime
        dates = [datetime.fromtimestamp(int(ts)).date() for ts in timestamps]
        
        # Count by date
        date_counts = {}
        for date in dates:
            date_counts[date] = date_counts.get(date, 0) + 1
        
        df = pd.DataFrame([
            {'rating_date': date, 'count': count}
            for date, count in sorted(date_counts.items())
        ])
        df['rating_date'] = pd.to_datetime(df['rating_date'])
        return df


#
def get_top_movies(driver, limit=20):
    """Get top rated movies"""
    with driver.session() as session:
        result = session.run("""
            MATCH (m:Movie)<-[r:RATED]-()
            WITH m, count(r) AS num_ratings, avg(r.rating) AS avg_rating
            WHERE num_ratings >= 50
            RETURN m.title AS title,
                   num_ratings,
                   round(avg_rating, 2) AS avg_rating
            ORDER BY avg_rating DESC, num_ratings DESC
            LIMIT $limit
        """, limit=limit)
        return pd.DataFrame([dict(record) for record in result])
        
def check_data_exists(driver):
    """Check if database already has data"""
    with driver.session() as session:
        result = session.run("MATCH (m:Movie) RETURN count(m) AS count")
        movie_count = result.single()["count"]
        
        result = session.run("MATCH (u:User) RETURN count(u) AS count")
        user_count = result.single()["count"]
        
        result = session.run("MATCH ()-[r:RATED]->() RETURN count(r) AS count")
        rating_count = result.single()["count"]
        
        return movie_count > 0 or user_count > 0 or rating_count > 0
# ===============================
# Vectorization
# ===============================
def build_genre_vocab(movie_df: pd.DataFrame) -> List[str]:
    vocab = []
    seen = set()
    for gstr in movie_df["genres"].fillna(""):
        for g in [x.strip() for x in gstr.split("|") if x and x.strip() and x.strip() != "(no genres listed)"]:
            if g not in seen:
                seen.add(g)
                vocab.append(g)
    vocab.sort()
    return vocab

def encode_movies_content(movie_df: pd.DataFrame, genre_vocab: List[str]) -> Tuple[np.ndarray, Dict[int,int], Dict[int,int]]:
    dim = len(genre_vocab)
    movie_to_row = {}
    row_to_movie = {}
    vectors = np.zeros((len(movie_df), dim), dtype=np.float32)
    g2i = {g:i for i,g in enumerate(genre_vocab)}
    for r, row in movie_df.reset_index(drop=True).iterrows():
        mid = int(row["movieId"])
        movie_to_row[mid] = r
        row_to_movie[r] = mid
        if isinstance(row["genres"], str):
            for g in row["genres"].split("|"):
                g = g.strip()
                if g and g != "(no genres listed)" and g in g2i:
                    vectors[r, g2i[g]] = 1.0
    # L2 normalize for cosine ANN
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors = vectors / norms
    return vectors, movie_to_row, row_to_movie

def encode_movies_cf(movie_df: pd.DataFrame, ratings_df: pd.DataFrame) -> np.ndarray:
    # Item-based CF: vector per movie over users (sparse -> dense via mean-centering per movie)
    if ratings_df.empty:
        return np.zeros((len(movie_df), 1), dtype=np.float32)
    # Build user id mapping compact
    unique_users = np.array(sorted(ratings_df["userId"].unique()))
    u2i = {u:i for i,u in enumerate(unique_users)}
    # mean rating per movie for centering
    mu = ratings_df.groupby("movieId")["rating"].mean().to_dict()

    num_users = len(unique_users)
    vectors = np.zeros((len(movie_df), num_users), dtype=np.float32)

    # For memory, we'll fill only available entries (dense may be big, but MovieLens 100k is OK)
    for _, row in ratings_df.iterrows():
        mid = int(row["movieId"])
        if mid not in st.session_state.movie_to_row:  # after content encoding we have mapping
            continue
        r = st.session_state.movie_to_row[mid]
        ui = u2i[int(row["userId"])]
        vectors[r, ui] = float(row["rating"]) - float(mu.get(mid, 0.0))

    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors = vectors / norms
    return vectors

# ===============================
# ANN index builders
# ===============================
def build_hnsw_index(vectors: np.ndarray, space: str = "cosine", M: int = 32, ef: int = 200) -> hnswlib.Index:
    if vectors.size == 0:
        return None
    dim = vectors.shape[1]
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=vectors.shape[0], ef_construction=ef, M=M)
    index.add_items(vectors, ids=np.arange(vectors.shape[0], dtype=np.int32))
    index.set_ef(ef)
    return index

# ===============================
# Recommendation pipelines
# ===============================


def recommend_content_ann(selected_title: str,  k: int)-> pd.DataFrame:
    """
    Content-based recommendations with optional diversity filtering
    """
    df = st.session_state.movie_df
    row = df.index[df["title"] == selected_title]
    if len(row) == 0:
        return pd.DataFrame()
    
    ridx = int(row[0])
    if st.session_state.content_index is None:
        return pd.DataFrame()
    
    # Get more candidates to ensure we have k results after filtering
    labels, distances = st.session_state.content_index.knn_query(
        st.session_state.movie_vectors_content[ridx:ridx+1], 
        k=min(k*3, 100)  # Get 3x more candidates
    )
    labels = labels[0].tolist()
    distances = distances[0].tolist()
    
    # First pass: Try to get diverse results (skip near-perfect matches)
    out = []
    for lbl, dist in zip(labels, distances):
        if lbl == ridx:  # Skip the anchor movie itself
            continue
        
        score = 1.0 - float(dist)

        threshold = 0.99
        # Skip movies that are TOO similar (likely identical genre combinations)
        if score >= threshold:  # Threshold for "too similar"
            continue
        
        mid = st.session_state.row_to_movie[lbl]
        title = df.iloc[lbl]["title"]
        genres = df.iloc[lbl]["genres"]
        out.append({
            "title": title, 
            "genres": genres, 
            "cosine_similarity": round(score, 4)
        })
        
        if len(out) >= k:
            break
    
    # Second pass: If we don't have enough diverse results, accept all results
    if len(out) == 0:
        for lbl, dist in zip(labels, distances):
            if lbl == ridx:
                continue
            
            mid = st.session_state.row_to_movie[lbl]
            title = df.iloc[lbl]["title"]
            genres = df.iloc[lbl]["genres"]
            score = 1.0 - float(dist)
            
            # Check if we already have this movie
            if title not in [item["title"] for item in out]:
                out.append({
                    "title": title,
                    "genres": genres,
                    "cosine_similarity": round(score, 4)
                })
            
            if len(out) >= k:
                break
    
    return pd.DataFrame(out)
    


def build_user_profile_vector_cf(user_id: int) -> Optional[np.ndarray]:
    # user vector = weighted average of liked movies' item vectors (ratings >= 4)
    df = st.session_state.ratings_df
    liked = df[(df["userId"] == user_id) & (df["rating"] >= 4.0)]
    if liked.empty:
        return None
    vecs = []
    weights = []
    for _, row in liked.iterrows():
        mid = int(row["movieId"])
        if mid not in st.session_state.movie_to_row:
            continue
        ridx = st.session_state.movie_to_row[mid]
        vecs.append(st.session_state.movie_vectors_cf[ridx])
        weights.append(float(row["rating"]))
    if not vecs:
        return None
    vecs = np.vstack(vecs)
    w = np.array(weights, dtype=np.float32).reshape(-1,1)
    profile = (vecs * w).sum(axis=0, keepdims=True)
    # normalize
    profile = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-8)
    return profile.astype(np.float32)

def recommend_cf_ann(user_id: int, k: int) -> pd.DataFrame:
    profile = build_user_profile_vector_cf(user_id)
    if profile is None or st.session_state.cf_index is None:
        return pd.DataFrame()
    labels, distances = st.session_state.cf_index.knn_query(profile, k=k*5)
    labels = labels[0].tolist()
    distances = distances[0].tolist()
    seen = set(st.session_state.ratings_df.loc[st.session_state.ratings_df["userId"]==user_id, "movieId"].astype(int).tolist())
    out = []
    for lbl, dist in zip(labels, distances):
        mid = st.session_state.row_to_movie[lbl]
        if mid in seen:
            continue
        title = st.session_state.movie_df.iloc[lbl]["title"]
        genres = st.session_state.movie_df.iloc[lbl]["genres"]
        score = 1.0 - float(dist)  # cosine similarity
        out.append({"title": title, "genres": genres, "cf_similarity": round(score, 4)})
        if len(out) >= k:
            break
    return pd.DataFrame(out)

def recommend_hybrid_union(user_id, anchor_title, k, alpha, beta):
    content_df = recommend_content_ann(anchor_title, k * 10)
    cf_df = recommend_cf_ann(user_id, k * 10)

    content_df["score_norm"] = (content_df["cosine_similarity"]
                                - content_df["cosine_similarity"].min()) / (
                                   content_df["cosine_similarity"].max()
                                   - content_df["cosine_similarity"].min())
    cf_df["score_norm"] = (cf_df["cf_similarity"]
                           - cf_df["cf_similarity"].min()) / (
                              cf_df["cf_similarity"].max()
                              - cf_df["cf_similarity"].min())

    content_df["hybrid_score"] = alpha * content_df["score_norm"]
    cf_df["hybrid_score"] = beta * cf_df["score_norm"]

    merged = pd.concat([content_df, cf_df], ignore_index=True)
    merged = merged.sort_values("hybrid_score", ascending=False)
    return merged.head(k)[["title", "genres", "hybrid_score"]]





def recommend_hybrid_ann(user_id: int, anchor_title: str, k: int,
                         alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
    """
    Mixed Hybrid Recommender (Weighted Proportion + Transfer Score)
    ---------------------------------------------------------------
    Each system contributes distinct items proportionally to α and β.
    The hybrid_score is a normalized transfer of each system's own ranking,
    used only for joint display and unified scaling (not a true fusion).
    """

    # 1️⃣ Get recommendations from both systems
    content_df = recommend_content_ann(anchor_title, k * 5)
    cf_df = recommend_cf_ann(user_id, k * 5)

    if content_df.empty and cf_df.empty:
        return pd.DataFrame()

    # 2️⃣ Normalize each recommender’s scores safely
    def normalize(series):
        if series is None or series.empty:
            return pd.Series(dtype=float)
        x = series.fillna(0).to_numpy(dtype=float)
        rng = (x.max() - x.min()) or 1e-6
        return pd.Series((x - x.min()) / rng, index=series.index)

    content_df["norm_score"] = normalize(content_df["cosine_similarity"])
    cf_df["norm_score"] = normalize(cf_df["cf_similarity"])

    # 3️⃣ Select proportional subsets from each list
    n_content = int(round(alpha * k))
    n_cf = k - n_content

    content_top = content_df.sort_values("norm_score", ascending=False).head(n_content)
    cf_top = cf_df.sort_values("norm_score", ascending=False).head(n_cf)

    content_top["source"] = "content"
    cf_top["source"] = "collaborative"

    # 4️⃣ Combine both sets
    final_df = pd.concat([content_top, cf_top], ignore_index=True)

    # 5️⃣ Transfer normalized scores into unified hybrid_score
    final_df["hybrid_score"] = np.where(
        final_df["source"] == "content",
        final_df["norm_score"],
        final_df["norm_score"]
    )

    # 6️⃣ Safety fallback in case of NaNs
    final_df["hybrid_score"].fillna(final_df["hybrid_score"].mean(), inplace=True)

    # 7️⃣ Sort final results for presentation
    final_df = final_df.sort_values("hybrid_score", ascending=False).head(k).reset_index(drop=True)

    # Return a consistent and clean structure
    return final_df[["title", "genres", "cosine_similarity", "cf_similarity",
                     "hybrid_score", "source"]]

# ===============================
# Sidebar navigation
# ===============================
def show_sidebar():
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        if st.button("🏠 About & Methodology", use_container_width=True,
                     type="primary" if st.session_state.current_page=="landing" else "secondary"):
            st.session_state.current_page = "landing"
            st.rerun()
        if not st.session_state.connected:
            if st.button("🔗 Connect to Database", use_container_width=True,
                         type="primary" if st.session_state.current_page=="connection" else "secondary"):
                st.session_state.current_page = "connection"
                st.rerun()
        if st.session_state.connected:
            if st.button("📊 Dashboard", use_container_width=True,
                         type="primary" if st.session_state.current_page=="dashboard" else "secondary"):
                st.session_state.current_page = "dashboard"
                st.rerun()

        st.markdown("---")
        if st.session_state.connected:
            try:
                stats = get_database_stats(st.session_state.driver)
                st.metric("👥 Users", f"{stats['users']:,}")
                st.metric("🎬 Movies", f"{stats['movies']:,}")
                st.metric("⭐ Ratings", f"{stats['ratings']:,}")
                st.metric("🎭 Genres", f"{stats['genres']:,}")
                st.success("✅ Connected")
                if st.button("🔌 Disconnect", use_container_width=True):
                    disconnect_neo4j()
                    st.session_state.current_page = "connection"
                    st.rerun()
            except Exception as e:
                st.error(f"Stats error: {e}")
        else:
            st.info("📡 Not connected")


def recommend_hybrid_ann(user_id: int, anchor_title: str, k: int,
                         alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
    """
    Mixed Hybrid Recommender (Proportional Blend, No Renormalization)
    -----------------------------------------------------------------
    - Takes α proportion of content-based results and β proportion of CF results.
    - Each system contributes distinct items (no forced overlap).
    - Scores are directly transferred from their native similarity scales (already 0–1).
    """

    # 1️Fetch both recommendation lists
    content_df = recommend_content_ann(anchor_title, k * 5)
    cf_df = recommend_cf_ann(user_id, k * 5)

    if content_df.empty and cf_df.empty:
        return pd.DataFrame()

    # 2 Determine proportional counts
    n_content = int(round(alpha * k))
    n_cf = k - n_content

    # 3Select top results
    content_top = content_df.sort_values("cosine_similarity", ascending=False).head(n_content)
    cf_top = cf_df.sort_values("cf_similarity", ascending=False).head(n_cf)

    content_top["source"] = "content"
    cf_top["source"] = "collaborative"

    # 4️ Combine both lists
    final_df = pd.concat([content_top, cf_top], ignore_index=True)

    # 5️⃣ Transfer their existing scores into a unified hybrid_score
    # (no re-normalization, just reuse original 0–1 range)
    final_df["hybrid_score"] = np.where(
        final_df["source"] == "content",
        final_df["cosine_similarity"],
        final_df["cf_similarity"]
    )

    # Fill any missing scores with the mean (for safety)
    final_df["hybrid_score"].fillna(final_df["hybrid_score"].mean(), inplace=True)

    # 6️⃣ Sort final ranked list for display
    final_df = final_df.sort_values("hybrid_score", ascending=False).head(k).reset_index(drop=True)

    # 7️⃣ Return unified structure
    return final_df[["title", "genres", "cosine_similarity", "cf_similarity", "hybrid_score", "source"]]

# ===============================
# Pages
# ===============================




# import streamlit as st

def show_landing():

    
  st.markdown("<h1 class='main-header'>🎬 ANN Movie Recommender</h1>", unsafe_allow_html=True)
  st.markdown("<p class='sub-header'>Fast Approximate Nearest Neighbors (HNSW) for content, collaborative, and hybrid recommendations on Neo4j</p>", unsafe_allow_html=True)
  st.markdown("---")

  # Overview
  st.header("1- Overview")

  st.markdown("""
  **Approximate Nearest Neighbor (ANN)** search is a method used to quickly find 
  items (movies, products, documents, etc.) that are *most similar* to a given reference item 
  based on their vector representations.

  Each movie in our database is represented as a **vector** - a list of numbers describing 
  its features such as genres, user ratings, or embeddings.  
  ANN algorithms allow us to find the *closest* movies in this high-dimensional space 
  **without scanning the entire dataset**, which would be too slow for large databases.

  ANN is at the heart of modern recommendation and search systems, used by:
  - **Meta (`FAISS`)** for Facebook and Instagram recommendations  
  - **Spotify (`Annoy` & `Voyager`)** for music discovery  
  - **Google** and **Pinterest** for image and content search  

  In this app, we use the **`HNSW` (Hierarchical Navigable Small World)** algorithm, 
  one of the fastest and most accurate ANN methods - implemented through Neo4j and HNSWlib.


    The system combines three complementary approaches:
  - **Content-Based (ANN):** recommends movies with similar **features or genres**.
  - **Collaborative (ANN):** recommends movies liked by **similar users**.
  - **Hybrid (Mixed):** proportionally merges both signals (α for content, β for collaborative).
  

  ---
  """)

  st.info("""
     💡**Next Step**

    To explore how this recommender works on your own dataset:
    - Go to the **Connection** tab in the sidebar.
    - Connect to your **Neo4j** database.
    - If you don’t have a database installed yet, see the project’s [GitHub page](https://github.com/fathnelle4/Movie-Recommendation-System/tree/main) for setup and installation details.
    """)


    
  # st.write("""
  # This app uses **Approximate Nearest Neighbors (ANN)** powered by **HNSW** to deliver fast, scalable, 
  # and interpretable movie recommendations from a **Neo4j** knowledge graph.

  #   **ANN-based search** is a core technique used by industry leaders like **Meta (via `FAISS`)**, **Spotify (via `Annoy` and `Voyager`)**, 
  # **Google**, and **Pinterest** to power **recommendation engines**, **feed ranking**, and **semantic search**.  
  # These frameworks enable large-scale vector search over millions of data points with near real-time responses.



  # The system combines three complementary approaches:
  # - **Content-Based (ANN):** recommends movies with similar **features or genres**.
  # - **Collaborative (ANN):** recommends movies liked by **similar users**.
  # - **Hybrid (Mixed):** proportionally merges both signals (α for content, β for collaborative).
  # """)
  # Custom CSS to enlarge tab labels and improve layout
  st.markdown("""
  <style>
  /* Increase the font size of tab labels */
  .stTabs [data-baseweb="tab"] {
      font-size: 10rem;
      font-weight: 600;
      color: #1e1e1e;
  }
  /* Optional: adjust padding for better spacing */
  .stTabs [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] {
      padding-top: 0.9rem;
      padding-bottom: 0.9rem;
  }
  /* Highlight active tab */
  .stTabs [aria-selected="true"] {
      color: #ff4b4b !important;
      border-bottom: 3px solid #ff4b4b !important;
  }
  </style>
  """, unsafe_allow_html=True)



    
  # Mathematical Foundations
  st.header("2- Mathematical Foundations (ANN)")
  c1, c2, c3 = st.tabs(["Content (Cosine)", "Collaborative (Item-ANN)", "Hybrid (Mixed)"])

  with c1:


    st.latex(r"""
    \textbf{Movie vector: } \mathbf{x} \in \mathbb{R}^G \\
    \textbf{Cosine similarity: } 
    \cos(\mathbf{x}, \mathbf{y}) = 
    \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \, \|\mathbf{y}\|} \\
    \textbf{HNSW retrieves top-}k \text{ most similar movies in this vector space.}
    """)
    st.markdown("""
    **Interpretation:** Each movie is represented as a dense **embedding vector** derived from its features 
    (e.g., genres, text description, or learned latent factors).  
    The **cosine similarity** measures how close two movies are in this semantic space:
    values near 1 indicate strong similarity.
    """)



  with c2:
    st.latex(r"""
      \textbf{Item (movie) vector: } 
      \quad v_i[u] = r(u,i) - \bar{r_i} \\[6pt]
      \textbf{User profile vector: } 
      \quad \mathbf{p_u} = 
      \frac{1}{|I_u|}\sum_{i \in I_u, \, r(u,i) > 4} v_i \\[6pt]
      \textbf{Similarity: } 
      \quad \cos(v_i, \mathbf{p_u}) =
      \frac{v_i \cdot \mathbf{p_u}}{\|v_i\|\,\|\mathbf{p_u}\|}
      """)

    st.markdown("**Interpretation:**")
    st.write("""
    1- In collaborative filtering, every **movie** is represented by a vector of user ratings.
    For a given movie *i*, each component shows how much a user *u* liked it, 
    adjusted by subtracting the movie’s **average rating**:
    """)
    st.latex(r"v_i[u] = r(u,i) - \bar{r_i}")
    st.write("""
    2- This creates a **centered item vector**, representing how users deviate from the average.
    """)
    
    st.write("""
    3- Next, a **user profile** is built by averaging the vectors of movies the user rated **above 4**, 
    capturing their personal taste:
    """)
    st.latex(r"\mathbf{p_u} = \frac{1}{|I_u|} \sum_{i \in I_u,\, r(u,i) > 4} v_i")
    
    st.write("""
    4- Finally, the recommender computes the **cosine similarity** between each movie vector and the user profile:
    """)
    st.latex(r"\cos(v_i, \mathbf{p_u}) = \frac{v_i \cdot \mathbf{p_u}}{\|v_i\|\,\|\mathbf{p_u}\|}")
    
    st.markdown("""
    5- A higher cosine value means the movie aligns closely with the user’s preferences —  
    essentially: *“`users who liked similar movies also liked this one`.”*
    """)




  with c3:
    st.latex(r"""
    \textbf{Hybrid combination: } 
    s(m) = \alpha \, s_{content}(m) + \beta \, s_{cf}(m), \quad \alpha+\beta=1
    """)
    st.markdown("""
    In the **mixed hybrid** version, α and β control the **proportion** of recommendations taken 
    from each method.  
    Example: α=0.6 → 60% content-based movies, β=0.4 → 40% collaborative ones.
    """)

  # Why HNSW
  # st.header("🧱 Why ANN (HNSW)?")
  # st.markdown("""
  # - **Fast retrieval:** logarithmic search time with sub-linear complexity.  
  # - **High recall:** ≈95–99% of true neighbors recovered.  
  # - **Scalable:** easily handles tens of thousands of items.  
  # - **Dynamic:** new movies can be added without full recomputation.  
  # """)

  st.subheader("3- How HNSW Works")
  st.write("""
  HNSW (**Hierarchical Navigable Small World**) builds a **multi-layer proximity graph**:  
  1. Each movie is linked to its nearest neighbors.  
  2. Search starts at the top layer and descends using greedy traversal.  
  3. The algorithm retrieves top-k most similar movies efficiently and accurately.
  """)

  # 🧭 How Each Method Works
  st.header("4- How Each Method Works")
  m1, m2, m3 = st.tabs(["🎭 Content-Based", "👥 Collaborative", "⚖️ Hybrid"])

  with m1:
    st.write("""
    1. Represent each movie as a **genre vector** (multi-hot encoding).  
    2. Compute cosine similarity between movies.  
    3. Use ANN (HNSW) to retrieve the top-k closest titles.  
    4. Recommend the most similar movies to the selected **anchor movie**.
    """)
    st.caption("Focus: semantic closeness and feature-based matching.")

  with m2:
    st.write("""
    1. Build a **user–item rating matrix**.  
    2. Mean-center each movie’s ratings.  
    3. Create item vectors based on user behavior.  
    4. Recommend movies similar to those rated **>4** by similar users.
    """)
    st.caption("Focus: learning patterns from collective user preferences.")

  with m3:
    st.write("""
    1. Generate recommendation lists from content and collaborative models.  
    2. Select α×k movies from content and β×k from CF.  
    3. Combine and present them as a unified list.  
    4. The **hybrid score** reflects how both models complement each other.
    """)
    st.caption("Focus: balancing familiarity and discovery.")

  # 📊 Evaluation Metrics
  st.header("5-  Evaluation Metrics")
  st.markdown("""
  Recommender systems are judged not only by accuracy, but also by **experience quality**:  
  Are the results relevant, varied, surprising, and aligned with the user’s interests?  
  Each model (content, collaborative, and hybrid) emphasizes different dimensions of evaluation.
  """)

  e1, e2, e3 = st.tabs(["🎭 Content-Based Metrics", "👥 Collaborative Metrics", "⚖️ Hybrid Metrics"])

  # Content-Based Metrics
  with e1:
    st.subheader("🎭 Content-Based Evaluation")
    st.write("""
    The goal is to ensure that recommended movies are **semantically similar** to the anchor 
    while maintaining **genre diversity**.
    """)

    st.markdown("""
    - **Similarity:** Average cosine similarity between the anchor and recommended movies.  
      High similarity = consistent themes and strong coherence.  
    - **Diversity:** Measures how varied the recommendations are across genres.  
      High diversity = more exploration.
    """)

    col1, col2 = st.columns(2)
    with col1:
      st.metric("Avg Similarity", "0.866")
    with col2:
      st.metric("Genre Diversity", "35.0%")
    st.caption("References: [Cosine Similarity – Medium](https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1) • [Content-Based Recommender – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-content-based-recommender-system/)")

  # Collaborative Metrics
  with e2:
    st.subheader("👥 Collaborative Evaluation")
    st.write("""
    Collaborative filtering is evaluated by **how well it learns user preferences** and proposes 
    relevant unseen items.
    """)

    st.markdown("""
    - **Personalization:** Fraction of recommended movies aligned with what the user rated **>4**.  
    - **Novelty:** Percentage of movies the user has *not* rated before.  
    - **Serendipity:** Measures how many results are both surprising and relevant.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Personalization", "25.0%")
    with col2:
      st.metric("Novelty", "66.7%")
    with col3:
      st.metric("Serendipity", "60.0%")

    st.caption("Reference: [Collaborative Filtering – GeeksforGeeks](https://www.geeksforgeeks.org/collaborative-filtering-in-recommendation-systems/)")

  # Hybrid Metrics
  with e3:
    st.subheader("⚖️ Hybrid Evaluation")
    # st.write("""
    # The hybrid approach merges the best of both worlds.  
    # Evaluation focuses on **balance**, ensuring the model provides both relevance and discovery.
    # """)
      
    st.markdown("""
    The hybrid recommender blends **content** and **collaborative** knowledge.  
    It’s evaluated for its ability to **balance familiarity with discovery**.
    
    - **Balance Score:** How evenly content and user-based signals contribute.  (α × Content + β × CF).    
    - **Personalization:** Retains the user's taste alignment from CF.  
    - **Novelty:** Maintains diversity by exploring unseen titles.  
    - **Serendipity:** Ensures results are both coherent and surprising.
    - **Diversity:** Measures how varied the recommendations are across genres. 
    """)

    # st.markdown("""
    # - **Balance Score:** Weighted sum of content and collaborative qualities (α × Content + β × CF).  
    # - **Exploration Factor:** Indicates how far the system moves beyond known preferences.  
    # """)
    col1, col2 = st.columns(2)
    with col1: st.metric("Balance Score", "0.466")
    with col2: st.metric("Personalization", "23.0%")
        
    col3, col4 = st.columns(2)
    
    with col3: st.metric("Novelty", "61.0%")
    with col4: st.metric("Serendipity", "55.0%")



    st.caption("Reference: [Approximate Nearest Neighbor Search – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/approximate-nearest-neighbor-ann-search/)")

  # References
  st.header("📚 References")
  st.markdown("""
  - Malkov, Y. & Yashunin, D. (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW*. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)  
  - [Movielens Dataset](https://grouplens.org/datasets/movielens/)  
  - [Cosine Similarity – Medium](https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1)  
  - [Content-Based Recommender – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-content-based-recommender-system/)  
  - [Approximate Nearest Neighbor Search – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/approximate-nearest-neighbor-ann-search/)  
  - [Collaborative Filtering – GeeksforGeeks](https://www.geeksforgeeks.org/collaborative-filtering-in-recommendation-systems/)  
  - [Neo4j Graph Data Science Documentation](https://neo4j.com/docs/graph-data-science/current/)
  - [FAISS – Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)  
  - [Annoy – Spotify Approximate Nearest Neighbors](https://github.com/spotify/annoy)  
  - [Voyager – Spotify HNSW-based ANN](https://github.com/spotify/voyager)  
  - [comprehensive-guide-on-item-based-recommendation-systems](https://medium.com/data-science/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d)
  """)

  # st.markdown("---")
  # st.caption("© 2025 ANN Movie Recommender – Powered by Neo4j, HNSWlib, and Streamlit")
  st.markdown("---")
  st.caption("""
  © 2025 **ANN Movie Recommender** – Powered by Neo4j, HNSWlib, and Streamlit  
  👥 **Authors:** [Glorie Metsa WOWO](https://www.linkedin.com/in/glorie-wowo-data-science-edtech) & [Fathnelle Mehouelley](https://www.linkedin.com/in/fathnelle-mehouelley/)
  """)

def show_connection():
    st.title("🔗 Connect to Your Neo4j Database ")
    st.write (" To start using the recommendation system, please connect to your Neo4j instance.")
    st.info("""
    
    **📋 Prerequisites**
    1. **Neo4j Database Running:** via Desktop, AuraDB, or Docker.  
    2. **MovieLens Dataset:** database must contain *Movies* and *Ratings* nodes or `movies.csv` / `ratings.csv` in this app’s folder.  
    3. **Credentials:** have your username and password ready.  
    4. **Access:** if local, visit [http://localhost:7474/](http://localhost:7474/).  
    5. **Need help?** See the [GitHub page](https://github.com/fathnelle4/Movie-Recommendation-System/tree/main) or [Neo4j Docs](https://neo4j.com/docs/).
    """)

    st.warning("Once your credentials are entered, double-click the **Connect** button to establish the connection.")

    
    st.markdown("""Enter your Neo4j credentials.""")
    with st.form("conn_form"):
        uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        user = st.text_input("Username", value="neo4j")
        pwd = st.text_input("Password", type="password", value="")
        submit = st.form_submit_button("Connect", type="primary", use_container_width=True)
    if submit:
        if not pwd:
            st.error("Please enter a password.")
            return
        with st.spinner("Connecting..."):
            driver, err = connect_to_neo4j(uri, user, pwd)
            if driver:
                st.session_state.driver = driver
                st.session_state.connected = True
                st.success("Connected!")
                st.balloons()
            else:
                st.error(f"Connection failed: {err}")

def ensure_data_and_indices():
    # load dataframes
    if st.session_state.movie_df is None:
        st.session_state.movie_df = fetch_movies(st.session_state.driver)
    if st.session_state.ratings_df is None:
        st.session_state.ratings_df = fetch_ratings(st.session_state.driver)

    # content vectors
    if st.session_state.genre_vocab == []:
        st.session_state.genre_vocab = build_genre_vocab(st.session_state.movie_df)

    if st.session_state.movie_vectors_content is None or len(st.session_state.genre_vocab)==0:
        vecs, m2r, r2m = encode_movies_content(st.session_state.movie_df, st.session_state.genre_vocab)
        st.session_state.movie_vectors_content = vecs
        st.session_state.movie_to_row = m2r
        st.session_state.row_to_movie = r2m

    if st.session_state.content_index is None:
        st.session_state.content_index = build_hnsw_index(st.session_state.movie_vectors_content, space="cosine")

    # CF vectors (depends on movie_to_row mapping)
    if st.session_state.movie_vectors_cf is None:
        st.session_state.movie_vectors_cf = encode_movies_cf(st.session_state.movie_df, st.session_state.ratings_df)

    if st.session_state.cf_index is None:
        st.session_state.cf_index = build_hnsw_index(st.session_state.movie_vectors_cf, space="cosine")

def show_dashboard():
    # driver = st.session_state.driver
    # st.title("📊 Recommendations Dashboard (ANN)")
    # st.markdown("---")

    # # with st.spinner("Loading data & building ANN indices (first time only)..."):
    # #     ensure_data_and_indices()
    # with st.spinner("Loading data & building ANN indices (first time only)..."):
    #     data_ready = ensure_data_and_indices()
    
    # if not data_ready:
    #     st.stop()  # stops Streamlit execution safely


    driver = st.session_state.driver
    st.title("📊 Movie Recommendations Dashboard")
    st.markdown("---")
    
    # CHECK DATABASE FIRST - before calling ensure_data_and_indices
    try:
        stats = get_database_stats(driver)
        database_is_empty = stats['movies'] == 0
    except Exception as e:
        st.error(f"Error checking database: {e}")
        database_is_empty = True
    
    # Only build indices if database has data
    if not database_is_empty:
        with st.spinner("Loading data & building ANN indices (first time only)..."):
            ensure_data_and_indices()

            
    

    # Tabs
    tab1,tab2,t1, t2, t3, tab7 = st.tabs(["📤 Upload Data",
                                         "📊 Descriptive Analysis",
                                         "🎯 Content-based",
                                         "👥 Collaborative Filtering",
                                         "🔀 Hybrid", 
                                         # "📈 Quick Data Peek",
                                          "📝 User Profile"
                                        ])
    # ==================== TAB 1: Data Upload ====================
    with tab1:
        st.header("📤 Data Loading")
        st.write("Load MovieLens dataset into Neo4j")
        
        # Check if data already exists
        try:
            data_exists = check_data_exists(driver)
            
            if data_exists:
                st.success("✅ Database already contains data!")
                
                # Show current data stats
                stats = get_database_stats(driver)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Users", f"{stats['users']:,}")
                with col2:
                    st.metric("Movies", f"{stats['movies']:,}")
                with col3:
                    st.metric("Ratings", f"{stats['ratings']:,}")
                with col4:
                    if stats['genres'] > 0:
                        st.metric("Genres", f"{stats['genres']:,}")
                
                st.info("🎉 Your database is ready! Navigate to other tabs to start using recommendations.")
                
                st.markdown("---")
                
                # Option to reload data
                st.subheader("🔄 Reload Data")
                st.write("Want to load different data? You can clear and reload.")
                
                with st.expander("📂 Load New Data"):
                    st.warning("⚠️ This will clear existing data and load from movies.csv and ratings.csv")
                    
                    # Check if files exist
                    import os
                    movies_file_exists = os.path.exists("movies.csv")
                    ratings_file_exists = os.path.exists("ratings.csv")
                    
                    if movies_file_exists and ratings_file_exists:
                        st.success("✅ CSV files found and ready to load")
                        
                        if st.button("🔄 Clear & Reload Data", type="primary"):
                            with st.spinner("Clearing old data and loading new..."):
                                try:
                                    # Clear database
                                    with driver.session() as sess:
                                        sess.run("MATCH (n) DETACH DELETE n")
                                    
                                    # Load new data
                                    movies_df = pd.read_csv("movies.csv")
                                    ratings_df = pd.read_csv("ratings.csv")
                                    
                                    progress_text = st.empty()
                                    progress_bar = st.progress(0)
                                    
                                    def update_progress(message, progress):
                                        progress_text.text(message)
                                        progress_bar.progress(progress)
                                    
                                    success, error = upload_data_to_neo4j(
                                        driver, movies_df, ratings_df, update_progress
                                    )
                                    
                                    if success:
                                        st.success("✅ Data reloaded successfully!")
                                        st.balloons()
                                        st.info("Refresh the page to see new data")
                                    else:
                                        st.error(f"❌ Reload failed: {error}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    else:
                        st.error("❌ CSV files not found. Place movies.csv and ratings.csv in the app folder.")
                
                with st.expander("🗑️ Clear Database (Danger Zone)"):
                    st.error("**WARNING:** This will delete ALL data in your Neo4j database!")
                    
                    confirm = st.text_input("Type 'DELETE ALL DATA' to confirm:", key="confirm_delete")
                    
                    if st.button("🗑️ Clear All Data", type="secondary"):
                        if confirm == "DELETE ALL DATA":
                            with st.spinner("Clearing database..."):
                                try:
                                    with driver.session() as session:
                                        session.run("MATCH (n) DETACH DELETE n")
                                    st.success("✅ Database cleared!")
                                    st.info("Refresh the page to load new data.")
                                    st.balloons()
                                except Exception as e:
                                    st.error(f"Error clearing database: {e}")
                        else:
                            st.error("Please type 'DELETE ALL DATA' to confirm deletion.")
            
            else:
                st.info("📂 Database is empty. Let's load the MovieLens dataset!")
                
                st.markdown("""
                ### 📋 Requirements
                
                Place these files in the **same directory** as the application:
                - `movies.csv` (movieId, title, genres)
                - `ratings.csv` (userId, movieId, rating, timestamp)
                
                **Expected format:**
                
                **movies.csv:**
                ```
                movieId,title,genres
                1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
                2,Jumanji (1995),Adventure|Children|Fantasy
                ```
                
                **ratings.csv:**
                ```
                userId,movieId,rating,timestamp
                1,1,4.0,964982703
                1,3,4.0,964981247
                ```
                """)
                
                st.markdown("---")
                
                # Check if files exist
                import os
                movies_file_exists = os.path.exists("movies.csv")
                ratings_file_exists = os.path.exists("ratings.csv")
                
                col1, col2 = st.columns(2)
                with col1:
                    if movies_file_exists:
                        st.success("✅ movies.csv found")
                        try:
                            movies_df = pd.read_csv("movies.csv")
                            st.write(f"**{len(movies_df):,} movies**")
                            with st.expander("Preview movies.csv"):
                                st.dataframe(movies_df.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error reading movies.csv: {e}")
                            movies_df = None
                    else:
                        st.error("❌ movies.csv not found")
                        st.info("Place movies.csv in the same folder as this app")
                        movies_df = None
                
                with col2:
                    if ratings_file_exists:
                        st.success("✅ ratings.csv found")
                        try:
                            ratings_df = pd.read_csv("ratings.csv")
                            st.write(f"**{len(ratings_df):,} ratings**")
                            with st.expander("Preview ratings.csv"):
                                st.dataframe(ratings_df.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error reading ratings.csv: {e}")
                            ratings_df = None
                    else:
                        st.error("❌ ratings.csv not found")
                        st.info("Place ratings.csv in the same folder as this app")
                        ratings_df = None
                
                st.markdown("---")
                
                # Load button
                if movies_file_exists and ratings_file_exists and movies_df is not None and ratings_df is not None:
                    # Validate data
                    movies_valid = all(col in movies_df.columns for col in ['movieId', 'title', 'genres'])
                    ratings_valid = all(col in ratings_df.columns for col in ['userId', 'movieId', 'rating', 'timestamp'])
                    
                    if not movies_valid:
                        st.error("❌ movies.csv missing required columns: movieId, title, genres")
                    elif not ratings_valid:
                        st.error("❌ ratings.csv missing required columns: userId, movieId, rating, timestamp")
                    else:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            st.markdown("### Ready to Load!")
                            st.write(f"📊 **{len(movies_df):,}** movies")
                            st.write(f"⭐ **{len(ratings_df):,}** ratings")
                            st.write(f"👥 **{ratings_df['userId'].nunique():,}** users")
                            
                            if st.button("🚀 Load Data into Neo4j", type="primary", use_container_width=True):
                                # Progress tracking
                                progress_text = st.empty()
                                progress_bar = st.progress(0)
                                
                                def update_progress(message, progress):
                                    progress_text.text(message)
                                    progress_bar.progress(progress)
                                
                                try:
                                    success, error = upload_data_to_neo4j(
                                        driver, 
                                        movies_df, 
                                        ratings_df, 
                                        progress_callback=update_progress
                                    )
                                    
                                    if success:
                                        progress_text.empty()
                                        progress_bar.empty()
                                        
                                        st.success("🎉 Data loaded successfully!")
                                        st.balloons()
                                        
                                        # Show summary
                                        st.markdown("### 📊 Upload Summary")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Movies", f"{len(movies_df):,}")
                                        with col2:
                                            st.metric("Ratings", f"{len(ratings_df):,}")
                                        with col3:
                                            st.metric("Users", f"{ratings_df['userId'].nunique():,}")
                                        with col4:
                                            genres = set()
                                            for g in movies_df['genres'].dropna():
                                                genres.update([x.strip() for x in g.split('|') if x.strip()])
                                            st.metric("Genres", f"{len(genres):,}")
                                        
                                        st.info("✨ Navigate to **Descriptive Analysis** or other tabs to explore your data!")
                                        
                                        # Refresh button
                                        if st.button("🔄 Refresh Page"):
                                            st.rerun()
                                    else:
                                        st.error(f"❌ Upload failed: {error}")
                                
                                except Exception as e:
                                    st.error(f"❌ Upload failed: {str(e)}")
                                    st.exception(e)
                
                else:
                    st.warning("⚠️ Please ensure both movies.csv and ratings.csv are in the same folder as this application")
                    
                    st.markdown("---")
                    st.markdown("""
                    ### 💡 Tips
                    
                    1. **Download MovieLens Dataset**: Visit [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)
                    2. **Extract Files**: Unzip and copy movies.csv and ratings.csv
                    3. **Place in App Folder**: Put them in the same directory as `movie_recommender_app.py`
                    4. **Refresh**: Click the refresh button above or press 'R' to reload
                    5. **Large Datasets**: 100K ratings take ~2 minutes, 1M+ ratings take ~10-15 minutes
                    """)
        
        except Exception as e:
            st.error(f"Error checking database: {str(e)}")
            st.info("Make sure your Neo4j connection is working properly.")
    
    
    # ==================== TAB 2: Descriptive Analysis ====================
    with tab2:
                # ✅ Check if database has data before showing analysis
      if database_is_empty:
            st.warning("⚠️ **Database is empty!**")
            st.info(" Please load data using the **'Upload Data'** tab first")
        
      else:
        st.header("📊 Descriptive Analysis")
        st.write("Explore your dataset with comprehensive statistics and visualizations")
        
        if st.button("🔄 Refresh Data", key="refresh_analysis"):
            st.rerun()
        
        try:
            # Get statistics
            stats = get_descriptive_stats(driver)
            
            # Overview metrics
            st.subheader("📈 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Users", f"{stats['total_users']:,}")
            with col2:
                st.metric("Total Movies", f"{stats['total_movies']:,}")
            with col3:
                st.metric("Total Ratings", f"{stats['total_ratings']:,}")
            with col4:
                if stats['total_genres'] > 0:
                    st.metric("Total Genres", f"{stats['total_genres']:,}")
            
            st.markdown("---")
            
            # Rating statistics
            st.subheader("⭐ Rating Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Rating", f"{stats['avg_rating']:.2f}")
            with col2:
                st.metric("Min Rating", f"{stats['min_rating']:.1f}")
            with col3:
                st.metric("Max Rating", f"{stats['max_rating']:.1f}")
            with col4:
                st.metric("Std Dev", f"{stats['std_rating']:.2f}")
            
            # Rating distribution
            st.subheader("📊 Rating Distribution")
            rating_dist_df = get_rating_distribution(driver)
            
            if not rating_dist_df.empty:
                fig = px.bar(rating_dist_df, x='rating', y='count',
                            title='Distribution of Ratings',
                            labels={'rating': 'Rating', 'count': 'Number of Ratings'},
                            color='count',
                            color_continuous_scale='viridis')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # User and Movie statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👥 User Activity")
                st.metric("Avg Ratings per User", f"{stats['avg_ratings_per_user']:.1f}")
                st.metric("Most Active User", f"{stats['max_ratings_per_user']:,} ratings")
                st.metric("Least Active User", f"{stats['min_ratings_per_user']:,} ratings")
            
            with col2:
                st.subheader("🎬 Movie Popularity")
                st.metric("Avg Ratings per Movie", f"{stats['avg_ratings_per_movie']:.1f}")
                st.metric("Most Rated Movie", f"{stats['max_ratings_per_movie']:,} ratings")
                st.metric("Least Rated Movie", f"{stats['min_ratings_per_movie']:,} ratings")
            
            st.markdown("---")
            
            # Top movies and users
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Rated Movies")
                top_movies = get_top_movies(driver, limit=15)
                if not top_movies.empty:
                    fig = px.bar(top_movies.head(15), 
                                x='avg_rating', 
                                y='title',
                                orientation='h',
                                title='Top 15 Movies by Average Rating (min 50 ratings)',
                                labels={'avg_rating': 'Average Rating', 'title': 'Movie'},
                                color='avg_rating',
                                color_continuous_scale='blues',
                                text='avg_rating')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("👤 Most Active Users")
                top_users = get_top_users(driver, limit=15)
                if not top_users.empty:
                    fig = px.bar(top_users.head(15),
                                x='num_ratings',
                                y='userId',
                                orientation='h',
                                title='Top 15 Most Active Users',
                                labels={'num_ratings': 'Number of Ratings', 'userId': 'User ID'},
                                color='num_ratings',
                                color_continuous_scale='reds',
                                text='num_ratings')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Genre analysis
            st.subheader("🎭 Genre Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Genre Distribution")
                genre_dist = get_genre_distribution(driver)
                if not genre_dist.empty:
                    fig = px.pie(genre_dist.head(10), 
                                values='count', 
                                names='genre',
                                title='Top 10 Genres by Number of Movies',
                                hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Average Rating by Genre")
                genre_ratings = get_genre_ratings(driver)
                if not genre_ratings.empty:
                    fig = px.bar(genre_ratings.head(10),
                                x='avg_rating',
                                y='genre',
                                orientation='h',
                                title='Top 10 Genres by Average Rating (min 100 ratings)',
                                labels={'avg_rating': 'Average Rating', 'genre': 'Genre'},
                                color='avg_rating',
                                color_continuous_scale='greens',
                                text='avg_rating')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Ratings over time
            st.subheader("📅 Ratings Over Time")
            ratings_time = get_ratings_over_time(driver)
            if not ratings_time.empty:
                # Aggregate by month for better visualization
                ratings_time['month'] = ratings_time['rating_date'].dt.to_period('M')
                monthly = ratings_time.groupby('month')['count'].sum().reset_index()
                monthly['month'] = monthly['month'].astype(str)
                
                fig = px.line(monthly, x='month', y='count',
                            title='Number of Ratings Over Time (Monthly)',
                            labels={'month': 'Month', 'count': 'Number of Ratings'},
                            markers=True)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timestamp data available for time-based analysis")
            
            st.markdown("---")
            

            # Data quality metrics
            st.subheader("🔍 Data Quality Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sparsity = 1 - (stats['total_ratings'] / (stats['total_users'] * stats['total_movies']))
                st.metric("Data Sparsity", f"{sparsity:.2%}")
                st.caption("Percentage of missing user-movie ratings")
            
            with col2:
                coverage_users = (stats['total_ratings'] / stats['total_users'])
                st.metric("User Coverage", f"{coverage_users:.1f}")
                st.caption("Average ratings per user (user activity level)")
            
            with col3:
                coverage_movies = (stats['total_ratings'] / stats['total_movies'])
                st.metric("Movie Coverage", f"{coverage_movies:.1f}")
                st.caption("Average number of ratings each movie received (popularity)")
            
            with col4:
                avg_movie_score = stats.get('avg_rating', None)
                if avg_movie_score:
                    st.metric("Average Rating Value", f"{avg_movie_score:.2f}")
                    st.caption("How much movies are liked overall (1–5 scale)")
                else:
                    st.warning("Average movie rating unavailable.")
            
            # Interpretation section
            st.markdown("---")
            st.subheader("📊 Data Interpretation")
            
            # Sparsity interpretation
            if sparsity > 0.99:
                st.warning("⚠️ **High Sparsity**: Very few user–movie pairs are rated. This may affect recommendation accuracy.")
            elif sparsity > 0.95:
                st.info("ℹ️ **Moderate Sparsity**: Typical for MovieLens-like datasets — users rate selectively.")
            else:
                st.success("✅ **Low Sparsity**: Good data density for recommendation modeling!")
            
            # Rating bias interpretation
            if stats['avg_rating'] > 3.5:
                st.info(f"📈 **Positive Rating Bias**: Average rating ({stats['avg_rating']:.2f}) is above neutral, indicating that users tend to give high scores (e.g., 4★ and 5★).")
            elif stats['avg_rating'] < 2.5:
                st.warning(f"📉 **Negative Rating Bias**: Average rating ({stats['avg_rating']:.2f}) is below neutral — users are stricter or more critical.")
            else:
                st.success(f"⚖️ **Balanced Ratings**: Average rating ({stats['avg_rating']:.2f}) is around neutral (3★).")
            
        except Exception as e:
                    st.error(f"Error loading statistics: {str(e)}")
                    st.info("Make sure your database has data loaded. Try the 'Upload Data' tab first!")
    
                  
          #--------------
    
    with t1:
      if database_is_empty:
            st.warning("⚠️ **Database is empty!**")
            st.info(" Please load data using the **'Upload Data'** tab first")
        
      else:
        st.subheader("Content-based (ANN on genre vectors)")
        movies = st.session_state.movie_df["title"].tolist()
        movie_sel = st.selectbox("Select a movie", movies)
        k = st.slider("Top-K", 5, 30, 10)
        if st.button("Find similar (Content ANN)"):
            
            df = recommend_content_ann(movie_sel, k)
            
 
            
            
            if df.empty:
                st.warning("No results.")
            else:
                st.dataframe(df, use_container_width=True)
                
                # --- Evaluation ---
      
                    # Get selected movie genres
                selected_genres = st.session_state.movie_df[
                    st.session_state.movie_df['title'] == movie_sel
                ].iloc[0]['genres']
                
                # Evaluate
                metrics = evaluate_content_based(df, selected_genres, k)
                display_content_based_metrics(metrics, st)


    with t2:
      if database_is_empty:
            st.warning("⚠️ **Database is empty!**")
            st.info(" Please load data using the **'Upload Data'** tab first")
        
      else:
        st.subheader("Collaborative (ANN on item vectors)")
        users = fetch_users(st.session_state.driver)
        if not users:
            st.warning("No users found.")
        else:
            uid = st.selectbox("Select user ID", users)
            k2 = st.slider("Top-K", 5, 30, 10, key="k2")
            
            if st.button("Recommend for user (CF ANN)"):
                df = recommend_cf_ann(uid, k2)
                if df.empty:
                    st.warning("No results (user may have few ratings).")
                else:
                    st.dataframe(df, use_container_width=True)

                    
                                    # --- Evaluation ---
                        # Evaluate
                    metrics = evaluate_collaborative(df, st.session_state.driver, uid, k)
                    display_collaborative_metrics(metrics, st)

                    

    with t3:
        
      if database_is_empty:
            st.warning("⚠️ **Database is empty!**")
            st.info(" Please load data using the **'Upload Data'** tab first")
        
      else:       
        st.subheader("Hybrid (α*Content + β*CF) via ANN")
        users = fetch_users(st.session_state.driver)
        movies = st.session_state.movie_df["title"].tolist()
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            uid_h = st.selectbox("User ID", users, key="uid_h") if users else None
        with c2:
            anchor = st.selectbox("Anchor movie (optional but recommended)", movies, key="anchor_h") if movies else None
        with c3:
            k3 = st.slider("Top-K", 5, 30, 10, key="k3")
        alpha = st.slider("α (content weight)", 0.0, 1.0, 0.6, 0.05)
        beta = 1.0 - alpha
        st.caption(f"β (CF weight) = {beta:.2f}")
        if st.button("Hybrid recommend (ANN)"):
            if uid_h is None:
                st.error("Pick a user.")
            else:
                df = recommend_hybrid_ann(uid_h, anchor, k3, alpha=alpha, beta=beta)
                if df is None or df.empty:
                    st.warning("No results.")
                else:
                    st.dataframe(df, use_container_width=True)

                                    # --- Evaluation ---
                   # Get anchor genres
                    anchor_genres = st.session_state.movie_df[
                        st.session_state.movie_df['title'] == anchor
                    ].iloc[0]['genres']
                    
                    # Evaluate
                    metrics = evaluate_hybrid(df, st.session_state.driver, uid_h, anchor_genres, alpha, beta, k)
                    display_hybrid_metrics(metrics, st)

    # with t4:
    #     st.subheader("Dataset peek")
    #     m = st.session_state.movie_df.head(20)
    #     r = st.session_state.ratings_df.head(20)
    #     c1, c2 = st.columns(2)
    #     with c1:
    #         st.write("Movies")
    #         st.dataframe(m, use_container_width=True)
    #     with c2:
    #         st.write("Ratings")
    #         st.dataframe(r, use_container_width=True)

    # ==================== TAB 7: User Profile ====================
    with tab7:
      if database_is_empty:
            st.warning("⚠️ **Database is empty!**")
            st.info(" Please load data using the **'Upload Data'** tab first")
        
      else:
        st.header("📝 User Profile")
        st.write("Explore what a user has watched and rated")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            try:
                users = get_all_users(driver)
                selected_user = st.selectbox(
                    "Select a user ID:",
                    users,
                    key="profile_user"
                )
            except Exception as e:
                st.error(f"Error loading users: {e}")
                selected_user = None
        
        with col2:
            num_ratings = st.slider(
                "Number of ratings to show:",
                min_value=5,
                max_value=50,
                value=10,
                key="profile_slider"
            )
        
        if st.button("View Profile", key="profile_btn", type="primary"):
            if selected_user:
                with st.spinner("Loading user profile..."):
                    try:
                        ratings = get_user_ratings(driver, selected_user, num_ratings)
                        
                        if ratings:
                            st.success(f"User {selected_user} profile loaded (showing top {len(ratings)} ratings)")
                            
                            df = pd.DataFrame(ratings)
                            df.index = df.index + 1
                            
                            st.dataframe(
                                df,
                                use_container_width=True,
                                column_config={
                                    "title": st.column_config.TextColumn("Movie Title", width="large"),
                                    "genres": st.column_config.TextColumn("Genres", width="medium"),
                                    "rating": st.column_config.NumberColumn("Rating", format="⭐ %.1f")
                                }
                            )
                            
                            # Statistics
                            avg_rating = sum([r['rating'] for r in ratings]) / len(ratings)
                            st.metric("Average Rating", f"⭐ {avg_rating:.2f}")
                        else:
                            st.warning("No ratings found for this user!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
# ===============================
# Main
# ===============================
def main():
    show_sidebar()
    if st.session_state.current_page == "landing":
        show_landing()
    elif st.session_state.current_page == "connection":
        show_connection()
    else:
        if st.session_state.connected and st.session_state.driver is not None:
            show_dashboard()
        else:
            st.warning("Please connect to Neo4j first.")
            st.session_state.current_page = "connection"

if __name__ == "__main__":
    main()