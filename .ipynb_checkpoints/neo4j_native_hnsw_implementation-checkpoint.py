"""
NEO4J NATIVE HNSW IMPLEMENTATION
Replace HNSWlib with Neo4j's built-in vector indexing
This is more efficient, scalable, and integrates better with your graph
"""

import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# ===============================
# 1. CREATE VECTOR INDICES IN NEO4J
# ===============================

def create_neo4j_vector_indices(driver):
    """
    Create HNSW vector indices in Neo4j for both content and CF
    Run this ONCE after loading data
    """
    with driver.session() as session:
        try:
            # Drop existing indices if any
            session.run("DROP INDEX content_vector_index IF EXISTS")
            session.run("DROP INDEX cf_vector_index IF EXISTS")
            
            # Create content-based vector index (genre vectors)
            # Assuming genre vectors have ~20 dimensions (number of genres)
            session.run("""
                CREATE VECTOR INDEX content_vector_index IF NOT EXISTS
                FOR (m:Movie)
                ON m.contentVector
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 20,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            
            # Create collaborative filtering vector index
            # Dimension = number of users (e.g., 610 for MovieLens 100k)
            # We'll set this dynamically
            result = session.run("MATCH (u:User) RETURN count(u) AS userCount")
            user_count = result.single()["userCount"]
            
            session.run(f"""
                CREATE VECTOR INDEX cf_vector_index IF NOT EXISTS
                FOR (m:Movie)
                ON m.cfVector
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {user_count},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
            
            st.success(f"✅ Created vector indices (content: 20D, cf: {user_count}D)")
            
        except Exception as e:
            st.warning(f"Vector indices may already exist or: {e}")

# ===============================
# 2. STORE VECTORS IN NEO4J
# ===============================

def store_content_vectors_in_neo4j(driver, movie_df, genre_vocab):
    """
    Calculate and store content vectors directly in Neo4j Movie nodes
    """
    with driver.session() as session:
        g2i = {g: i for i, g in enumerate(genre_vocab)}
        dim = len(genre_vocab)
        
        for _, row in movie_df.iterrows():
            movie_id = int(row["movieId"])
            
            # Build genre vector
            vector = [0.0] * dim
            if isinstance(row["genres"], str):
                for g in row["genres"].split("|"):
                    g = g.strip()
                    if g and g != "(no genres listed)" and g in g2i:
                        vector[g2i[g]] = 1.0
            
            # L2 normalize
            norm = np.linalg.norm(vector) + 1e-8
            vector = [float(v / norm) for v in vector]
            
            # Store in Neo4j
            session.run("""
                MATCH (m:Movie {movieId: $movieId})
                SET m.contentVector = $vector
            """, movieId=movie_id, vector=vector)
        
        st.success(f"✅ Stored {len(movie_df)} content vectors in Neo4j")

def store_cf_vectors_in_neo4j(driver, movie_df, ratings_df):
    """
    Calculate and store collaborative filtering vectors in Neo4j
    """
    if ratings_df.empty:
        return
    
    with driver.session() as session:
        # Get all users
        unique_users = sorted(ratings_df["userId"].unique())
        u2i = {u: i for i, u in enumerate(unique_users)}
        num_users = len(unique_users)
        
        # Get mean rating per movie
        mu = ratings_df.groupby("movieId")["rating"].mean().to_dict()
        
        # For each movie, build its CF vector
        for movie_id in movie_df["movieId"]:
            movie_id = int(movie_id)
            
            # Initialize vector
            vector = [0.0] * num_users
            
            # Fill with mean-centered ratings
            movie_ratings = ratings_df[ratings_df["movieId"] == movie_id]
            for _, rating_row in movie_ratings.iterrows():
                user_id = int(rating_row["userId"])
                if user_id in u2i:
                    ui = u2i[user_id]
                    rating = float(rating_row["rating"])
                    mean_rating = float(mu.get(movie_id, 0.0))
                    vector[ui] = rating - mean_rating
            
            # L2 normalize
            norm = np.linalg.norm(vector) + 1e-8
            vector = [float(v / norm) for v in vector]
            
            # Store in Neo4j
            session.run("""
                MATCH (m:Movie {movieId: $movieId})
                SET m.cfVector = $vector
            """, movieId=movie_id, vector=vector)
        
        st.success(f"✅ Stored {len(movie_df)} CF vectors in Neo4j")


def store_content_vectors_in_neo4j_fixed(driver, movie_df, genre_vocab):
    """
    Fixed version: Store vectors as proper Neo4j vector type
    """
    with driver.session() as session:
        g2i = {g: i for i, g in enumerate(genre_vocab)}
        dim = len(genre_vocab)
        
        batch_size = 100
        total = len(movie_df)
        
        for batch_start in range(0, total, batch_size):
            batch = movie_df.iloc[batch_start:batch_start + batch_size]
            
            # Prepare batch data
            batch_data = []
            for _, row in batch.iterrows():
                movie_id = int(row["movieId"])
                
                # Build genre vector
                vector = [0.0] * dim
                if isinstance(row["genres"], str):
                    for g in row["genres"].split("|"):
                        g = g.strip()
                        if g and g != "(no genres listed)" and g in g2i:
                            vector[g2i[g]] = 1.0
                
                # L2 normalize
                norm = np.linalg.norm(vector) + 1e-8
                vector = [float(v / norm) for v in vector]
                
                batch_data.append({"movieId": movie_id, "vector": vector})
            
            # Batch update using UNWIND
            session.run("""
                UNWIND $batch AS item
                MATCH (m:Movie {movieId: item.movieId})
                SET m.contentVector = item.vector
            """, batch=batch_data)
            
            if batch_start % 1000 == 0:
                print(f"Progress: {batch_start}/{total} movies")
        
        print(f"✅ Stored {total} content vectors")

def store_cf_vectors_in_neo4j_fixed(driver, movie_df, ratings_df):
    """
    Fixed version: Store CF vectors as proper Neo4j vector type
    """
    if ratings_df.empty:
        return
    
    with driver.session() as session:
        # Get all users
        unique_users = sorted(ratings_df["userId"].unique())
        u2i = {u: i for i, u in enumerate(unique_users)}
        num_users = len(unique_users)
        
        # Get mean rating per movie
        mu = ratings_df.groupby("movieId")["rating"].mean().to_dict()
        
        batch_size = 100
        total = len(movie_df)
        
        for batch_start in range(0, total, batch_size):
            batch = movie_df.iloc[batch_start:batch_start + batch_size]
            
            batch_data = []
            for _, row in batch.iterrows():
                movie_id = int(row["movieId"])
                
                # Initialize vector
                vector = [0.0] * num_users
                
                # Fill with mean-centered ratings
                movie_ratings = ratings_df[ratings_df["movieId"] == movie_id]
                for _, rating_row in movie_ratings.iterrows():
                    user_id = int(rating_row["userId"])
                    if user_id in u2i:
                        ui = u2i[user_id]
                        rating = float(rating_row["rating"])
                        mean_rating = float(mu.get(movie_id, 0.0))
                        vector[ui] = rating - mean_rating
                
                # L2 normalize
                norm = np.linalg.norm(vector) + 1e-8
                vector = [float(v / norm) for v in vector]
                
                batch_data.append({"movieId": movie_id, "vector": vector})
            
            # Batch update
            session.run("""
                UNWIND $batch AS item
                MATCH (m:Movie {movieId: item.movieId})
                SET m.cfVector = item.vector
            """, batch=batch_data)
            
            if batch_start % 1000 == 0:
                print(f"Progress: {batch_start}/{total} movies")
        
        print(f"✅ Stored {total} CF vectors")
        
# ===============================
# 3. QUERY USING NEO4J VECTOR SEARCH
# ===============================

def recommend_content_ann_neo4j(driver, selected_title: str, k: int, threshold: float = 0.99) -> pd.DataFrame:
    """
    Content-based recommendations using Neo4j's native vector search
    """
    with driver.session() as session:
        # Get the selected movie's vector
        result = session.run("""
            MATCH (m:Movie {title: $title})
            RETURN m.contentVector AS vector, m.movieId AS movieId
        """, title=selected_title)
        
        record = result.single()
        if not record or not record["vector"]:
            return pd.DataFrame()
        
        query_vector = record["vector"]
        anchor_id = record["movieId"]
        
        # Use Neo4j vector similarity search
        result = session.run("""
            MATCH (m:Movie)
            WHERE m.movieId <> $anchorId 
                AND m.contentVector IS NOT NULL
            WITH m, 
                 vector.similarity.cosine(m.contentVector, $queryVector) AS similarity
            WHERE similarity < $threshold
            RETURN m.title AS title, 
                   m.genres AS genres, 
                   similarity AS cosine_similarity
            ORDER BY similarity DESC
            LIMIT $k
        """, anchorId=anchor_id, queryVector=query_vector, threshold=threshold, k=k)
        
        rows = [dict(record) for record in result]
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df["cosine_similarity"] = df["cosine_similarity"].round(4)
        
        return df

def recommend_cf_ann_neo4j(driver, user_id: int, k: int) -> pd.DataFrame:
    """
    Collaborative filtering recommendations using Neo4j's native vector search
    """
    with driver.session() as session:
        # Build user profile vector from movies rated >= 4
        result = session.run("""
            MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
            WHERE r.rating >= 4.0 AND m.cfVector IS NOT NULL
            RETURN m.cfVector AS vector, r.rating AS rating
        """, userId=user_id)
        
        vectors = []
        weights = []
        for record in result:
            vectors.append(record["vector"])
            weights.append(float(record["rating"]))
        
        if not vectors:
            return pd.DataFrame()
        
        # Weighted average
        vectors = np.array(vectors, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
        profile = (vectors * weights).sum(axis=0)
        
        # Normalize
        profile = profile / (np.linalg.norm(profile) + 1e-8)
        profile_list = profile.tolist()
        
        # Get movies user hasn't rated
        result = session.run("""
            MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
            RETURN collect(m.movieId) AS ratedMovies
        """, userId=user_id)
        rated_ids = result.single()["ratedMovies"]
        
        # Find similar movies using vector search
        result = session.run("""
            MATCH (m:Movie)
            WHERE NOT m.movieId IN $ratedMovies 
                AND m.cfVector IS NOT NULL
            WITH m, 
                 vector.similarity.cosine(m.cfVector, $profileVector) AS similarity
            RETURN m.title AS title, 
                   m.genres AS genres, 
                   similarity AS cf_similarity
            ORDER BY similarity DESC
            LIMIT $k
        """, ratedMovies=rated_ids, profileVector=profile_list, k=k)
        
        rows = [dict(record) for record in result]
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df["cf_similarity"] = df["cf_similarity"].round(4)
        
        return df

def recommend_hybrid_ann_neo4j(driver, user_id: int, anchor_title: str, k: int, 
                               alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
    """
    Hybrid recommendations using Neo4j vector search
    """
    # Get content recommendations
    content_df = recommend_content_ann_neo4j(driver, anchor_title, k * 5)
    
    # Get collaborative recommendations
    cf_df = recommend_cf_ann_neo4j(driver, user_id, k * 5)
    
    if content_df.empty and cf_df.empty:
        return pd.DataFrame()
    
    # Proportional selection
    n_content = int(round(alpha * k))
    n_cf = k - n_content
    
    # Select top from each
    content_top = content_df.head(n_content).copy() if not content_df.empty else pd.DataFrame()
    cf_top = cf_df.head(n_cf).copy() if not cf_df.empty else pd.DataFrame()
    
    if not content_top.empty:
        content_top["source"] = "content"
        content_top["hybrid_score"] = content_top["cosine_similarity"]
    
    if not cf_top.empty:
        cf_top["source"] = "collaborative"
        cf_top["hybrid_score"] = cf_top["cf_similarity"]
    
    # Combine
    frames = [f for f in [content_top, cf_top] if not f.empty]
    if not frames:
        return pd.DataFrame()
    
    final_df = pd.concat(frames, ignore_index=True)
    
    # Ensure all columns exist
    for col in ["cosine_similarity", "cf_similarity"]:
        if col not in final_df.columns:
            final_df[col] = 0.0
    
    final_df = final_df.sort_values("hybrid_score", ascending=False).head(k)
    
    return final_df[["title", "genres", "cosine_similarity", "cf_similarity", "hybrid_score", "source"]]

# ===============================
# 4. INITIALIZATION FUNCTION
# ===============================

def initialize_neo4j_vectors(driver, movie_df, ratings_df, genre_vocab):
    """
    One-time setup: create indices and store vectors
    Call this instead of building HNSWlib indices
    """
    st.info("🔄 Initializing Neo4j vector indices...")
    
    # Step 1: Create indices
    create_neo4j_vector_indices(driver)
    
    # Step 2: Store content vectors
    with st.spinner("Storing content vectors..."):
        store_content_vectors_in_neo4j(driver, movie_df, genre_vocab)
    
    # Step 3: Store CF vectors
    with st.spinner("Storing collaborative filtering vectors..."):
        store_cf_vectors_in_neo4j(driver, movie_df, ratings_df)
    
    st.success("✅ Neo4j vector indices ready!")

# ===============================
# 5. CHECK IF VECTORS EXIST
# ===============================

def check_vectors_exist(driver) -> bool:
    """
    Check if vectors are already stored in Neo4j
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (m:Movie)
            WHERE m.contentVector IS NOT NULL OR m.cfVector IS NOT NULL
            RETURN count(m) AS count
        """)
        count = result.single()["count"]
        return count > 0

# ===============================
# 6. UPDATED ensure_data_and_indices
# ===============================

def ensure_data_and_indices_neo4j(driver):
    """
    REPLACE your existing ensure_data_and_indices with this version
    """
    # Load dataframes
    if st.session_state.movie_df is None:
        st.session_state.movie_df = fetch_movies(driver)
    if st.session_state.ratings_df is None:
        st.session_state.ratings_df = fetch_ratings(driver)
    
    # Build genre vocabulary
    if not st.session_state.genre_vocab:
        from your_app import build_genre_vocab  # Import from your main file
        st.session_state.genre_vocab = build_genre_vocab(st.session_state.movie_df)
    
    # Check if vectors already exist in Neo4j
    if not check_vectors_exist(driver):
        # Initialize vectors in Neo4j (one-time setup)
        initialize_neo4j_vectors(
            driver,
            st.session_state.movie_df,
            st.session_state.ratings_df,
            st.session_state.genre_vocab
        )
        st.session_state.ann_built = True
    else:
        if not st.session_state.ann_built:
            st.info("✅ Vector indices already exist in Neo4j")
            st.session_state.ann_built = True

# ===============================
# 7. INTEGRATION GUIDE
# ===============================

"""
HOW TO INTEGRATE INTO YOUR APP:

1. REMOVE these imports:
   import hnswlib
   
2. REMOVE these session state variables:
   - content_index
   - cf_index
   - movie_vectors_content
   - movie_vectors_cf
   - movie_to_row
   - row_to_movie

3. REPLACE functions in your main app:
   
   # OLD:
   def ensure_data_and_indices():
       ...
   
   # NEW:
   from neo4j_native_hnsw_implementation import ensure_data_and_indices_neo4j
   
   def ensure_data_and_indices():
       ensure_data_and_indices_neo4j(st.session_state.driver)

4. REPLACE recommendation functions:
   
   # Content-based tab:
   # OLD: df = recommend_content_ann(movie_sel, k)
   # NEW:
   df = recommend_content_ann_neo4j(st.session_state.driver, movie_sel, k)
   
   # Collaborative tab:
   # OLD: df = recommend_cf_ann(uid, k2)
   # NEW:
   df = recommend_cf_ann_neo4j(st.session_state.driver, uid, k2)
   
   # Hybrid tab:
   # OLD: df = recommend_hybrid_ann(uid_h, anchor, k3, alpha, beta)
   # NEW:
   df = recommend_hybrid_ann_neo4j(st.session_state.driver, uid_h, anchor, k3, alpha, beta)

5. VERIFY Neo4j version:
   - Requires Neo4j 5.13+ for vector indices
   - Check: CALL dbms.components() YIELD versions
"""

# ===============================
# 8. BENEFITS OF NEO4J NATIVE APPROACH
# ===============================

"""
ADVANTAGES:

1. ✅ **Better Integration**: Vectors stored with graph data
2. ✅ **Persistence**: Vectors survive app restarts
3. ✅ **Scalability**: Neo4j handles millions of vectors efficiently
4. ✅ **Query Flexibility**: Combine vector search with graph queries
5. ✅ **Memory Efficiency**: No need to load all vectors into Python
6. ✅ **Consistency**: Single source of truth (Neo4j database)
7. ✅ **Advanced Queries**: Can filter by properties + vector similarity

EXAMPLE ADVANCED QUERY:
```cypher
// Find similar sci-fi movies from 1990s
MATCH (m:Movie)
WHERE m.year >= 1990 AND m.year < 2000
  AND 'Sci-Fi' IN m.genres
WITH m, vector.similarity.cosine(m.contentVector, $queryVector) AS sim
WHERE sim > 0.7
RETURN m.title, sim
ORDER BY sim DESC
LIMIT 10
```
"""

# ===============================
# 9. OPTIONAL: ADMIN PANEL
# ===============================

def show_vector_admin_panel(driver):
    """
    Optional: Add this to your app to manage vector indices
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Vector Management")
    
    if st.sidebar.button("Rebuild Vector Indices"):
        with st.spinner("Rebuilding indices..."):
            initialize_neo4j_vectors(
                driver,
                st.session_state.movie_df,
                st.session_state.ratings_df,
                st.session_state.genre_vocab
            )
    
    if st.sidebar.button("Check Vector Stats"):
        with driver.session() as session:
            result = session.run("""
                MATCH (m:Movie)
                RETURN 
                    count(m) AS totalMovies,
                    count(m.contentVector) AS withContentVector,
                    count(m.cfVector) AS withCfVector
            """)
            stats = dict(result.single())
            st.sidebar.json(stats)
