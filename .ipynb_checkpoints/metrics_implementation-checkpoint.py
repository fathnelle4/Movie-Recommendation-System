"""
PROPER EVALUATION METRICS FOR RECOMMENDATION SYSTEMS
Implements metrics that actually make sense for each recommendation type
"""

import numpy as np
import pandas as pd
from collections import Counter
import re

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def upload_data_to_neo4j(driver, movies_df, ratings_df, progress_callback=None):
    """
    Upload movies and ratings data to Neo4j database
    Uses the proven batch loading method
    """
    try:
        # Load movies
        if progress_callback:
            progress_callback("Loading movies...", 10)
        
        with driver.session() as session:
            # Create constraints first
            try:
                session.run("CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
                session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
            except:
                pass  # Constraints might already exist
            
            # Load movies one by one (more reliable)
            for idx, row in movies_df.iterrows():
                session.run("""
                    MERGE (m:Movie {movieId: $movieId})
                    SET m.title = $title, m.genres = $genres
                """, movieId=int(row['movieId']), 
                     title=row['title'], 
                     genres=row['genres'])
                
                # Update progress every 100 movies
                if idx > 0 and idx % 100 == 0 and progress_callback:
                    progress = 10 + int((idx / len(movies_df)) * 20)  # 10-30%
                    progress_callback(f"Loaded {idx:,} / {len(movies_df):,} movies...", progress)
        
        if progress_callback:
            progress_callback(f"✓ Loaded {len(movies_df):,} movies", 30)
        
        # Create Genre nodes and relationships
        if progress_callback:
            progress_callback("Creating genre relationships...", 35)
        
        with driver.session() as session:
            session.run("""
                MATCH (m:Movie)
                WHERE m.genres IS NOT NULL
                WITH m, split(m.genres, '|') AS genreList
                UNWIND genreList AS genreName
                WITH m, trim(genreName) AS genreName
                WHERE genreName <> '' AND genreName <> '(no genres listed)'
                MERGE (g:Genre {name: genreName})
                MERGE (m)-[:HAS_GENRE]->(g)
            """)
        
        if progress_callback:
            progress_callback("✓ Genre relationships created", 40)
        
        # Load ratings in batches
        if progress_callback:
            progress_callback("Loading ratings...", 45)
        
        batch_size = 1000
        total_batches = (len(ratings_df) + batch_size - 1) // batch_size
        
        for i in range(0, len(ratings_df), batch_size):
            batch = ratings_df.iloc[i:i+batch_size]
            
            with driver.session() as session:
                session.run("""
                    UNWIND $ratings AS rating
                    MERGE (u:User {userId: rating.userId})
                    MERGE (m:Movie {movieId: rating.movieId})
                    MERGE (u)-[r:RATED]->(m)
                    SET r.rating = rating.rating, r.timestamp = rating.timestamp
                """, ratings=[row.to_dict() for _, row in batch.iterrows()])
            
            # Update progress
            if progress_callback:
                batch_num = (i // batch_size) + 1
                progress = 45 + int((batch_num / total_batches) * 50)  # 45-95%
                progress_callback(f"Loaded {min(i + batch_size, len(ratings_df)):,} / {len(ratings_df):,} ratings...", progress)
        
        if progress_callback:
            progress_callback(f"✓ Loaded {len(ratings_df):,} ratings", 95)
        
        # Create indexes
        if progress_callback:
            progress_callback("Creating indexes...", 97)
        
        with driver.session() as session:
            try:
                session.run("CREATE INDEX movie_title IF NOT EXISTS FOR (m:Movie) ON (m.title)")
                session.run("CREATE INDEX genre_name IF NOT EXISTS FOR (g:Genre) ON (g.name)")
            except:
                pass
        
        if progress_callback:
            progress_callback("✓ Data loading complete!", 100)
        
        return True, None
        
    except Exception as e:
        return False, str(e)
        

def clean_title(title):
    """Normalize titles"""
    if not isinstance(title, str):
        return ""
    title = re.sub(r'\s*\(\d{4}\)$', '', title)
    return title.lower().strip()

def extract_genres(genres_str):
    """Extract genre list from pipe-separated string"""
    if not isinstance(genres_str, str):
        return []
    genres = [g.strip() for g in genres_str.split('|') 
             if g.strip() and g.strip() != '(no genres listed)']
    return genres

def get_user_genre_profile(driver, user_id, min_rating=4.0):
    """Get genres the user likes"""
    query = """
    MATCH (u:User {userId: $user_id})-[r:RATED]->(m:Movie)
    WHERE r.rating >= $min_rating AND m.genres IS NOT NULL
    RETURN m.genres AS genres
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id, min_rating=min_rating)
        all_genres = []
        for record in result:
            all_genres.extend(extract_genres(record['genres']))
        
        # Return as set of unique genres
        return set(all_genres)

# ============================================================
# CONTENT-BASED METRICS
# ============================================================

def evaluate_content_based(recommendations, selected_movie_genres, k=10):
    """
    Evaluate content-based recommendations
    Measures: similarity quality, diversity
    """
    recs = recommendations.head(k) if len(recommendations) > k else recommendations
    
    # 1. Average Similarity Score
    if 'cosine_similarity' in recs.columns:
        avg_similarity = recs['cosine_similarity'].mean()
        min_similarity = recs['cosine_similarity'].min()
        max_similarity = recs['cosine_similarity'].max()
    else:
        avg_similarity = min_similarity = max_similarity = 0
    
    # 2. Genre Diversity
    all_genres = []
    for genres_str in recs['genres']:
        all_genres.extend(extract_genres(genres_str))
    
    unique_genres = len(set(all_genres))
    total_genre_mentions = len(all_genres)
    genre_diversity = unique_genres / total_genre_mentions if total_genre_mentions > 0 else 0
    
    # 3. Genre Distribution (how concentrated?)
    genre_counts = Counter(all_genres)
    if genre_counts:
        max_count = max(genre_counts.values())
        genre_concentration = max_count / total_genre_mentions
    else:
        genre_concentration = 0
    
    # 4. Shared Genres with Selected Movie
    selected_genres = set(extract_genres(selected_movie_genres))
    shared_genre_counts = []
    for genres_str in recs['genres']:
        rec_genres = set(extract_genres(genres_str))
        shared = len(selected_genres & rec_genres)
        shared_genre_counts.append(shared)
    
    avg_shared_genres = np.mean(shared_genre_counts) if shared_genre_counts else 0
    
    return {
        'avg_similarity': round(avg_similarity, 4),
        'min_similarity': round(min_similarity, 4),
        'max_similarity': round(max_similarity, 4),
        'unique_genres': unique_genres,
        'genre_diversity': round(genre_diversity, 3),
        'genre_concentration': round(genre_concentration, 3),
        'avg_shared_genres': round(avg_shared_genres, 2),
        'recommendations_evaluated': len(recs)
    }

# ============================================================
# COLLABORATIVE METRICS
# ============================================================

def evaluate_collaborative(recommendations, driver, user_id, k=10):
    """
    Evaluate collaborative filtering recommendations
    Measures: personalization, novelty, discovery
    """
    recs = recommendations.head(k) if len(recommendations) > k else recommendations
    
    # 1. Get user's genre preferences
    user_genres = get_user_genre_profile(driver, user_id)
    
    # 2. Get recommended genres
    rec_genres_list = []
    for genres_str in recs['genres']:
        rec_genres_list.extend(extract_genres(genres_str))
    rec_genres = set(rec_genres_list)
    
    # 3. Genre Alignment (Jaccard similarity)
    if user_genres and rec_genres:
        genre_overlap = len(user_genres & rec_genres)
        genre_union = len(user_genres | rec_genres)
        genre_alignment = genre_overlap / genre_union
    else:
        genre_alignment = 0
    
    # 4. Novelty: How many are NEW genres for user?
    new_genres = rec_genres - user_genres
    novelty_score = len(new_genres) / len(rec_genres) if rec_genres else 0
    
    # 5. Genre Diversity of recommendations
    unique_genres = len(set(rec_genres_list))
    genre_diversity = unique_genres / len(rec_genres_list) if rec_genres_list else 0
    
    # 6. Serendipity: Unexpected + Potentially Relevant
    # Movies with user's genres + new genres = serendipitous
    serendipity_count = 0
    for genres_str in recs['genres']:
        g = set(extract_genres(genres_str))
        has_familiar = bool(g & user_genres)  # Has genres user likes
        has_novel = bool(g - user_genres)     # Also has new genres
        if has_familiar and has_novel:
            serendipity_count += 1
    
    serendipity = serendipity_count / len(recs) if len(recs) > 0 else 0
    
    # 7. Average Similarity Score (if available)
    if 'cf_similarity' in recs.columns:
        avg_similarity = recs['cf_similarity'].mean()
    else:
        avg_similarity = 0
    
    return {
        'avg_similarity': round(avg_similarity, 4),
        'genre_alignment': round(genre_alignment, 3),
        'novelty': round(novelty_score, 3),
        'serendipity': round(serendipity, 3),
        'genre_diversity': round(genre_diversity, 3),
        'user_genres_count': len(user_genres),
        'rec_unique_genres': unique_genres,
        'new_genres_introduced': len(new_genres),
        'recommendations_evaluated': len(recs)
    }

# ============================================================
# HYBRID METRICS
# ============================================================

# def evaluate_hybrid(recommendations, driver, user_id, selected_movie_genres, alpha, beta, k=10):
#     """
#     Evaluate hybrid recommendations
#     Combines content + collaborative metrics
#     # for weigthed system
#     """
#     recs = recommendations.head(k) if len(recommendations) > k else recommendations
    
#     # Get both sets of metrics
#     content_metrics = evaluate_content_based(recs, selected_movie_genres, k)
#     collab_metrics = evaluate_collaborative(recs, driver, user_id, k)
    
#     # Calculate balance score
#     # Good hybrid should score well on both dimensions
#     if 'avg_similarity' in content_metrics and 'genre_alignment' in collab_metrics:
#         content_score = content_metrics['avg_similarity']
#         personalization_score = collab_metrics['genre_alignment']
        
#         # Weighted combination based on alpha/beta
#         balance_score = alpha * content_score + beta * personalization_score
#     else:
#         balance_score = 0
    
#     return {
#         **content_metrics,
#         **{f'collab_{k}': v for k, v in collab_metrics.items()},
#         'balance_score': round(balance_score, 4),
#         'alpha_weight': alpha,
#         'beta_weight': beta
#     }

def evaluate_hybrid(recommendations, driver, user_id, selected_movie_genres, alpha, beta, k=10):
    """
    Evaluate hybrid recommendations
    --------------------------------
    Combines content + collaborative metrics for the mixed hybrid model.
    In this setup, α and β represent proportions (not score weights),
    but we still use them to compute an interpretive balance score.
    """

    # Use top-k recommendations
    recs = recommendations.head(k) if len(recommendations) > k else recommendations

    # Compute metrics for each component
    content_metrics = evaluate_content_based(recs[recs["source"] == "content"], selected_movie_genres, k)
    collab_metrics  = evaluate_collaborative(recs[recs["source"] == "collaborative"], driver, user_id, k)

    # Compute balance score only if both metrics are available
    if "avg_similarity" in content_metrics and "genre_alignment" in collab_metrics:
        content_score        = content_metrics["avg_similarity"]
        personalization_score = collab_metrics["genre_alignment"]

        # Interpretive balance (not a fused similarity)
        balance_score = alpha * content_score + beta * personalization_score
        # balance_score = (content_score + personalization_score)/2
    else:
        balance_score = 0.0

    # Return unified metrics dictionary
    return {
        **content_metrics,
        **{f"collab_{key}": val for key, val in collab_metrics.items()},
        "balance_score": round(balance_score, 4),
        "alpha_weight": alpha,
        "beta_weight": beta,
        "n_content": len(recs[recs["source"] == "content"]),
        "n_cf": len(recs[recs["source"] == "collaborative"])
    }

# ============================================================
# DISPLAY FUNCTIONS FOR STREAMLIT
# ============================================================

def display_content_based_metrics(metrics, st):
    """Display content-based metrics in Streamlit"""
    st.markdown("### 📊 Content-Based Evaluation")
    st.info("**Goal:** Find movies similar to your selected movie")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Similarity", f"{metrics['avg_similarity']:.3f}")
        st.caption("Cosine similarity to anchor")
        if metrics['avg_similarity'] >= 0.7:
            st.success("✅ High similarity")
        elif metrics['avg_similarity'] >= 0.5:
            st.info("ℹ️ Moderate similarity")
        else:
            st.warning("⚠️ Low similarity")
    
    with col2:
        st.metric("Genre Diversity", f"{metrics['genre_diversity']:.1%}")
        st.caption("Variety of genres")
        if metrics['genre_diversity'] >= 0.6:
            st.success("✅ Good variety")
        elif metrics['genre_diversity'] >= 0.4:
            st.info("ℹ️ Moderate variety")
        else:
            st.warning("⚠️ Too similar")
    
    with col3:
        st.metric("Shared Genres", f"{metrics['avg_shared_genres']:.1f}")
        st.caption("Avg genres in common")
    
    with st.expander("📈 Detailed Metrics"):
        st.write(f"**Similarity Range:** {metrics['min_similarity']:.3f} - {metrics['max_similarity']:.3f}")
        st.write(f"**Unique Genres:** {metrics['unique_genres']}")
        st.write(f"**Genre Concentration:** {metrics['genre_concentration']:.1%} (lower = more diverse)")
        st.write(f"**Recommendations Evaluated:** {metrics['recommendations_evaluated']}")

def display_collaborative_metrics(metrics, st):
    """Display collaborative filtering metrics in Streamlit"""
    st.markdown("### 📊 Collaborative Filtering Evaluation")
    st.info("**Goal:** Recommend what you'll like based on similar users")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Genre Alignment", f"{metrics['genre_alignment']:.1%}")
        st.caption("Match with your taste")
        if metrics['genre_alignment'] >= 0.6:
            st.success("✅ Strong match")
        elif metrics['genre_alignment'] >= 0.3:
            st.info("ℹ️ Moderate match")
        else:
            st.warning("⚠️ Weak match")
    
    with col2:
        st.metric("Novelty", f"{metrics['novelty']:.1%}")
        st.caption("% new genres for you")
        if metrics['novelty'] >= 0.3:
            st.success("✅ Good discovery")
        elif metrics['novelty'] >= 0.1:
            st.info("ℹ️ Some discovery")
        else:
            st.warning("⚠️ Too safe")
    
    with col3:
        st.metric("Serendipity", f"{metrics['serendipity']:.1%}")
        st.caption("Familiar + Surprising")
        if metrics['serendipity'] >= 0.4:
            st.success("✅ Great balance")
        else:
            st.info("ℹ️ Could explore more")
    
    with col4:
        st.metric("Diversity", f"{metrics['genre_diversity']:.1%}")
        st.caption("Recommendation variety")
    
    with st.expander("📈 Detailed Metrics"):
        st.write(f"**Your Genre Profile:** {metrics['user_genres_count']} genres")
        st.write(f"**Recommended Genres:** {metrics['rec_unique_genres']} unique genres")
        st.write(f"**New Genres Introduced:** {metrics['new_genres_introduced']}")
        st.write(f"**Average Similarity:** {metrics['avg_similarity']:.3f}")
        st.write(f"**Recommendations Evaluated:** {metrics['recommendations_evaluated']}")

def display_hybrid_metrics(metrics, st):
    """Display hybrid metrics in Streamlit"""
    st.markdown("### 📊 Hybrid Evaluation")
    st.info("**Goal:** Balance content similarity + personalization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Balance Score", f"{metrics['balance_score']:.3f}")
        st.caption(f"α={metrics['alpha_weight']:.2f}, β={metrics['beta_weight']:.2f}")
    
    with col2:
        st.metric("Content Quality", f"{metrics['avg_similarity']:.3f}")
        st.caption("Similarity to anchor")
    
    with col3:
        st.metric("Personalization", f"{metrics['collab_genre_alignment']:.1%}")
        st.caption("Match with your taste - User Rating- C")
    
    # Show both dimensions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Content Dimension")
        st.write(f"**Genre Diversity:** {metrics['genre_diversity']:.1%}")
        st.write(f"**Shared Genres:** {metrics['avg_shared_genres']:.1f}")
    
    with col2:
        st.markdown("#### 👤 Personalization Dimension")
        st.write(f"**Novelty:** {metrics['collab_novelty']:.1%}")
        st.write(f"**Serendipity:** {metrics['collab_serendipity']:.1%}")
    
    if metrics['balance_score'] >= 0.5:
        st.success("✅ Well-balanced hybrid recommendations!")
    elif metrics['balance_score'] >= 0.3:
        st.info("ℹ️ Decent balance. Try adjusting α/β weights.")
    else:
        st.warning("⚠️ Imbalanced. Increase weight for weaker dimension.")

# ============================================================
# INTEGRATION EXAMPLES
# ============================================================

"""
HOW TO USE IN YOUR APP:

# Content-Based Tab
if st.button("Find Similar"):
    df = recommend_content_ann(movie_sel, k)
    st.dataframe(df)
    
    # Get selected movie genres
    selected_genres = st.session_state.movie_df[
        st.session_state.movie_df['title'] == movie_sel
    ].iloc[0]['genres']
    
    # Evaluate
    metrics = evaluate_content_based(df, selected_genres, k)
    display_content_based_metrics(metrics, st)

# Collaborative Tab
if st.button("Recommend"):
    df = recommend_cf_ann(user_id, k)
    st.dataframe(df)
    
    # Evaluate
    metrics = evaluate_collaborative(df, st.session_state.driver, user_id, k)
    display_collaborative_metrics(metrics, st)

# Hybrid Tab
if st.button("Hybrid Recommend"):
    df = recommend_hybrid_ann(user_id, anchor, k, alpha=alpha, beta=beta)
    st.dataframe(df)
    
    # Get anchor genres
    anchor_genres = st.session_state.movie_df[
        st.session_state.movie_df['title'] == anchor
    ].iloc[0]['genres']
    
    # Evaluate
    metrics = evaluate_hybrid(df, st.session_state.driver, user_id, anchor_genres, alpha, beta, k)
    display_hybrid_metrics(metrics, st)
"""
