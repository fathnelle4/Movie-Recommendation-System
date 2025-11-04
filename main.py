import streamlit as st
from neo4j  import GraphDatabase
import pandas as pd


# Configuration de la page
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Neo4j connection
@st.cache_resource
def get_neo4j_driver():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "JeanAlice#123"
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_neo4j_driver()

def get_all_movies():
    """Retrieve all films from the database"""
    with driver.session() as session:
        result = session.run("""
            MATCH (m:Movie)
            RETURN m.title AS title, m.movieId AS movieId, m.genres AS genres
            ORDER BY m.title
        """)
        return [record["title"] for record in result]

def get_all_users():
    """Retrieve all users"""
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User)
            RETURN u.userId AS userId
            ORDER BY u.userId
        """)
        return [record["userId"] for record in result]
    
def content_based (movie_title, limit = 10):
    """
    Recommend similar movies based on commons genders
    """
    
    with driver.session() as session:
        result = session.run("""
                              // step 1 : Find the reference movie
                              MATCH (m1:Movie {title: $title})
                              
                              // step 2 : Find his genders
                              MATCH (m1)-[:HAS_GENRE]->(g:Genre)
                              
                              // step 3 : Find others movies with the same genders
                              MATCH (m2:Movie)-[:HAS_GENRE]->(g)
                              
                              // step 4 : Exclude the original movie
                              WHERE m1 <> m2
                              
                              // step 5 : Count the genders they have in common and compute score
                              WITH m2, COUNT(DISTINCT g) AS common_genres, 
                                  COLLECT(DISTINCT g.name) AS shared_genres

                              // step 6 : Trier par similarité
                              RETURN m2.title AS recommended_movie, 
                                    m2.genres AS all_genres,
                                    common_genres AS similarity_score,
                                    shared_genres
                              ORDER BY similarity_score DESC
                              LIMIT $limit """, title = movie_title, limit = limit)
        recommendations = []
        for record in result:
            recommendations.append({
                'title':record['recommended_movie'], 
                'genres':record['all_genres'],
                'similarity_score': record['similarity_score'],
                'shared_genres': record['shared_genres']
            })
        return recommendations


def collaboratif_filtering (user_id, limit = 10):
    """
    Recommend movies based on similar users are liked
    """
    with driver.session() as session:
        result = session.run("""
            // Step 1 : Movies that the user has liked
            MATCH (u:User {userId: $userId})-[r1:RATED]->(m:Movie)
            WHERE r1.rating >= 4.0
            
            // Step 2 : Others users that liked the same movie
            MATCH (u2:User)-[r2:RATED]->(m)
            WHERE u2 <> u AND r2.rating >= 4.0
            
            // Step 3 : Movies those similar users liked
            MATCH (u2)-[r3:RATED]->(m2:Movie)
            WHERE r3.rating >= 4.0
            
            // Step 4 : Exclurde the movie already watched
            AND NOT EXISTS((u)-[:RATED]->(m2))
            
            // Step 5 : Computing of a recommendation score
            WITH m2, 
                 COUNT(DISTINCT u2) AS users_who_liked,
                 AVG(r3.rating) AS avg_rating
            
            // Step 6 : Score composite
            RETURN m2.title AS recommended_movie,
                   m2.genres AS genres,
                   users_who_liked,
                   ROUND(avg_rating, 2) AS average_rating,
                   (users_who_liked * avg_rating) AS recommendation_score
            ORDER BY recommendation_score DESC
            LIMIT $limit
                    """, userId = user_id, limit = limit)
        
        recommendations = []
        for record in result:
            recommendations.append({
                'title' : record['recommended_movie'],
                'genres': record['genres'],
                'users_who_liked' : record['users_who_liked'],
                'avg_rating' : record['average_rating'],
                'score' : record['recommendation_score']
            })
        return recommendations
    
def recommend_hybrid(user_id, movie_title=None, limit=10):
    """
    Combine both approach for best recommendations
    """
    with driver.session() as session:
        result = session.run("""
            // Step 1 : Films de l'utilisateur bien notés
            MATCH (u:User {userId: $userId})-[r:RATED]->(liked:Movie)
            WHERE r.rating >= 4.0
            
            // Step 2 : Genres de ces films
            MATCH (liked)-[:HAS_GENRE]->(g:Genre)
            
            // Step 3 : Similar movies by gender
            MATCH (recommended:Movie)-[:HAS_GENRE]->(g)
            WHERE NOT EXISTS((u)-[:RATED]->(recommended))
            
            // Step 4 : Others users that liked similar movies
            OPTIONAL MATCH (u2:User)-[r2:RATED]->(recommended)
            WHERE u2 <> u AND r2.rating >= 4.0
            
            // Step 5 : Computing of scores
            WITH recommended,
                 COUNT(DISTINCT g) AS genre_match,
                 COUNT(DISTINCT u2) AS collaborative_score,
                 AVG(r2.rating) AS avg_rating
            
            // Step 6 : Hybrid score
            RETURN recommended.title AS title,
                   recommended.genres AS genres,
                   genre_match,
                   collaborative_score,
                   COALESCE(avg_rating, 0) AS avg_rating,
                   (genre_match * 2 + collaborative_score + COALESCE(avg_rating, 0)) AS hybrid_score
            ORDER BY hybrid_score DESC
            LIMIT $limit
        """, userId=user_id, limit=limit)
        
        recommendations = []
        for record in result:
            recommendations.append({
                'title': record['title'],
                'genres': record['genres'],
                'genre_match': record['genre_match'],
                'collaborative_score': record['collaborative_score'],
                'avg_rating': record['avg_rating'],
                'hybrid_score': record['hybrid_score']
            })
        
        return recommendations
    

def get_user_ratings(user_id, limit=10):
    """Retrieves films rated by a user"""
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
            RETURN m.title AS title, 
                   m.genres AS genres,
                   r.rating AS rating
            ORDER BY r.rating DESC
            LIMIT $limit
        """, userId=user_id, limit=limit)
        
        ratings = []
        for record in result:
            ratings.append({
                'title': record['title'],
                'genres': record['genres'],
                'rating': record['rating']
            })
        return ratings
    
def get_database_stats():
    """Statistiques de la base de données"""
    with driver.session() as session:
        stats = {}
        
        result = session.run("MATCH (u:User) RETURN count(u) AS count")
        stats['users'] = result.single()["count"]
        
        result = session.run("MATCH (m:Movie) RETURN count(m) AS count")
        stats['movies'] = result.single()["count"]
        
        result = session.run("MATCH ()-[r:RATED]->() RETURN count(r) AS count")
        stats['ratings'] = result.single()["count"]
        
        result = session.run("MATCH (g:Genre) RETURN count(g) AS count")
        stats['genres'] = result.single()["count"]
        
        return stats
    



# ==================== UI ====================

# Titre principal
st.title("🎬 Movie Recommendation System")
st.markdown("---")

# Sidebar pour les statistiques
with st.sidebar:
    st.header("📊 Database Statistics")
    try:
        stats = get_database_stats()
        st.metric("Users", f"{stats['users']:,}")
        st.metric("Movies", f"{stats['movies']:,}")
        st.metric("Ratings", f"{stats['ratings']:,}")
        st.metric("Genres", f"{stats['genres']:,}")
    except Exception as e:
        st.error(f"Error loading stats: {e}")
    
    st.markdown("---")
    st.info("💡 **Tip**: Try different recommendation methods to compare results!")

# Tabs pour différentes fonctionnalités
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Content-Based", 
    "👥 Collaborative Filtering", 
    "🔀 Hybrid", 
    "📝 User Profile"
])

# ==================== TAB 1: Content-Based ====================
with tab1:
    st.header("🎯 Content-Based Recommendations")
    st.write("Find movies similar to a movie you like based on genres")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            movies = get_all_movies()
            selected_movie = st.selectbox(
                "Select a movie:",
                movies,
                key="content_movie"
            )
        except Exception as e:
            st.error(f"Error loading movies: {e}")
            selected_movie = None
    
    with col2:
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            key="content_slider"
        )
    
    if st.button("Get Recommendations", key="content_btn"):
        if selected_movie:
            with st.spinner("Finding similar movies..."):
                try:
                    recommendations = content_based(selected_movie, num_recommendations)
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        
                        df = pd.DataFrame(recommendations)
                        df.index = df.index + 1
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "title": st.column_config.TextColumn("Movie Title", width="large"),
                                "genres": st.column_config.TextColumn("Genres", width="medium"),
                                "similarity_score": st.column_config.NumberColumn("Similarity Score", format="%d"),
                                "shared_genres": st.column_config.TextColumn("Shared Genres", width="medium")
                            }
                        )
                    else:
                        st.warning("No recommendations found. Try another movie!")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== TAB 2: Collaborative ====================
with tab2:
    st.header("👥 Collaborative Filtering")
    st.write("Discover movies loved by users with similar tastes")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            users = get_all_users()
            selected_user = st.selectbox(
                "Select a user ID:",
                users,
                key="collab_user"
            )
        except Exception as e:
            st.error(f"Error loading users: {e}")
            selected_user = None
    
    with col2:
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            key="collab_slider"
        )
    
    if st.button("Get Recommendations", key="collab_btn"):
        if selected_user:
            with st.spinner("Analyzing user preferences..."):
                try:
                    recommendations = collaboratif_filtering(selected_user, num_recommendations)
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        
                        df = pd.DataFrame(recommendations)
                        df.index = df.index + 1
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "title": st.column_config.TextColumn("Movie Title", width="large"),
                                "genres": st.column_config.TextColumn("Genres", width="medium"),
                                "users_who_liked": st.column_config.NumberColumn("Users Who Liked", format="%d"),
                                "avg_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
                                "score": st.column_config.NumberColumn("Recommendation Score", format="%.1f")
                            }
                        )
                    else:
                        st.warning("No recommendations found. User might not have enough ratings!")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== TAB 3: Hybrid ====================
with tab3:
    st.header("🔀 Hybrid Recommendations")
    st.write("Best of both worlds: combining content and collaborative filtering")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            users = get_all_users()
            selected_user = st.selectbox(
                "Select a user ID:",
                users,
                key="hybrid_user"
            )
        except Exception as e:
            st.error(f"Error loading users: {e}")
            selected_user = None
    
    with col2:
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            key="hybrid_slider"
        )
    
    if st.button("Get Recommendations", key="hybrid_btn"):
        if selected_user:
            with st.spinner("Computing hybrid recommendations..."):
                try:
                    recommendations = recommend_hybrid(selected_user, num_recommendations)
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        
                        df = pd.DataFrame(recommendations)
                        df.index = df.index + 1
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "title": st.column_config.TextColumn("Movie Title", width="large"),
                                "genres": st.column_config.TextColumn("Genres", width="medium"),
                                "genre_match": st.column_config.NumberColumn("Genre Match", format="%d"),
                                "collaborative_score": st.column_config.NumberColumn("Collaborative Score", format="%d"),
                                "avg_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
                                "hybrid_score": st.column_config.NumberColumn("Hybrid Score", format="%.1f")
                            }
                        )
                    else:
                        st.warning("No recommendations found!")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== TAB 4: User Profile ====================
with tab4:
    st.header("📝 User Profile")
    st.write("Explore what a user has watched and rated")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            users = get_all_users()
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
    
    if st.button("View Profile", key="profile_btn"):
        if selected_user:
            with st.spinner("Loading user profile..."):
                try:
                    ratings = get_user_ratings(selected_user, num_ratings)
                    
                    if ratings:
                        st.success(f"User {selected_user} has rated {len(ratings)} movies (showing top {num_ratings})")
                        
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
                        
                        # Statistiques
                        avg_rating = sum([r['rating'] for r in ratings]) / len(ratings)
                        st.metric("Average Rating", f"⭐ {avg_rating:.2f}")
                    else:
                        st.warning("No ratings found for this user!")
                except Exception as e:
                    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Neo4j</p>
    </div>
    """,
    unsafe_allow_html=True
)