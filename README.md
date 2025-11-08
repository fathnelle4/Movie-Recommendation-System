# 🎬 Movie Recommendation system
**Fast, Scalable, and Interpretable Movie Recommendation System using Neo4j + HNSW (Approximate Nearest Neighbors) on the Movielens dataset**  

---

## 1-Overview
This project aims to build a **film recommendation system** based on the **MovieLens** dataset.  To achieve our goal we use a method called `Approximate Nearest Neighbors(ANN)`

The **ANN Movie Recommender** is a recommendation engine that combines **content-based**, **collaborative**, and **hybrid filtering** approaches using **Approximate Nearest Neighbor (ANN)** search with **HNSW (Hierarchical Navigable Small World)** graphs.

Our application runs as an interactive **Streamlit app**, powered by **Neo4j** for data storage and **HNSWlib** for fast vector similarity search.  
This framework makes it easy to explore, visualize, and understand how movie recommendations are computed based on genre similarity, user ratings, and hybrid weighting.

---

## 2-Key Features
- **Content-Based Filtering** -> Recommends movies with similar *features or genres*.  
- **Collaborative Filtering** -> Suggests movies liked by *similar users* (based on ratings).  
- **Hybrid Filtering (Mixed)** -> Merges both models proportionally using weights ``` alpha ```and ``` beta ```.  
- **Interactive Evaluation Metrics**: Balance Score, Content Quality, Personalization, Novelty, and Serendipity.  
- **Neo4j Integration** for graph-based querying of users, movies, and relationships.  

---

## 3-Architecture

<img src="assets/architecture.png" alt="System Architecture" width="300"/>



---

## 4-Recommendation Methods
### Content-Based (Cosine Similarity)
Each movie (item) is represented as a vector based on content features such as genres 

Similarity is computed as:

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \, \|\mathbf{y}\|}
$$

---

### 5-Collaborative Filtering (Item-Item ANN)
Each movie (item) is represented as a vector based on **user ratings** rather than content features.  
The goal is to find movies that tend to be liked by the *same group of users*.

**a.** Each item vector is constructed from user ratings:

$$
v_i[u] = r(u,i) - \bar{r_i}
$$

- \(r(u,i)\): rating of movie *i* by user *u*.  
- \(\bar{r_i}\): average rating of movie *i* across all users.  
- This centering step removes **popularity bias**, highlighting how much each user’s opinion deviates from the average.  
  Positive values mean the user liked it more than most people; negative means less.


**b.** User profile:

$$
\mathbf{p_u} = \frac{1}{|I_u|} \sum_{i \in I_u, r(u,i) > 4} v_i
$$

- \(I_u\): set of movies rated by user *u*.  
- \(r(u,i) > 4\): we only keep movies the user *really liked*.  
- \(|I_u|\): number of liked movies.  
- The resulting vector \(\mathbf{p_u}\) is the **average of all item vectors the user enjoyed**,  
  capturing their **personal taste profile**.
  

**Key Idea:**  
Collaborative filtering does not care about *what* the movie is about - 
it cares about **who** liked it.  
It finds patterns like:  
> `“Users who loved *Movie A* also loved *Movie B*.”`

---

### 6-Hybrid (Mixed)
A combined recommendation list using weighted sampling:

$$
s(m) = \alpha \cdot s_{content}(m) + \beta \cdot s_{cf}(m)
$$

---

## 7-Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **Balance Score** | Measures equilibrium between content similarity and personalization. |
| **Content Quality** | How close results are to the anchor movie (semantic similarity). |
| **Personalization** | How well results match the user’s high-rated preferences . |
| **Novelty** | Share of new or unseen movies for the user. |
| **Serendipity** | Mix of familiar yet surprising recommendations. |

---

## 8-Prerequisites

Before running the app, ensure you have:
1. **Neo4j Database** (Desktop, AuraDB, or Docker).  

2. **MovieLens Dataset** (`movies.csv` & `ratings.csv`).  

3. **Python 3.8+** installed.  

4. Required packages installed:  
   ```bash
   pip install streamlit neo4j hnswlib pandas numpy
   ```

5. **Valid Neo4j credentials** (`username`, `password`, `bolt://localhost:7687`).  

6. Access Neo4j Browser locally at [http://localhost:7474/](http://localhost:7474/).  

If you experience issues setting up Neo4j, refer to the official [Neo4j Docs](https://neo4j.com/docs/)


---



## 9-Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/fathnelle4/Movie-Recommendation-System.git 
   cd Movie-Recommendation-System
   ```
2. Launch the app:
   ```bash
   streamlit run Movie-Recommendation-System.py
   ```
3. Connect to your Neo4j database in the sidebar.

4. Choose a recommendation mode (Content, Collaborative, or Hybrid).

5. Explore metrics and visualize recommendation quality!

6. Browse through descriptive analyse

---



## 📚 References

  - Malkov, Y. & Yashunin, D. (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW*. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)  
  - [Movielens Dataset](https://grouplens.org/datasets/movielens/)  
  - [Cosine Similarity - Medium](https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1)  
  - [Content-Based Recommender - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-content-based-recommender-system/)  
  - [Approximate Nearest Neighbor Search - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/approximate-nearest-neighbor-ann-search/)  
  - [Collaborative Filtering - GeeksforGeeks](https://www.geeksforgeeks.org/collaborative-filtering-in-recommendation-systems/)  
  - [Naomy Gomes - Cosine Similarity and its use in Recommender Systems](https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1)  

  - [Neo4j Graph Data Science Documentation](https://neo4j.com/docs/graph-data-science/current/)
  - [FAISS - Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)  
  - [Annoy - Spotify Approximate Nearest Neighbors](https://github.com/spotify/annoy)  
  - [Voyager - Spotify HNSW-based ANN](https://github.com/spotify/voyager)  

---

## 👥 Authors

- **Glorie Metsa Wowo** - [LinkedIn](https://www.linkedin.com/in/glorie-wowo-data-science-edtech)  
- **Fathnelle Mehouelley** - [LinkedIn](https://www.linkedin.com/in/fathnelle-mehouelley/)

---

© 2025 ANN Movie Recommender - Powered by Neo4j, HNSWlib, and Streamlit
