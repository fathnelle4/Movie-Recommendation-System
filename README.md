# Movie-Recommendation-System

## Description
This project aims to build a **film recommendation system** based on the **MovieLens** dataset.  
Two main approaches are implemented:

-  **Content-Based Filtering**: recommends similar films based on their characteristics (genres, actors, etc.)  
-  **Collaborative Filtering**: recommends films enjoyed by users with similar tastes

The objective is to combine **relational analysis** with the power of the **Neo4j graph**, while using Python for data preparation, exploration and evaluation.

## Project Architecture 
Movie-Recommendation-System/
│
├── data/ # Fichiers CSV (movies.csv, ratings.csv, etc.)
├── notebooks/ # Notebooks Jupyter : EDA, Neo4j, Recommandations
├── src/ # Scripts Python pour le chargement et les requêtes
├── neo4j/ # Fichiers Cypher (.cql) pour la création du graphe
├── README.md # Documentation principale du projet
└── requirements.txt # Librairies nécessaires


---

## Installation

### Prerequisites

- **Python ≥ 3.8**
- **Neo4j Desktop** (ou Neo4j Aura)
- Librairies Python suivantes :
  ```bash
  pip install pandas numpy seaborn matplotlib neo4j scikit-learn

### Utilisation
Cloner le projet
git clone https://github.com/fathnelle4/Movie-Recommendation-System.git
cd Movie-Recommendation-System
