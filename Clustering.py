import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Load data
df = pd.read_csv('data.csv')
df = df[df['Entity'] != 'Algeria']  # Remove Algeria because it has missing values

# Group by country and calculate the sum of Cases, Deaths and Daily tests
df_grouped = df.groupby('Entity').agg({'Cases': 'sum', 'Deaths': 'sum', 'Daily tests': 'sum', 'Population': 'first', 
                                        'GDP/Capita': 'first', 'Hospital beds per 1000 people': 'first', 'Medical doctors per 1000 people': 'first'}).reset_index()

# Preprocessing: create the features we're interested in
df_grouped['Cases per Capita'] = df_grouped['Cases'] / df_grouped['Population']
df_grouped['Deaths per Capita'] = df_grouped['Deaths'] / df_grouped['Population']
df_grouped['Tests per Capita'] = df_grouped['Daily tests'] / df_grouped['Population']
df_grouped['Positivity Rate'] = df_grouped['Cases'] / df_grouped['Daily tests']
df_grouped['Death Rate'] = df_grouped['Deaths'] / df_grouped['Cases']


# Select features and drop rows with missing values
features = ['Cases per Capita', 'Deaths per Capita', 'Tests per Capita', 'Positivity Rate', 'Death Rate', 'GDP/Capita', 'Hospital beds per 1000 people', 'Medical doctors per 1000 people', 'Positivity Rate', 'Death Rate']
df_cluster = df_grouped.dropna(subset=features)

# Normalize the features
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster[features])

# # Determine the optimal number of clusters using silhouette score
# silhouette_scores = []
# for n_cluster in range(2, 10):
#     kmeans = KMeans(n_clusters=n_cluster)
#     preds = kmeans.fit_predict(df_cluster_scaled)
#     silhouette_scores.append(silhouette_score(df_cluster_scaled, preds))

# # Plot silhouette scores to find the 'elbow'
# plt.plot(range(2, 10), silhouette_scores, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Silhouette Score')
# plt.show()

# Perform KMeans clustering using the optimal number of clusters
# kmeans = KMeans(n_clusters=np.argmax(silhouette_scores)+2)  # +2 because we started from 2 clusters
kmeans = KMeans(n_clusters=4)  # +2 because we started from 2 clusters
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

df_cluster.round(2).to_csv('clustered_countries.csv', index=False)

# Analyze the resulting clusters
for cluster in set(df_cluster['Cluster']):
    print(f'\nCluster {cluster} stats:')
    print(df_cluster[df_cluster['Cluster'] == cluster][features].describe().round(2))
    print(f'\nCluster {cluster} countries:')
    countries = df_cluster[df_cluster['Cluster'] == cluster]['Entity'].values
    print(countries)

# Select features and display top and bottom countries for each feature
features = ['Cases per Capita', 'Deaths per Capita', 'Tests per Capita', 'Positivity Rate', 'Death Rate', 'GDP/Capita', 'Hospital beds per 1000 people', 'Medical doctors per 1000 people']
for feature in features:
    sorted_df = df_grouped.sort_values(by=feature, ascending=False)
    print(f"\nCategory: {feature}")
    print("Top 3 Countries:")
    print(sorted_df[['Entity', feature]].head(3))
    
    print("Bottom 3 Countries:")
    print(sorted_df[['Entity', feature]].tail(3))
