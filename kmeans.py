#Lib
import pandas as pd
from joblib import load
#Clustering
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
#Zips
#elbow point
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
# plots 
import plotly.express as px
import matplotlib.pyplot as plt
#Save plots as .png


#Clustering function
def mainClustering(dataframe, clusters_number):
    dataframe.reset_index
    #k-means algorithm
    kmeans = KMeans(
            n_clusters=clusters_number, #number of clusters for general dataset
            init="k-means++",
            n_init=10,
            tol=1e-04, 
            random_state=42
        )
    #fitting the algorithm
    kmeans.fit(dataframe)
    #labeling
    clusters=pd.DataFrame(dataframe)
    clusters['label']=kmeans.labels_
    return clusters

#Getting number of k-means clusters
def elbow(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    X=scaler.transform(dataset)
    inertia = []
    max_clusters = dataset.shape[1] #number of variables
    cluster_number = list(range(1, max_clusters))
    for i in cluster_number:
        kmeans = KMeans(
            n_clusters=i, init="k-means++",
            n_init=10,
            tol=1e-04, random_state=42
        )
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    #locating elbow point
    kneedle = KneeLocator(cluster_number, inertia, S=1.0, curve="convex", direction="decreasing")
    clusters_number = kneedle.knee
    if (cluster_number != int):
        kneedle = KneeLocator(cluster_number, inertia, S=0.0, curve="convex", direction="decreasing")
        clusters_number = kneedle.knee
    return clusters_number

def clustering(target, file_name, transformed_csv):


    print("transformed: ", transformed_csv.columns)
    name = file_name
    target_value = 1.0

    #Reading transformed data
    #checar qp con el archivo
    df = transformed_csv

    #Preparing dataset for clustering
    y = df[target]
    x = df.drop([target], axis=1)
    
    clusters_number = elbow(x)

    #Transformed dataframe labeled 
    df = pd.DataFrame(mainClustering(x, clusters_number))
    #Main clustering polar graph
    polar=df.groupby("label").mean().reset_index()
    polar=pd.melt(polar,id_vars=["label"])
    fig0 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
    fig0.show()
    fig0.write_image(f'./results_clustering/main_cluster_img.png')
    #Main clustering pie plot to see clustering distribution
    pie0=df.groupby('label').size().reset_index()
    pie0.columns=['label','value']
    pie0 = px.pie(pie0,values='value',names='label')
    pie0.show()
    pie0.write_image(f'./results_clustering/main_cluster_distribution.png')

    df.to_csv(f'./results_clustering/main_cluster.csv') #creating filtered cvs 

    #Create a csv focusing in cluster churns
    clust = 0 #iterator
    df[target] = y
    while clust < clusters_number:
        dataframe = pd.DataFrame(df.loc[df['label'] == clust]) #Filter by cluster
        dataframe = dataframe.loc[dataframe[target] == target_value] #Filter by positive targets
        dataframe = dataframe.drop(columns=[target,'label'])
        dataframe.to_csv(f'./results_clustering/cluster{clust}.csv') #creating filtered cvs 
        clust = clust + 1 #iterator

    clust = 0 #iterator
    while clust < clusters_number:
        sub_cluster = pd.read_csv(f'./results_clustering/cluster{clust}.csv') #read every subcluster .csv
        sub_cluster = sub_cluster.drop(columns=['Unnamed: 0']) #drop the added column
        if sub_cluster.shape[0] == 0:
            clust = clust + 1
        else:
            sub_clusters_number = elbow(sub_cluster) #calculate elbow point for every sub-cluster
            clusters = pd.DataFrame(mainClustering(sub_cluster, sub_clusters_number)) #k-means clustering method
            #polar sub-clusters graph
            sub_polar=clusters.groupby("label").mean().reset_index() 
            sub_polar=pd.melt(sub_polar,id_vars=["label"])
            fig = px.line_polar(sub_polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
            fig.show() #print here
            fig.write_image(f'./results_clustering/cluster{clust}img.png') #save as .png file
            #Pie plot to see sub-cluster's distribution
            pie=clusters.groupby('label').size().reset_index()
            pie.columns=['label','value']
            pie = px.pie(pie,values='value',names='label')
            pie.show() #print here
            pie.write_image(f'./results_clustering/cluster{clust}distribution.png') #save as .png file
            clust = clust + 1
    