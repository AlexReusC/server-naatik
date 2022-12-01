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
    print("DF2_ ", dataframe.shape)
    print("N CLUST_ ", clusters_number)
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
    clusters['big_group']=kmeans.labels_
    return clusters

#Getting number of k-means clusters
def elbow(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    X=scaler.transform(dataset)
    inertia = []
    max_clusters = dataset.shape[1] #number of variables
    clusters_number = list(range(1, max_clusters))
    for i in clusters_number:
        kmeans = KMeans(
            n_clusters=i, init="k-means++",
            n_init=10,
            tol=1e-04, random_state=42
        )
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    #locating elbow point
    kneedle = KneeLocator(clusters_number, inertia, S=1.0, curve="convex", direction="decreasing")
    cluster_number = kneedle.knee
    if (cluster_number != int):
        kneedle = KneeLocator(clusters_number, inertia, S=0.0, curve="convex", direction="decreasing")
        cluster_number = kneedle.knee
        if cluster_number == None:
            cluster_number = 1
    return cluster_number

def clustering(target, file_name, transformed_csv, ui):


    print("columnas en kmeans: ", transformed_csv.columns)
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
    print("SHAPE: ", df.shape)
    df = pd.DataFrame(mainClustering(x, clusters_number))
    #Main clustering polar graph
    polar=df.groupby("big_group").mean().reset_index()
    polar=pd.melt(polar,id_vars=["big_group"])
    fig0 = px.line_polar(polar, r="value", theta="variable", color="big_group", line_close=True,height=800,width=1400)
    #fig0.show()
    fig0.write_image(f'./static/results_clustering/{ui}/main_cluster_img.png')
    #Main clustering pie plot to see clustering distribution
    pie0=df.groupby('big_group').size().reset_index()
    pie0.columns=['big_group','value']
    pie0 = px.pie(pie0,values='value',names='big_group')
    #pie0.show()
    pie0.write_image(f'./static/results_clustering/{ui}/main_cluster_distribution.png')

    df.to_csv(f'./static/results_clustering/{ui}/main_cluster.csv') #creating filtered cvs 

    #Create a csv focusing in cluster churns
    clust = 0 #iterator
    df[target] = y
    while clust < clusters_number:
        dataframe = pd.DataFrame(df.loc[df['big_group'] == clust]) #Filter by cluster
        dataframe = dataframe.loc[dataframe[target] == target_value] #Filter by positive targets
        dataframe = dataframe.drop(columns=[target,'big_group'])
        dataframe.to_csv(f'./static/results_clustering/{ui}/cluster{clust}.csv') #creating filtered cvs 
        clust = clust + 1 #iterator

    clust = 0 #iterator
    while clust < clusters_number:
        sub_cluster = pd.read_csv(f'./static/results_clustering/{ui}/cluster{clust}.csv') #read every subcluster .csv
        if sub_cluster.shape[0] == 0:
            sub_cluster = sub_cluster.drop(columns=['Unnamed: 0']) #drop the added column
            clust = clust + 1
        else:
            sub_cluster = sub_cluster.drop(columns=['Unnamed: 0']) #drop the added column
            sub_clusters_number = elbow(sub_cluster) #calculate elbow point for every sub-cluster
            print("subcluster: ", sub_clusters_number)
            clusters = pd.DataFrame(mainClustering(sub_cluster, sub_clusters_number)) #k-means clustering method
            #polar sub-clusters graph
            sub_polar=clusters.groupby("big_group").mean().reset_index() 
            sub_polar=pd.melt(sub_polar,id_vars=["big_group"])
            fig = px.line_polar(sub_polar, r="value", theta="variable", color="big_group", line_close=True,height=800,width=1400)
            #fig.show() #print here
            fig.write_image(f'./static/results_clustering/{ui}/cluster{clust}img.png') #save as .png file
            #Pie plot to see sub-cluster's distribution
            pie=clusters.groupby('big_group').size().reset_index()
            pie.columns=['big_group','value']
            pie = px.pie(pie,values='value',names='big_group')
            #pie.show() #print here
            pie.write_image(f'./static/results_clustering/{ui}/cluster{clust}distribution.png') #save as .png file
            clust = clust + 1
    