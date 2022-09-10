import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import SimpSOM as sps

// cmd -> cd ..
// cmd -> cd C:\Users\alper\AppData\Local\Programs\Python\Python36-32
// cmd -> pip install sklearn
// cmd -> pip install SimpSOM

def soru1(): 
    veri = pd.read_csv("csv/mall_customers.csv")
    X = veri.drop(["CustomerID","Genre"],axis=1)
    net = sps.somNet(13,13,X.values,PBC=True)
    net.train(0.01,10000)
    hrt=np.array((net.project(X.values)))
    kmeans = KMeans(n_clusters=2, max_iter=200, random_state =0) # Zengin Müşteri, Fakir Müşteri Ayrımı Yaptım
    y_kmeans = kmeans.fit_predict(hrt)
    veri["kumeler"] = kmeans.labels_
    print(veri[veri["kumeler"]==0].head(5))
    print(veri[veri["kumeler"]==1].head(5))

def soru2(): # Kullanım Yoğunluğuna Göre düşük orta yüksek Ayrımı Yaptım
    veri = pd.read_csv("csv/cc_general.csv")
    X = veri.drop(["CUST_ID"],axis=1)
    net = sps.somNet(34,34,X.values,PBC=True)
    net.train(0.01,5000)
    hrt=np.array((net.project(X.values)))
    kmeans = KMeans(n_clusters=3, max_iter=200, random_state =0)
    y_kmeans = kmeans.fit_predict(hrt)
    veri["kumeler"] = kmeans.labels_
    print(veri[veri["kumeler"]==0].head(5))
    print(veri[veri["kumeler"]==1].head(5))
    print(veri[veri["kumeler"]==2].head(5))

