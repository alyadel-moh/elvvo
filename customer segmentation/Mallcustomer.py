import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)','Spending Score (1-100)']]

scaler = StandardScaler()
x_Scaled = scaler.fit_transform(X)

wcss = []
for k in range(1,11):
    km = KMeans(n_clusters=k,random_state=42)
    km.fit(x_Scaled)
    wcss.append(km.inertia_)  #cost function

plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("cost function")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=5,random_state=42,n_init=10)
y_kmeans = kmeans.fit_predict(x_Scaled)

df['cluster'] = y_kmeans

plt.scatter(x_Scaled[:, 0], x_Scaled[:, 1], c=y_kmeans, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segments")
plt.show()

