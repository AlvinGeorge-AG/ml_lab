import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline


df = pd.read_csv("./digits.csv")
X = df.drop(columns=["target"])

k_values = [2,3,4,5,6,7,8,9]
results = {}

for k in k_values:
        model = make_pipeline(
                StandardScaler(),
                KMeans(n_clusters=k,random_state=42)
        )
        model.fit(X)
        X_scaled = model.named_steps["standardscaler"].transform(X)
        cluster_labels = model.named_steps["kmeans"].labels_
        results[k] = {
                "inertia":model.named_steps["kmeans"].inertia_,
                "cluster_labels":cluster_labels,
                "silhouette_score":silhouette_score(X_scaled,cluster_labels)
        }

for k,d in results.items():
        print(f"K = {k}")
        print(f"cluster_labels : {d['cluster_labels']}")
        print(f"inertia : {d['inertia']}")
        print(f"silhouette_score : {d['silhouette_score']}")

fig,axes = plt.subplots(1,2,figsize=(20,10))
axes = axes.flatten()

axes[0].plot(k_values,[d['inertia'] for _,d in results.items()])
axes[0].set_xlabel("K Values")
axes[0].set_ylabel("Inertia")

axes[1].plot(k_values,[d['silhouette_score'] for _,d in results.items()])
axes[1].set_xlabel("K Values")
axes[1].set_ylabel("silhouette_score")

plt.tight_layout()
plt.show()
