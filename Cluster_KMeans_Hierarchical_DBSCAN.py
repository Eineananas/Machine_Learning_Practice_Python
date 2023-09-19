from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese labels properly
plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs correctly

iris_data = datasets.load_iris()
iris = pd.DataFrame(iris_data.data, columns=["SpealLength", "SpealWidth", "PetalLength", "PetalWidth"])
target = iris_data.target

print(iris.isnull().sum())
iris.describe().T.to_csv(r'./describe.csv')

# Min-Max normalize
def normalization(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data

iris_norm = normalization(iris.values)

def plotScatter(x, y, title, savePath):
    plt.scatter(x, y)
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.title(title)
    plt.savefig(savePath)
    plt.close()

plotScatter(iris_norm[:,0], iris_norm[:, 1], title="Speal", savePath=r'./figures/speal.png')
plotScatter(iris_norm[:,2], iris_norm[:, 3], title="Petal", savePath=r'./figures/petal.png')

# choose n_cluster
def manhattan_distance(x, y):
    return np.sum(abs(x - y))

def chooseNCluster(data, tag):
    distance = []
    k = []
    for n_clusters in range(1, 16):
        cls = KMeans(n_clusters).fit(data)
        distance_sum = 0
        for i in range(n_clusters):
            group = cls.labels_ == i
            members = data[group, :]
            for v in members:
                distance_sum += manhattan_distance(np.array(v), cls.cluster_centers_[i])
        distance.append(distance_sum)
        k.append(n_clusters)
    plt.scatter(k, distance)
    plt.plot(k, distance)
    plt.xlabel("k")
    plt.ylabel("Loss")
    plt.title(f'{tag} KMeans Distance')
    plt.savefig(f'./figures/{tag}Dist.png')
    plt.close()

chooseNCluster(iris_norm, tag='iris')

# visualization
def plotRes(data, c, tag, method='Real'):
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x, y, c=c)
    plt.title(f'{tag} {method} Result')
    plt.legend(c)
    plt.savefig(f'./figures/{tag}_{method}_res.png')
    plt.close()

iris_tsne = TSNE(n_components=2).fit_transform(iris_norm)
rs= []
eps = np.arange(0.1, 0.5, 0.1)
min_samples = np.arange(2, 9, 1)
best_score = 0
best_score_eps = 0
best_score_min_samples = 0

for i in eps:
    for j in min_samples:
        try:
            db = DBSCAN(eps=i, min_samples=j).fit(iris_norm)
            labels = db.labels_
            k = metrics.silhouette_score(iris_norm, labels)
            raito = len(labels[labels[:] == -1]) / len(labels)
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            rs.append([i,j,k,raito,n_clusters_])
            if k > best_score:
                best_score = k
                best_score_eps = i
                best_score_min_samples = j
        except:
            db = ''
        else:
            db = ''

rs = pd.DataFrame(rs)
rs.columns=['eps','min_samples','score','raito','n_clusters']
sns.relplot(x="eps",y="min_samples", size='score',data=rs)
plt.savefig('./figures/score.png')
sns.relplot(x="eps",y="min_samples", size='raito',data=rs)
plt.savefig('./figures/ratio.png')

print(f"best_score_eps:{best_score_eps},best_score_min_samples:{best_score_min_samples}")

K = 3
kmeans_model = KMeans(n_clusters = K, max_iter = 1000)
agg_model = AgglomerativeClustering(n_clusters = K)
dbscan_model = DBSCAN(eps = 0.3, min_samples = 2)

methods = ["KMeans", "AgglomerativeClustering", "DBSCAN"]
models = [kmeans_model, agg_model, dbscan_model]
silhouettes = [metrics.silhouette_score(iris_norm, target)]

plotRes(iris_tsne, target, tag='iris')

for i in range(len(methods)):
    auc_tmp = []
    model = models[i].fit(iris_norm)
    predict = model.labels_
    silhouettes.append(metrics.silhouette_score(iris_norm, predict))
    plotRes(iris_tsne, predict, tag='iris', method = methods[i])

for line in silhouettes:
    print(line)

# Airline Customer Value Analysis
air_path = r'./air_data.csv'
air_data = pd.read_csv(air_path, encoding = "utf-8")
air_data.isnull().sum()
print(air_data.shape)

data = air_data.dropna().reset_index(drop=True)
print(data.shape)

# Remove data with ticket price equals 0, average discount rate not equals 0, and total flight kilometers greater than 0
t1 = data['SUM_YR_1'] == 0
t2 = data['SUM_YR_2'] == 0
t3 = data['avg_discount'] > 0
t4 = data['SEG_KM_SUM'] > 0
idx = []

for i in range(len(t1)):
    if t1[i] & t2[i] & t3[i] & t4[i] == True:
        idx.append(i)

data = data.drop(idx, axis=0)
data = data.reset_index(drop=True)
data.shape

# Take out the columns we want to analyze
data1 = data[['LOAD_TIME','FFP_DATE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]

# Calculate the membership period in months = Observation window end time - Membership registration time
m = (pd.to_datetime(data1['LOAD_TIME']) - pd.to_datetime(data1['FFP_DATE'])) // 30
data1['L'] = m.dt.days

# Take out the columns we need for the final model
data2 = data1[['L','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]

# Rename columns
data2 = data2.rename(columns={'L': 'ZL', 'LAST_TO_END': 'ZR','FLIGHT_COUNT':'ZF','SEG_KM_SUM':'ZM','avg_discount':'ZC'})

data2.describe().to_csv('describe.csv')

# Standard Normalize
std_scale = preprocessing.StandardScaler().fit(data2[["ZL","ZR","ZF","ZM","ZC"]])
df_std = std_scale.transform(data2[["ZL","ZR","ZF","ZM","ZC"]])
df_data = pd.DataFrame(df_std)
df_data.columns = list(data2.columns)

chooseNCluster(df_std, tag='air')

# Clustering
model = KMeans(n_clusters=4)
model.fit(df_std)
label_pred = model.labels_

r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis=1)
r.columns = list(data2.columns) + ['Number of Categories']

r3 = pd.concat([df_data,pd.Series(model.labels_,index=df_data.index)],axis=1)
r3.columns = list(data2.columns) + ['Cluster Category']

# Radar chart based on r2
labels = np.array(['L','R','F','M','C'])
labels = np.concatenate((labels,[labels[0]]))
N = r2.shape[1]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
data = pd.concat([r2,r2.loc[:,0]],axis=1)
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, polar=True)
for i in range(4):
    ax.plot(angles,data.loc[i,:],'o-',label=f'group{i+1}')

ax.set_thetagrids(angles*180/np.pi,labels)
plt.title(u'Customer Feature Radar Chart')
plt.legend(loc='lower right')
plt.savefig(r'./figures/redar.png')
