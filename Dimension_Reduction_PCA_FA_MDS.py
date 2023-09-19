import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import svd
from factor_analyzer import FactorAnalyzer
from sklearn import preprocessing


def loadData():
    path = r'./LANeighborhoods.csv'
    df = pd.read_csv(path, encoding='utf-8')
    df = df.drop('LA.Nbhd', axis=1)
    return df


def pca(X):
    covX = np.cov(X.T)
    featValue, featVec = np.linalg.eig(covX)
    pairs = [[v, i] for i, v in enumerate(featValue)]
    pairs.sort(key=lambda x: -x[0])
    featValue = [pairs[i][0] for i in range(len(pairs))]
    idx = [pairs[i][1] for i in range(len(pairs))]
    # Scree Plot
    plt.scatter(range(1, df.shape[1] + 1), featValue)
    plt.plot(range(1, df.shape[1] + 1), featValue)
    plt.title('Scree Plot')
    plt.xlabel("Factors")
    plt.ylabel('Eigenvalue')
    plt.savefig(r'./figures/scree.png')
    plt.close()
    gx = np.cumsum(featValue / np.sum(featValue))
    # Explained Variance Ratio Plot
    plt.scatter(range(1, df.shape[1] + 1), gx)
    plt.plot(range(1, df.shape[1] + 1), gx)
    plt.title('Explained Variance Ratio Plot')
    plt.xlabel("Factors")
    plt.ylabel('Ratio')
    plt.savefig(r'./figures/ratio.png')
    plt.close()
    selectVec = [featVec.T[idx[i]] for i in range(4)]
    selectVec = np.matrix(selectVec).T
    lowDimData = np.dot(X, selectVec)
    return lowDimData


def FactorAnalysisFeatureSelection(data, n_comp):
    fa = FactorAnalyzer(n_comp, rotation='varimax')
    fa.fit(data)
    df_cm = pd.DataFrame(np.abs(fa.loadings_))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_cm, cmap='BuPu', ax=ax, yticklabels=df.columns)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_title("Factor Analysis", fontsize=12)
    ax.set_ylabel('Sepal Width')
    plt.savefig(r'./figures/res.png')
    plt.close()
    return pd.DataFrame(fa.transform(X))


def mds(d, dimension):
    (n, n) = shape(d)
    t = zeros((n, n))
    d_square = d ** 2
    d_sum = sum(d_square)
    d_sum_row = sum(d_square, axis=0)
    d_sum_col = sum(d_square, axis=1)
    for i in range(n):
        for j in range(n):
            t[i, j] = -(d_square[i, j] - d_sum_row[i] / n - d_sum_col[j] / n + d_sum / (n * n)) / 2
    [U, S, V] = svd(t)
    X_original = U * sqrt(S)
    X = X_original[:, 0:dimension]
    return X


def plotMDS(a):
    plt.scatter(a[:, 0], a[:, 1])
    for i in range(len(df.columns)):
        plt.annotate(df.columns[i], xy=(a[i, 0], a[i, 1]), xytext=(a[i, 0] + 0.01, a[i, 1] + 0.01))

    plt.title('2-D MDS Analysis')
    plt.savefig(r'./figures/mds.png')
    plt.close()


if __name__ == '__main__':
    df = loadData()
    # Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(df.values)
    pca_data = pca(X)
    fa_data = FactorAnalysisFeatureSelection(X, 4)
    mds_data = mds(X, 2)
    plotMDS(mds_data)
