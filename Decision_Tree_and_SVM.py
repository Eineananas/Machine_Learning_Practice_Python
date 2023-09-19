from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# Display Chinese characters in plots
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Unicode encoding for minus sign

# Load data
data_tianmao_train = pd.read_csv(r'./data/Train_1.txt', encoding='utf-8')
data_tianmao_test = pd.read_csv(r'./data/Test_1.txt', encoding='utf-8')

def plot_hist(data, outcome):
    sns.set_theme(style='ticks')
    sns.pairplot(data, hue=outcome)
    plt.savefig(f'./figures/{outcome}.png')
    plt.show()

X_train = data_tianmao_train.iloc[:, 1:]
y_train = data_tianmao_train.iloc[:, 0]
X_test = data_tianmao_test.iloc[:, 1:]
y_test = data_tianmao_test.iloc[:, 0]

data_tianmao_train.describe().T.to_csv(r'./data/tianmao.csv')
plot_hist(data_tianmao_train, 'BuyOrNot')

# Function for Decision Tree parameter search
def dt_param_search(X_train, y_train, tag):
    dt = DecisionTreeClassifier()
    param_grid = {
        'max_depth': [None, 2, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_

# Function for SVM parameter search
def svm_param_search(X_train, y_train, tag):
    svm = SVC()
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_

# Function to plot ROC and AUC
def plot_roc_auc(classifier, y_test, tag):
    if hasattr(classifier, "decision_function"):
        y_scores = classifier.decision_function(X_test)
    else:
        y_scores = classifier.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'./figures/{tag}_{classifier}_roc.png')
    plt.show()

# Function to perform modeling using Decision Tree and SVM
def modeling_by_dt_svm(X_train, y_train, X_test, y_test, tag, isbinary=True):
    dt = dt_param_search(X_train, y_train, tag)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    if isbinary:
        plot_roc_auc(dt, y_test, tag)
    acc = accuracy_score(y_pred, y_test)
    print("dt_acc:", acc)
    svc = svm_param_search(X_train, y_train, tag)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    if isbinary:
        plot_roc_auc(svc, y_test, tag)
    print('svc_acc:', accuracy_score(y_pred, y_test))
    return dt, svc

# Function to plot Decision Tree
def plot_dt(dt, tag):
    dot_data = export_graphviz(dt, out_file=None, feature_names=X_train.columns, class_names=y_train.name, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"./figures/{tag}_decision_tree")
    dt, svc = modeling_by_dt_svm(X_train, y_train, X_test, y_test, 'tianmao')
    plot_dt(dt, 'tianmao')

# Function to load data from a file
def load_data(path):
    datasets = []
    f = open(path)
    for line in f:
        recording = line[:-1].split(' ')
        while '' in recording:
            recording.remove('')
        datasets.append(recording)
    f.close()
    cols = datasets[0]
    return pd.DataFrame(datasets[1:], columns=cols)

data_hemophilia = load_data(r'./data/hemophilia.txt')
for col in data_hemophilia.columns:
    if col == 'py':
        data_hemophilia[col] = data_hemophilia[col].astype('float')
    else:
        data_hemophilia[col] = data_hemophilia[col].astype('int64')
plot_hist(data_hemophilia, 'deaths')
data_hemophilia.describe().T.to_csv(r'./data/deaths.csv')
X_train, X_test, y_train, y_test = train_test_split(data_hemophilia.iloc[:, :-1], data_hemophilia['deaths'], test_size=0.2, random_state=2023)
dt, svc = modeling_by_dt_svm(X_train, y_train, X_test, y_test, 'deaths', isbinary=False)

data = pd.read_excel(r'./data/tax_data.xls', index_col=0)
data.describe().T.to_csv(r'./data/car.csv')
data['输出'] = pd.factorize(data['输出'])[0]
one_hots = pd.get_dummies(data[['销售类型', '销售模式']])
data[one_hots.columns] = one_hots
data = data.drop(['销售类型', '销售模式'], axis=1)
X = data[[col for col in data.columns if col != '输出']]
y = data['输出']
minmax = preprocessing.MinMaxScaler()
X = pd.DataFrame(minmax.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
dt, svc = modeling_by_dt_svm(X_train, y_train, X_test, y_test, 'car')
plot_dt(dt, 'car')
