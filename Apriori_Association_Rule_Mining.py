import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori # Generate frequent itemsets
from mlxtend.frequent_patterns import association_rules # generate strong association rules
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

#Data preparation and visualization
def loadData():
    path = r"titanic.csv"
    data = pd.read_csv(path)
    print(data.info())
    data = data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'],axis=1)
    data['Age'] = data.Age.fillna(data.Age.median())
    data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])
    data["familySize"] = data["SibSp"] + data["Parch"] + 1
    data = data.drop(["SibSp", "Parch"], axis=1)
    print(data.describe())
    data.familySize = pd.cut(data.familySize, bins=[0, 4, 8, 11], labels=['small','midium', 'large'])
    data.Age = pd.cut(data.Age, bins=[0, 12, 22, 40, 100], labels=["child", "teen", "adult", "elder"])
    fareLevel = (data.Fare.max() + 2) // 3
    data.Fare = pd.cut(data.Fare, bins=[0, fareLevel, fareLevel * 2,fareLevel * 3], labels=["low","average", "high"])
    data.Pclass = data.Pclass.map({1:'Upper', 2:'Middle',3:'Lower'})
    data.Survived = data.Survived.map({1:'survived', 0:'died'})
    data = data.astype(str)
    # return data
    return np.array(data).tolist()

def solve(data):
    Encoder = TransactionEncoder()
    encodedData = Encoder.fit_transform(data)
    df = pd.DataFrame(encodedData, columns=Encoder.columns_)
    frequent_items = apriori(df, min_support=0.2,
    use_colnames=True). \
    sort_values(by='support', ascending=False)
    # print(frequent_items)
    rules = association_rules(frequent_items,
    metric='confidence', min_threshold=0.8)
    rules.sort_values(by='lift', ascending=False, inplace=True)
    print(rules)
    return rules

def plotRules(rules):
    x, y, z = rules.support, rules.confidence, rules.lift
    plt.figure(figsize=(16, 12))
    plt.scatter(x, y, c=z)
    plt.colorbar()
    plt.xlabel('support')
    plt.ylabel('confidence')
    # plt.show()
    plt.savefig(r"./rules.png")

if __name__ == '__main__':
    data = loadData()
    rules = solve(data)
    plotRules(rules)
