import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.model_selection import KFold

filePath = r'C:/Users/WeiTh/Desktop/bladder.xlsx'
dt = pd.read_excel(filePath,index_col=False)
dt=dt.drop("序号",axis=1)
df=pd.DataFrame(dt)
#Dict = {'T':1,'F':0}
#df['train'] = df['train'].map(Dict)

zscore = preprocessing.StandardScaler()
for col in df.columns :
        if col not in ["train"]:
                a=np.asarray(df[col])
                df[col] = zscore.fit_transform(a.reshape(-1,1))
#X = X[[col for col in X.columns if col not in ['lweight','age','gleason']]]

xx=df.iloc[:, :-2]
#xx = sm.add_constant(xx)
yy=df.iloc[:, -2]
train = df.loc[df.train.values =='T']
test = df.loc[df.train.values =='F']
x_train=train.iloc[:, :-2]
x_test=test.iloc[:, :-2]
y_train =train.iloc[:, -2]
y_test =test.iloc[:, -2]




print(xx)
from sklearn.model_selection import cross_validate
kfold = KFold(n_splits=10)

for train_index, test_index in kfold.split(x_train, y_train):
        # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
        this_train_x, this_train_y = x_train[train_index], y_train[train_index]  # 本组训练集
        this_test_x, this_test_y = x_train[test_index], y_train[test_index]  # 本组验证集
        # 训练本组的数据，并计算准确率
        my_model.fit(this_train_x, this_train_y)
        prediction = my_model.predict(this_test_x)
        score = accuracy_score(this_test_y, prediction)
        print(score)  # 得到预测结果区间[0,1]
	# 参数依次是：模型，训练集x，训练集y，kfold，评价指标




#10折交叉验证
from math import log
# 定义aic计算函数
def calculate_aic(n, mse, num_params):
    aic0 = n * log(mse) + 2 * num_params
    return aic0

# 定义bic计算函数
def calculate_bic(n, mse, num_params):
    bic0 = n * log(mse) + num_params * log(n)
    return bic0

def modeling(model):
        model.model_selection = True
        model.fit(x, y)
        y_pred = model.predict(x_test)
        coefi=model.coef_
        mse = mean_squared_error(y_pred, y_test)
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        mae = mean_absolute_error(y_pred, y_test)
        label = str(model)
        print(f'{label} MAE:%.4f'%mae)
        print(f'{label} MSE: %.4f'%mse)
        print(f'{label} RMSE: %.4f'%rmse)



def modeling2(alf):
        alf = math.exp(alf)
        lasso_model = LassoLars(alpha=alf)
        ridge_model = Ridge(alpha=alf)
        enet_model = ElasticNet(alpha=alf, l1_ratio=0.7)
        models = [lasso_model, ridge_model, enet_model]
        for model in models:
                model.fit(xx, yy)
                y_hat = model.predict(xx)
                mse = mean_squared_error(y_hat, yy)
                coefi = model.coef_
                num_params=np.count_nonzero(coefi)+1
                aic1 = calculate_aic(len(yy), mse, num_params)
                bic1 = calculate_bic(len(yy), mse, num_params)
                a=str(model)
                aic_.append(aic1)
                bic_.append(bic1)
                beta_.append(coefi)
                scores = -1 * cross_val_score(model, xx,yy, cv=10, scoring='neg_mean_squared_error')
                avg = sum(scores) / len(scores)
                cv_.append(avg)
                mse_.append(mse)



coef=[]
aic=[]
bic=[]
lassobeta=[]
rbeta=[]
enetbeta=[]
cv=[]
mse1=[]
for lnalf in np.arange(-9.0, 11.0, 0.01):
        aic_=[]
        bic_=[]
        beta_=[]
        cv_=[]
        mse_=[]
        modeling2(lnalf)
        aic.append(aic_)
        bic.append(bic_)
        lassobeta.append(beta_[0])
        rbeta.append(beta_[1])
        enetbeta.append(beta_[2])
        cv.append(cv_)
        mse1.append(mse_)

af=pd.DataFrame(cv)
af.to_csv('note.csv')



af=pd.DataFrame(rbeta)
af.to_csv('note.csv')
af=pd.DataFrame(enetbeta)
af.to_csv('note.csv')


def huatu(data):
        x=np.arange(-9.0, 11.0, 0.01)
        data=np.asarray(data)
        for i in range(0,data.shape[1]):
                plt.plot(x, data[:,i], marker='o', markersize=3)
        plt.legend(['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'])
        plt.ylabel('Coefficients')  # x轴标题
        plt.xlabel('Ln(Lamda)')  # y轴标题


huatu(lassobeta)
plt.title('Lasso Coefficients')
plt.show()
huatu(rbeta)
plt.title('Ridge Regression Coefficients')
plt.show()
huatu(enetbeta)
plt.title('Elastic Net Regression Coefficients')
plt.show()


def huatu1(data):
        x=np.arange(-9.0, 11.0, 0.01)
        data = np.asarray(data)
        for i in range(0, data.shape[1]):
                plt.plot(x, data[:, i], marker='o', markersize=3)
        plt.legend(['Lasso', 'Ridge', 'Elastic Net'])
        plt.xlabel('Ln(Lamda)')  # y轴标题


huatu1(aic)
plt.title('AIC')
plt.ylabel('AIC')  # x轴标题
plt.show()
huatu1(bic)
plt.title('BIC')
plt.ylabel('BIC')  # x轴标题
plt.show()
huatu1(cv)
plt.title('Cross Validation Mean-Squared Error')
plt.ylabel('Cross Validation MSE')  # x轴标题
plt.show()
huatu1(mse1)
plt.title('Mean-Squared Error')
plt.ylabel('MSE')  # x轴标题
plt.show()


from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
alpha_range = np.logspace(-18,10,200,base=10)
# LassoCV
lasso_ = LassoCV(alphas=alpha_range,cv=10).fit(xx,yy)
ridge_=RidgeCV(alphas=alpha_range,cv=10).fit(xx,yy)
enet_=ElasticNetCV(alphas=alpha_range,cv=10).fit(xx,yy)
# 查看最佳正则化系数
print([lasso_.alpha_,ridge_.alpha_,enet_.alpha_])
lasso = Lasso(alpha=lasso_.alpha_).fit(xx,yy)
ridge=Ridge(alpha=ridge_.alpha_).fit(xx,yy)
enet=ElasticNet(alpha=enet_.alpha_).fit(xx,yy)
# 返回LASSO回归的系数
res = pd.Series(index=['Intercept'] + xx.columns.tolist(), data=[lasso.intercept_] + lasso.coef_.tolist())
print(res)
res = pd.Series(index=['Intercept'] + xx.columns.tolist(), data=[ridge.intercept_] + ridge.coef_.tolist())
print(res)
res = pd.Series(index=['Intercept'] + xx.columns.tolist(), data=[enet.intercept_] + enet.coef_.tolist())
print(res)

import statsmodels.api as sm
#x = sm.add_constant(x) # 若模型中有截距，必须有这一步
model = sm.OLS(yy, xx).fit() # 构建最小二乘模型并拟合
print(model.summary()) # 输出回归结果


