import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns

filePath = r'C:/Users/WeiTh/Desktop/bladder.xlsx'
dt = pd.read_excel(filePath,index_col=False)
dt=dt.drop("序号",axis=1)
print(dt)
df=pd.DataFrame(dt)
Dict = {'T':1,
        'F':0}
df['train'] = df['train'].map(Dict)
#文字生成哑变量
dx=df.drop("lpsa",axis=1)
print(dx)
#sns.pairplot(dt)
#plt.show()

pd.get_dummies(dt["train"])
label=["lcavol","lweight","age","lbph",	"svi","lcp","gleason","pgg45","lpsa","train"]
print(dt)
for i in list([0,1,2,3,5,6,7,8]):
    plt.subplot(2, 5, i + 1)
    plt.boxplot(dt[label[i]],
                medianprops={'color': 'green', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                )
    plt.xlabel(label[i])
plt.subplot(2, 5, 5)
plt.hist(dt.svi)
plt.xlabel(label[4])
plt.subplot(2, 5, 10)
plt.hist(dt.train)
plt.xlabel(label[9])
plt.show()

af=df.describe()
af.to_csv('note.csv')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
def checkVIF_new(df):
    df["c"]=1
    name = df.columns
    x=np.matrix(df)
    VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
    VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
    VIF = VIF.drop([9],axis = 0)
    max_VIF = max(VIF_list)
    print(max_VIF)
    return VIF
af=checkVIF_new(dx)
#fileName='note.csv'
#af.to_csv('note.csv')
#print(af)

#用sklearn包
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x=dx
y=df["lpsa"]
model.fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


x=dx
y=df["lpsa"]
#y = df.iloc[:, 0] # 因变量为第1列数据
#x = df.iloc[:, 1:10] # 自变量为第n列数据



#用statsmodels包
import statsmodels.api as sm
x = sm.add_constant(x) # 若模型中有截距，必须有这一步
model = sm.OLS(y, x).fit() # 构建最小二乘模型并拟合
print(model.summary()) # 输出回归结果
# 画图
# 这两行代码在画图时添加中文必须用
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#Cook距离
infl = model.get_influence()
cook_d = infl.cooks_distance[0]
print(infl.cooks_distance[0])
plt.stem(range(1,y.shape[0]+1),infl.cooks_distance[0],linefmt="-.",markerfmt="o",basefmt="-")
plt.xlabel("Observation")
plt.ylabel("Cook's D")
plt.show()
af=infl.cooks_distance
af=pd.DataFrame(infl.cooks_distance)
af.to_csv('note.csv')
predicts = model.predict() # 模型的预测值
err=y-predicts

plt.subplot(1, 2, 1)
plt.scatter(y,predicts) # 散点图
plt.xlabel("实际值")
plt.ylabel("预测值")
plt.plot(y, y, color = 'green', label='y=x')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(y,err) # 散点图
plt.xlabel("因变量")
plt.ylabel("残差值")
a=y-y
plt.plot(y, a, color = 'green', label='y=0')
plt.legend()
plt.show() # 显示图形

label1=["lcavol","lweight","age","lbph","svi","lcp","gleason","pgg45","train"]
for i in range(0,9):
    plt.subplot(2, 5, i + 1)
    plt.scatter(dx.iloc[:,i], err)  # 散点图
    plt.xlabel(label1[i])
    plt.ylabel("残差值")
plt.show()




#变量选择

from math import log
# 定义aic计算函数
def calculate_aic(n, mse, num_params):
    aic0 = n * log(mse) + 2 * num_params
    return aic0

# 定义bic计算函数
def calculate_bic(n, mse, num_params):
    bic0 = n * log(mse) + num_params * log(n)
    return bic0

model = LinearRegression()
n=[]
aic=[]
bic=[]
im=[1]
im=np.repeat(im,len(y),axis=0)
im=im.reshape(-1,1)
#矩阵叉乘，向量点乘用pd.dot
#矩阵点乘用*
#向量叉乘用pd.cross
from sklearn.metrics import mean_squared_error
for aa in range(0,2):
    for bb in range(0,2):
        for cc in range(0,2):
            for dd in range(0,2):
                for ee in range(0,2):
                    for ff in range(0,2):
                        for gg in range(0,2):
                            for hh in range(0,2):
                                for ii in range(0,2):
                                    x = dx
                                    k= np.array([aa,bb,cc,dd,ee,ff,gg,hh,ii])
                                    #k=k.reshape(-1,1)
                                    if sum(k)==0:
                                        continue
                                    kk=np.repeat(k, len(y), axis=0)
                                    kk=kk.reshape(len(k),-1)
                                    kk=np.transpose(kk)
                                    x=x.iloc[:,0:9]*kk
                                    x=np.array(x)
                                    x=x[:,x.sum(axis=0)!=0]
                                    #去除全0列
                                    x = sm.add_constant(x)  # 若模型中有截距，必须有这一步
                                    model.fit(x, y)
                                    yhat = model.predict(x)
                                    mse = mean_squared_error(y, yhat)
                                    num_params = len(model.coef_) + 1
                                    aic1 = calculate_aic(len(y), mse, num_params)
                                    bic1 = calculate_bic(len(y), mse, num_params)
                                    n.append(k)
                                    aic.append(aic1)
                                    bic.append(bic1)

'''
print("n=",n)
print("aic=",aic)
print("bic=",bic)
'''
n=pd.DataFrame(n)
aic=np.asarray(aic)
bic=np.asarray(bic)
aic=aic.reshape(-1,1)
bic=bic.reshape(-1,1)
aic=pd.DataFrame(aic)
bic=pd.DataFrame(bic)
out=pd.concat([n,aic,bic],axis=1)
print(out)
out.to_csv("note.csv")
print(out.describe())

plt.figure(figsize=(14, 14))
cor = df.corr(method='pearson')
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(font_scale=0.7,rc=rc)  # 设置字体大小
sns.heatmap(cor,
            annot=True,  # 显示相关系数的数据
            center=0.5,  # 居中
            fmt='.2f',  # 只显示两位小数
            linewidth=0.5,  # 设置每个单元格的距离
            linecolor='blue',  # 设置间距线的颜色
            vmin=0, vmax=1,  # 设置数值最小值和最大值
            xticklabels=True, yticklabels=True,  # 显示x轴和y轴
            square=True,  # 每个方格都是正方形
            cbar=True,  # 绘制颜色条
            cmap='Blues',  # 设置热力图颜色
            )
plt.show() #显示图片
#颜色有很多种，只要输入”？“=“Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r,
# CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r,
# OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r,
# Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r,
# Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r,RdYlBu, RdYlBu_r, RdYlGn,
# RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r,
# Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r,
# afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r,
# cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r,
# flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar,
# gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r,gist_yarg, gist_yarg_r,
# gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r,hsv, hsv_r, icefire,icefire_r,
# inferno,inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r,
# ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket,
# rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r,
# tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted,
# twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r”













