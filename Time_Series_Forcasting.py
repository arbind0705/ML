import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from sklearn.linear_model import linearRegression

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("monthly_csv.csv")
df.head()
df.shape

print(f"Date range of gold prices avalable from -{df.loc[:, 'Date'] [0]} to {df.loc[:, 'Date'] [len(df)-1]}")

date = pd.date_range(start = '1/1/1950', end = '8/1/2020', freq = 'M')

df['month'] = date
df.drop('Date', axis = 1, inplace = True)
df = df.set_index('month')
df.head()

df.plot(figsize =(20,8))
plt.title("Gold Prices from 1950 to 2020")
plt.xlabel("months")
plt.ylabel("prices")
plt.grid()

round(df.describe(),3)

_, ax = plt.subplots(figsize =(25,8))
sns.boxplot(x = df.index.year, y=df.values[:, 0], ax = ax)
plt.title('gold price monthly since 1950 onwards')
plt.xlabel('year')
plt.ylabel('price')
plt.xticks(rotation = 90)
plt.grid()

from statemodel.graphics.tsaaplots import month_plot
fig, ax = plt.subplots(figsize =(22,8))
month_plot(df, ylabel = 'gold price', ax = ax)
plt.title('gold price monthly since 1950 onwards')
plt.ylabel('month')
plt.ylabel('price')
plt.show()

df_yearly_sum = df.resample('A').mean()
df_yearly_sum.plot()
plt.title('Yearly average gold price since 1950 onwards')
plt.xlabel('year')
plt.ylabel('price')
plt.grid()

df_quaterly_sum = df.resample('Q').mean()
df_quaterly_sum.plot()
plt.title('Quarterly average gold price since 1950 onwards')
plt.xlabel('quarter')
plt.ylabel('price')
plt.grid()

df_decade_sum = df.resample('10Y').mean()
df_decade_sum.plot()
plt.title('Decade average gold price since 1950 onwards')
plt.xlabel('decade')
plt.ylabel('price')
plt.grid()

df_1 = df.groupby(df.index.year).mean().rename(columns = {'Price': 'yearly_mean'})
df_1 = df_1.merge(df.groupby(df.index.year).std().rename(columns = {'Price': 'yearly_std'}), left_index = True, right_index = True)
df_1['cov_pct'] = ((df_1['std'] / df_1['mean']) * 100).round(2)
df_1.head()

fig, ax = plt.subplots(figsize =(15,10))
df_1['cov_pct'].plot()
plt.title('Yearly Coefficient of Variation of gold price since 1950 onwards')
plt.xlabel('year')
plt.ylabel("cv_in %")
plt.grid()

train = df [df.index.year <= 2015]
test = df [df.index.year>2015]

print(f"train data shape : {train.shape}")
print(f"test data shape : {test.shape}")


train["price"].plot(figsize = (13,5), fontsize = 15)
test["price"].plot(figsize = (13,5), fontsize = 15)
plt.grid()
plt.legend(["train", "test"])
plt.show()

train_time =[i+1 for i in range(len(train))]
test_time = [i+len(train) + 1 for i in range(len(test))]
len(train_time), len(test_time)

LR_train = train.copy()
LR_test = test.copy()

LR_train['time'] = train_time
LR_test['time'] = test_time

LR = linearRegression()
LR.fit(LR_train[['time']], LR_train['price'].values)

test_predictions_model1 = LR.predict(LR_test[['time']])
LR_test['forecast'] = test_predictions_model1

plt.figure(figsize = (14,6))
plt.plot(train['price'], label = 'Train')
plt.plot(test['price'], label = 'Test')
plt.plot(LR_test['forecast'], label = 'reg on time_test data')

plt.legend(loc='best')
plt.grid();

def mape(actual, pred): return round(np.mean(np.abs((actual - pred) / actual)) * 100, 2)

mape_model1_test = mape(test['price'].values, test_predictions_model1)
print(f"mape is {mape_model1_test:.3f}%", "%")
result = pd.DataFrame({'test mape(%)' : [mape_model1_test]}, index = ["RegressionTime"])

final_model = ExponentialSmoothing(df, trend = 'additive').fit()(smoothing_trend = 0.3, smoothing_seasonal = 0.6)

mape_final_model = mape(df['price'].values, final_model.fittedvalues)
print("MAPE:", mape_final_model)

prediction = final_model.forecast(step = len(test))

pred_df = pd.DataFrame({'lower_CI' : prediction - 1.96 + np.std(final_model.resid,dduf = 1), 'prediction':prediction, 'upper_C1':prediction+1.96*np.std(final_model.resid, ddof = 1)})

pred_df.head()

axis = df.plot(label='Actual', figsize = (18,9))
pred_df['prediction'].plot(ax=axis, label = 'forecast' alpha = 0.5)
axisfill_between(pred_df.index, pred_df['lower'], pred_df['upper_CI'], color = 'm', alpha = .15)
axis.set_xlabel('year_month')
axis.set_ylabel('price')
plt.legend(loc = 'best')
plt.grind()
plt.show()