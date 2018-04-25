# This project is to apply ARMA model to China GDP annual data from 1992 to 2015
# It uses statsmodels package to fit ARMA model to select the best order.
# It also captures GARCH effect and select best order for GARCH.
# The model is tested using test group and draws forecast line compared with true data line.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools as stl
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error as mse
Gseries = pd.read_csv(r'C:\Users\xj537\Documents\GoodGoodStudy\Undergraduate\major\time series\GDP1.csv',index_col='Date')

#processing missing data
print('Missing data amount is ',Gseries.isnull().sum().values)
Gseries = Gseries.dropna()

#stationary test
plt.figure()
plt.subplot(4,2,1)
plt.title('GDP Series')
plt.plot(Gseries)
adftest = stl.adfuller(Gseries['GDP'].values,autolag='t-stat')
print('GDP series ADF test P-value is: ',adftest[1])
GRseries = np.log(Gseries)-np.log(Gseries.shift(1))
GRseries = GRseries.dropna()
plt.subplot(4,2,2)
plt.title('GDP log return series')
plt.plot(GRseries)
adftest = stl.adfuller(GRseries['GDP'].values,autolag='t-stat')
print('GDP log return series ADF test P-value is: ',adftest[1])
#plt.show()

#train-test split
test_ratio = 0.2
test_start = round((1-test_ratio)*len(GRseries))
GRtrain = GRseries[0:test_start]
GRtest = GRseries[test_start:]

#ARMA model selection
acf = stl.acf(GRtrain,nlags=10)
plt.subplot(4,2,3)
plt.title('ACF')
plt.bar(range(len(acf)),acf)
pacf = stl.pacf(GRtrain,nlags=10)
plt.subplot(4,2,4)
plt.title('PACF')
plt.bar(range(len(pacf)),pacf)
#plt.show()
tg_bic = np.inf
tg_order = [0,0]
tg_model = None
for p in range(3):
    for q in range(3):
        try:
            model = ARMA(GRtrain['GDP'].values,order=(p,q)).fit(disp=-1,method='css',trend='c')
        except:
            continue
        bic = model.bic
        if bic<tg_bic:
            tg_order = [p,q]
            tg_model = model
            tg_bic = bic
print(tg_model.summary())

#Detecting GARCH effect
train_predict = tg_model.predict()
res = GRtrain['GDP'].values-train_predict
res_adftest = stl.adfuller(res)
print('Residuals series ADF test P-value is: ',res_adftest[1])

#GARCH model selection
acf = stl.acf(res,nlags=10)
plt.subplot(4,2,5)
plt.title('Residuals ACF')
plt.bar(range(len(acf)),acf)
pacf = stl.pacf(res,nlags=10)
plt.subplot(4,2,6)
plt.title('Residuals PACF')
plt.bar(range(len(pacf)),pacf)
#plt.show()
garch_model = stl.arma_order_select_ic(res,max_ar=1,max_ma=1,ic='bic',fit_kw={'method':'css'})
print('GARCH Model order is :',garch_model.bic_min_order)

#Test
forecast,stderr,cfitvl = tg_model.forecast(steps=len(GRtest))
print('Forecast MSE is: ',mse(GRtest,forecast))
Forecast_series = pd.concat([GRtrain,pd.DataFrame(forecast,index=GRtest.index,columns=['GDP'])])
plt.subplot(4,1,4)
plt.title('Forecast(r) vs True data(b)')
plt.plot(GRseries)
plt.plot(Forecast_series,'r')
plt.show()