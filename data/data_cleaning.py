import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


########## data cleaning #########

# merge minute level data 
years = [2017, 2018, 2019, 2020, 2021]
df = pd.DataFrame()
for y in years:
    # might need to adjust file paths
    df_y =  pd.read_csv('bitcoin/Bitstamp_BTCUSD_{0}_minute.csv'.format(str(y)), skiprows=(1))
    print(y)
    print(df_y.shape)
    df_y = df_y.sort_values('date')
    df = df.append(df_y)


# select data from 9 AM - 4 PM
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].apply(lambda x: x.hour)
df = df[(df['hour'] >= 9) & (df['hour'] <= 15)]
df = df.set_index('date')


# perform data resample
agg_dict = {'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min',
            'Volume BTC': 'sum', 'Volume USD': 'sum'}
    
df_daily = df.resample('D',label='right').agg(agg_dict)
df_daily.to_csv('total_BTC_day.csv')


# fill in missing values using data from yahoo finance
df_miss = pd.read_csv('bitcoin/BTC-USD.csv')
df_miss['Date'] = pd.to_datetime(df_miss['Date'])
df_miss.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'Volume USD']
df_miss = df_miss.set_index('date')
df_miss['Volume BTC'] = df_miss['Volume USD'] / df_miss['close']
df_miss = df_miss[['open', 'close', 'high', 'low', 'Volume BTC', 'Volume USD']]

df_daily.loc[df_miss.index] = df_miss
df_daily['vwap'] = df_daily['Volume USD'] / df_daily['Volume BTC']

df_daily['rtn'] = df_daily['close'].pct_change()


##### technical indicators calculation #####

# momentum

df_daily['ROC'] = ta.momentum.ROCIndicator(df_daily['close'], 7).roc()
df_daily['RSI'] = ta.momentum.RSIIndicator(df_daily['close'], 7).rsi()
df_daily['TSI'] = ta.momentum.TSIIndicator(df_daily['close'], 14, 7).tsi()

# volume
df_daily['ADI'] = ta.volume.AccDistIndexIndicator(df_daily['high'], df_daily['low'], df_daily['close'], df_daily['Volume BTC']).acc_dist_index()
df_daily['MFI'] = ta.volume.MFIIndicator(df_daily['high'], df_daily['low'], df_daily['close'], df_daily['Volume BTC'], 7).money_flow_index()
df_daily['OBV'] = ta.volume.OnBalanceVolumeIndicator(df_daily['close'], df_daily['Volume BTC']).on_balance_volume()

# volatility
df_daily['ATR'] = ta.volatility.AverageTrueRange(df_daily['high'], df_daily['low'], df_daily['close'], 7).average_true_range()
df_daily['volatility'] = df_daily['rtn'].rolling(7).std()

# trend
df_daily['MA'] = df_daily['close'] / df_daily['close'].rolling(7).mean()
df_daily['aroon'] = ta.trend.AroonIndicator(df_daily['close'], 7).aroon_indicator()
df_daily['EMA'] = ta.trend.EMAIndicator(df_daily['close'], 7).ema_indicator()
df_daily['WMA'] = ta.trend.WMAIndicator(df_daily['close'], 7).wma()


##### other factors
hash_rate = pd.read_csv('bitcoin/BCHAIN-HRATE.csv')
hash_rate['Date'] = pd.to_datetime(hash_rate['Date'])
hash_rate = hash_rate.set_index('Date')
hash_rate.columns = ['hash_rate']
df_daily = pd.merge(df_daily, hash_rate, 'left', left_index = True, right_index = True)

FB_daily = pd.read_csv('bitcoin/FB_daily.csv')
FB_daily['Date'] = pd.to_datetime(FB_daily['Date'])
FB_daily = FB_daily.set_index('Date')
FB_daily = FB_daily[['Close']]
FB_daily.columns = ['FB_close']
df_daily = pd.merge(df_daily, FB_daily, 'left', left_index = True, right_index = True)

Microstrategy_daily = pd.read_csv('bitcoin/Microstrategy_daily.csv')
Microstrategy_daily['Date'] = pd.to_datetime(Microstrategy_daily['Date'])
Microstrategy_daily = Microstrategy_daily.set_index('Date')
Microstrategy_daily = Microstrategy_daily[['Close']]
Microstrategy_daily.columns = ['Microstrategy_close']
df_daily = pd.merge(df_daily, Microstrategy_daily, 'left', left_index = True, right_index = True)

gold = pd.read_csv('bitcoin/gold.csv')
gold['Date'] = pd.to_datetime(gold['Date'])
gold = gold.set_index('Date')
gold = gold[['Close']]
gold.columns = ['gold_price']
df_daily = pd.merge(df_daily, gold, 'left', left_index = True, right_index = True)


GPU = pd.read_csv('bitcoin/NVDA_daily.csv')
GPU['Date'] = pd.to_datetime(GPU['Date'])
GPU = GPU.set_index('Date')
GPU = GPU[['Close']]
GPU.columns = ['GPU_price']
df_daily = pd.merge(df_daily, GPU, 'left', left_index = True, right_index = True)

Google_trend = pd.read_csv('bitcoin/Google_Trends_daily.csv')
Google_trend['date'] = pd.to_datetime(Google_trend['date'])
Google_trend = Google_trend.set_index('date')
Google_trend = Google_trend[['bitcoin_unscaled']]
Google_trend.columns = ['Google_trend']
df_daily = pd.merge(df_daily, Google_trend, 'left', left_index = True, right_index = True)

df_daily['next_day_close'] = df_daily['close'].shift(-1) 
df_daily['next_day_group'] = df_daily['close'].shift(-1) > df_daily['open'].shift(-1)
df_daily['next_day_group'] = df_daily['next_day_group'].apply(lambda x: int(x))
factors = df_daily.iloc[:, 8:]
factors = factors.fillna(method = 'ffill')
factors = factors.dropna()
factors.to_csv('total_factors.csv')
factors['date'] = factors.index
factors['year'] = factors['date'].apply(lambda x: x.year)

train_x = factors[factors.year <= 2020].iloc[:, :18]
test_x = factors[factors.year > 2020].iloc[:, :18]
train_y_regression = factors[factors.year <= 2020][['next_day_close']]
train_y_classfication = factors[factors.year <= 2020][['next_day_group']]
test_y_regression = factors[factors.year > 2020][['next_day_close']]
test_y_classfication = factors[factors.year > 2020][['next_day_group']]

train_x.to_csv('train_x.csv')
test_x.to_csv('test_x.csv')
train_y_regression.to_csv('train_y_regression.csv')
train_y_classfication.to_csv('train_y_classification.csv')
test_y_regression.to_csv('test_y_regression.csv')
test_y_classfication.to_csv('test_y_classification.csv')


####### classification ########
technical_x_train = train_x.iloc[:, 0:12]
technical_x_test = test_x.iloc[:, 0:12]

fundamental_x_train = train_x.iloc[:, 12:]
fundamental_x_test = test_x.iloc[:, 12:]

y = train_y_classfication
model = LogisticRegression().fit(train_x, y)
print('train accuracy overall: {0}'.format(str(model.score(train_x, y))))
print('test accuracy overall: {0}'.format(str(model.score(test_x, test_y_classfication))))

model1 = LogisticRegression().fit(technical_x_train, y)
print('train accuracy of technical: {0}'.format(str(model1.score(technical_x_train, y))))
print('test accuracy of technical: {0}'.format(str(model1.score(technical_x_test, test_y_classfication))))

model2 = LogisticRegression().fit(fundamental_x_train, y)
print('train accuracy of fundamental: {0}'.format(str(model2.score(fundamental_x_train, y))))
print('test accuracy of fundamental: {0}'.format(str(model2.score(fundamental_x_test, test_y_classfication))))


###### heat map ######
correlation = train_x.corr()
sns.heatmap(correlation)
