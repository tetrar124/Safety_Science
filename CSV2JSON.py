import pandas as pd
import os

os.chdir(r"G:\マイドライブ\Data")
df = pd.read_csv('nikkei_stock_average_daily_jp.csv',encoding='cp932')
df.to_csv('nikkei.csv',index=None)
df = pd.read_csv('nikkei.csv')
df.to_json('nikkei.json',orient='records',force_ascii=False)