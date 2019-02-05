import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, train_test_split
import pylab as plt
import pandas as pd
from sklearn import preprocessing
import scipy.stats

class boosting(object):
    ejectCAS = ['10124-36-4', '108-88-3', '111991-09-4', '116-29-0', '120-12-7', '126833-17-8', '13171-21-6',
                '1333-82-0','137-30-4', '148-79-8', '1582-09-8', '1610-18-0', '2058-46-0', '2104-64-5', '21725-46-2',
                '2303-17-5', '25311-71-1', '25812-30-0', '298-00-0', '298-04-4', '314-40-9', '330-54-1', '4170-30-3',
                '4717-38-8', '50-00-0', '52645-53-1', '55406-53-6', '56-35-9', '56-38-2', '60207-90-1', '6051-87-2',
                '62-53-3', '6317-18-6', '69-72-7', '7440-02-0', '7447-40-7', '7722-84-1', '7733-02-0', '7758-94-3',
                '80844-07-1', '82657-04-3', '84852-15-3', '86-73-7', '9016-45-9', '99-35-4']
    def __init__(self):
        self.ejectCAS
    def  boost(self,df,type='lgb'):
        if len(df) == 0:
            boston = load_boston()
            df = pd.DataFrame(boston.data,columns=boston.feature_names)
            df['target']= boston.target

            y = df['target']
            x = df.drop(columns=['target'])
            X_train, X_test, y_train, y_test = train_test_split( df.drop(columns='target'), df.target, test_size=0.1, random_state=1)

        else:
            baseDf = df
            extractDf =  df['CAS'].isin(self.ejectCAS)
            df = df[~df['CAS'].isin(self.ejectCAS)]
            y = df['logTox']
            df.drop(columns=['CAS','toxValue','logTox','HDonor', 'HAcceptors', 'AromaticHeterocycles', 'AromaticCarbocycles', 'FractionCSP3'])
            dfX = df.drop(columns=['CAS','toxValue','logTox'])
            #Normalize
            for name in dfX.columns:
                print(dfX[str(name)])
                dfX.apply(lambda x: ((x - x.mean()) * 1 / x.std() + 0),axis=0)


            dfX['weight'] = scipy.stats.zscore(df['weight'])
            dfX['logP'] = scipy.stats.zscore(df['logP'])
            dfX['RotatableBonds'] = scipy.stats.zscore(df['RotatableBonds'])
            dfX['HeavyAtomCounts'] = scipy.stats.zscore(df['HeavyAtomCounts'])
            dfX['AromProp'] = scipy.stats.zscore(df['AromProp'])
            dfX['HDonor'] = scipy.stats.zscore(df['HDonor'])
            dfX['HAcceptors'] = scipy.stats.zscore(df['HAcceptors'])
            dfX['AromaticHeterocycles'] = scipy.stats.zscore(df['AromaticHeterocycles'])
            dfX['AromaticCarbocycles'] = scipy.stats.zscore(df['AromaticCarbocycles'])
            dfX['FractionCSP3'] = scipy.stats.zscore(df['FractionCSP3'])

            #x = df.drop(columns=['CAS','toxValue','logTox')
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.1, random_state=2)
        # create dataset for lightgbm
        if type=='lgb':
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
            # LightGBM parameters
            params = {
                    'task' : 'train',
                    'boosting_type' : 'gbdt',
                    'objective' : 'regression',
                    'metric' : {'l2'},
                    'num_leaves' : 60,
                    'learning_rate' : 0.08,
                    'feature_fraction' : 0.9,
                    'bagging_fraction' : 0.9,
                    'bagging_freq': 10,
                    'verbose' : 0
            }
            # train
            model = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=10)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)

        elif type == 'svr':
            from sklearn.svm import SVR
            from sklearn.model_selection import GridSearchCV
            model = SVR(gamma=0.3, C=0.01, epsilon=0.1,kernel='poly',)
            y_pred=model.fit(X_train,y_train).predict(X_test)
            y_train_pred=model.fit(X_train,y_train).predict(X_train)

        elif type == 'xgb':
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

        elif type == 'exRF':
            import sklearn.ensemble
            clf = sklearn.ensemble.ExtraTreesRegressor()
            y_pred=clf.fit(X_train,y_train).predict(X_test)
            model = clf.fit(X_train,y_train)
            y_train_pred=model.predict(X_train)

        else:
            print('no method')

        RMSE =(np.sum ((y_pred-y_test)**2)/len(y_pred))**(1/2)
        print(RMSE)
        print('test data',np.corrcoef(y_pred,y_test))
        print('train data',np.corrcoef(y_train_pred,y_train))
        extractDf = baseDf[baseDf['CAS'].isin(self.ejectCAS)]
        ex_y = extractDf['logTox']
        ex_x = extractDf.drop(
            columns=['CAS', 'toxValue', 'logTox', 'HDonor', 'HAcceptors', 'AromaticHeterocycles', 'AromaticCarbocycles',
                     'FractionCSP3'])
        val_predict = model.predict(ex_x)
        print('val data',np.corrcoef(val_predict,ex_y))
        temp =10**val_predict
        temp.tolist()
        plt.axis([-6,6, -6, 6])
        plt.scatter(y_pred,y_test)
        plt.show()
        plt.axis([-6,6, -6, 6])
        plt.scatter(y_train,y_train_pred)
        plt.show()


if __name__ == '__main__':
    import os
    os.chdir('G:\マイドライブ\Data\Meram Chronic Data')
    #df= pd.read_csv('chronicMACCSkeys.csv')
    #df= pd.read_csv('chronicMorgan.csv')
    df= pd.read_csv('MorganMACCS.csv')
    #df= pd.read_csv('clogP.csv')
    boost=boosting()
    #MorganMACCS lgb 64% xgb 61% svr 65%
    #MACCSKey lgb 62% xgb 59% svr60%
    #Morgan lgb60% xgb 60% svr 55%
    #descriptor Morgan1024MACCS lgb 68% xgb71% svr 62%
    #descriptor Morgan512MACCS lgb 70% xgb68% svr 63%

    #result
    #descriptor Morgan512MACCS lgb72% xgb66% svr64%
    boost.boost(df,'xgb')