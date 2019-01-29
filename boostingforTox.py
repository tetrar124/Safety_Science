import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, train_test_split
import pylab as plt
import pandas as pd
class boosting(object):
    def  boost(self,df,type):
        if len(df) == 0:
            boston = load_boston()
            df = pd.DataFrame(boston.data,columns=boston.feature_names)
            df['target']= boston.target

            y = df['target']
            x = df.drop(columns=['target'])
            X_train, X_test, y_train, y_test = train_test_split( df.drop(columns='target'), df.target, test_size=0.2, random_state=1)

        else:
            y = df['logTox']
            x = df.drop(columns=['CAS','toxValue','logTox'])
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.1, random_state=1)

        # create dataset for lightgbm
        if type=lgb:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
            # LightGBM parameters
            params = {
                    'task' : 'train',
                    'boosting_type' : 'gbdt',
                    'objective' : 'regression',
                    'metric' : {'l2'},
                    'num_leaves' : 31,
                    'learning_rate' : 0.1,
                    'feature_fraction' : 0.9,
                    'bagging_fraction' : 0.8,
                    'bagging_freq': 5,
                    'verbose' : 0
            }

            # train
            gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=10)
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        if type = svr:
            from sklearn.svm import SVR
            from sklearn.model_selection import GridSearchCV
            clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
            y_pred=clf.fit(X_test,y_test).predict(X_test)
        if type = xgb:
            import xgboost as xgb
            mod = xgb.XGBRegressor()
            mod.fit(X_train, y_train)
            y_pred = mod.predict(X_test)
        RMSE =(np.sum ((y_pred-y_test)**2)/len(y_pred))**(1/2)
        print(RMSE)
        print(np.corrcoef(y_pred,y_test))
        plt.scatter(y_pred,y_test)
        plt.show()
if __name__ == '__main__':
    import os
    os.chdir('G:\マイドライブ\Data\Meram Chronic Data')
    df= pd.read_csv('chronicMACCSkeys.csv')
    boost=boosting()
    boost.boost(df)