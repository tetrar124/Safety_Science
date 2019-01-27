import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, train_test_split
import pylab as plt
import pandas as pd
class boosting(object):
    def  boost(self,df):
        df = None

        if df == None:
            boston = load_boston()
            df = pd.DataFrame(boston.data,columns=boston.feature_names)
            df['target']= boston.target

            y = df['target']
            x = df.drop(columns=['target'])
            X_train, X_test, y_train, y_test = train_test_split( df.drop(columns='target'), df.target, test_size=0.2, random_state=1)

        else:
            X_train, X_test, y_train, y_test = train_test_split( df.drop(columns='target'), df.target, test_size=0.2, random_state=1)

        # create dataset for lightgbm
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
        RMSE =(np.sum ((y_pred-y_test)**2)/len(y_pred))**(1/2)
        print(RMSE)
        print(np.corrcoef(y_pred,y_test))
        plt.scatter(y_pred,y_test)
        plt.show()
if __name__ == '__main__':
    boost=boosting()
    boost.boost(df)