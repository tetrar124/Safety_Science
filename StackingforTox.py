import numpy as np
import itertools
import pandas as pd
import os
from lightgbm import LGBMRegressor
#from fastFM import sgd
from rgf.sklearn import RGFRegressor
import xgboost
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate,ShuffleSplit

from mlxtend.regressor import StackingRegressor
from mlxtend.feature_selection import ColumnSelector
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.pipeline import make_pipeline
import scipy as sp
from sklearn.preprocessing import normalize
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA

from bayes_opt import BayesianOptimization

class predata(object):

    # ensamble average
    class extAverage(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            result = np.average(X, axis=1)
            return self

        def transform(self, X):
            return self

        def predict(self, X):
            print(X.shape)
            result = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 0.5) / 3.5
            # result = np.average(X, axis = 1)
            return result

    # PCA
    class extPCA(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            model = PCA(n_components=64)
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            # morgan = morgan.reset_index().drop('index', axis=1)
            W = pd.DataFrame(model.fit_transform(X))
            # W = pd.concat([morgan,W],axis=1)
            return W

        def predict(self, X):
            model = PCA(n_components=64)
            # maccs,morgan,descriptor,klekotaToth,newFingerprint = sepTables(X)
            # morgan = morgan.reset_index().drop('index', axis=1)
            W = pd.DataFrame(model.fit_transform(X))
            # W = pd.concat([morgan,W],axis=1)
            return W

    # All
    class extAll(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            valiable = pd.concat([maccs, morgan, descriptor, klekotaToth, newFingerprint], axis=1)
            return valiable

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptor = pd.concat([maccs, morgan, descriptor, klekotaToth, newFingerprint], axis=1)
            return descriptor
    # for linear regression
    class forlinear(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            valiable = pd.concat([maccs, morgan, descriptor, newFingerprint], axis=1)
            return valiable

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptor = pd.concat([maccs, morgan, descriptor, newFingerprint], axis=1)
            return descriptor
    # with new
    class extdescriptorNew(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptorNew = pd.concat([descriptor, newFingerprint], axis=1)
            return descriptorNew

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptorNew = pd.concat([descriptor, newFingerprint], axis=1)
            return descriptorNew

    class extMorganKlekotaTothNew(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            three = pd.concat([morgan, klekotaToth, newFingerprint], axis=1)
            return three

        def predict(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            three = pd.concat([morgan, klekotaToth, newFingerprint], axis=1)
            return three

    class extMorganNew(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            morganAndNew = pd.concat([morgan, newFingerprint], axis=1)
            return morganAndNew

        def predict(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            morganAndNew = pd.concat([morgan, newFingerprint], axis=1)
            return morganAndNew

    class extklekotaTothNew(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            klekotaTothandandNew = pd.concat([klekotaToth, newFingerprint], axis=1)
            return klekotaTothandandNew

        def predict(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            klekotaTothandandNew = pd.concat([klekotaToth, newFingerprint], axis=1)
            return klekotaTothandandNew

    class extMACCSNew(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptorNew = pd.concat([maccs, newFingerprint], axis=1)
            return descriptorNew

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([maccs, newFingerprint], axis=1)
            return variable

    # with MACCS
    class extMorganMACCS(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([morgan, maccs], axis=1)
            return variable

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([morgan, maccs], axis=1)
            return variable

    class extKlekotaTothMACCS(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([klekotaToth, maccs], axis=1)
            return variable

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([klekotaToth, maccs], axis=1)
            return variable

    class extDescriptorMACCS(BaseEstimator, TransformerMixin, RegressorMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([descriptor, maccs], axis=1)
            return variable

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([descriptor, maccs], axis=1)
            return variable

    # old
    class extMorgan(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            _, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            morgan = pd.concat([morgan, klekotaToth, newFingerprint], axis=1)
            return morgan

    class extMACCS(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, _, klekotaToth, newFingerprint = sepTables(X)
            maccs = pd.concat([maccs, morgan, klekotaToth, newFingerprint], axis=1)

            return maccs

        def predict(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptor = pd.concat([maccs, morgan, klekotaToth, newFingerprint], axis=1)
            return descriptor

    class extDescriptor(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            descriptor = pd.concat([maccs, morgan, descriptor, klekotaToth, newFingerprint], axis=1)
            return descriptor

    # without new
    class extAllwithoutNew(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([maccs, morgan, descriptor, klekotaToth], axis=1)
            return variable

    # without MACCS
    class extAllwithoutMaccs(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([morgan, descriptor, klekotaToth, newFingerprint], axis=1)
            return variable

    # without Descriptor
    class extAllwithoutDescriptor(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            variable = pd.concat([morgan, maccs, klekotaToth, newFingerprint], axis=1)
            return variable

    # only 1 fingerprint
    class extOnlyDescriptor(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            return descriptor

    class extOnlyMorgan(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            return morgan

    class extOnlyklekotaToth(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            return klekotaToth

    class extOnlyMACCS(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            maccs, morgan, descriptor, klekotaToth, newFingerprint = sepTables(X)
            return maccs
class optimizeHyperparamerte(object):
    def optimizeRF(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def randomforest_cv(n_estimators, min_samples_split, max_features):
            score = cross_validate(
                RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    min_samples_split=int(min_samples_split),
                    max_features=max_features,
                    random_state=0
                ),
                X, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val
        opt = BayesianOptimization(
                randomforest_cv,
                {'n_estimators': (10, 250),
                 'min_samples_split': (2, 25),
                 'max_features': (0.1, 0.999)}
            )
        opt.maximize(init_points=10,n_iter=50)
        opt.max
    def optimizeXGBoost(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def xgboost_cv(n_estimators,max_depth, gamma, colsample_bytree):
            score = cross_validate(
                xgboost.XGBRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    gamma=gamma,
                    colsample_bytree=colsample_bytree,
                ),
                X, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val
        opt = BayesianOptimization(
                xgboost_cv,
                {
                    'n_estimators':(5,10000),
                    'max_depth': (3, 1000),
                    'gamma': (0,2),
                    'colsample_bytree': (0.1, 1)
                }
            )
        opt.maximize(init_points=10,n_iter=50)
        opt.max
    def optimizeRGF(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def RGF_cv(max_leaf, l2, min_samples_leaf):
            score = cross_validate(
                RGFRegressor(
                    max_leaf=int(max_leaf),
                    algorithm="RGF",
                    test_interval=100,
                    loss="LS",
                    verbose=False,
                    l2=l2,
                    min_samples_leaf = int(min_samples_leaf)
                ),
                X, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val
        opt = BayesianOptimization(
                RGF_cv,
                {'max_leaf': (100, 2000),
                 'l2': (0.1, 1),
                 'min_samples_leaf': (1,20)}
            )
        opt.maximize(init_points=20,n_iter=100)
        opt.max
    def optimizeLightGBM(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def LGBM_cv(num_leaves,feature_fraction, bagging_fraction, min_data_in_leaf,max_depth):
            score = cross_validate(
                LGBMRegressor(
                    boosting_type='gbdt',
                    num_leaves = int(num_leaves),
                    feature_fraction = feature_fraction,
                    bagging_fraction=bagging_fraction,
                    learning_rate=0.06,
                    min_data_in_leaf = int(min_data_in_leaf),
                    max_depth=int(max_depth)
                ),
                X, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val
        opt = BayesianOptimization(
                LGBM_cv,
                {'min_data_in_leaf': (0, 300),
                 'feature_fraction': (0.01, 1),
                 'max_depth': (7,50),
                 'num_leaves':(100,3000),
                 'bagging_fraction' :(0.01,1)
                 }
            )
        opt.maximize(init_points=20,n_iter=100)
        opt.max
    def optimizeExtratreeReg(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def ExtraTree_cv(n_estimators, max_features, min_samples_split, max_depth, min_samples_leaf):
            score = cross_validate(
                ExtraTreesRegressor(
                    n_estimators = int(n_estimators),
                    max_features = int(max_features),
                    min_samples_split = min_samples_split,
                    max_depth = int(max_depth),
                    min_samples_leaf = int(min_samples_leaf),
                ),
                X, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val
        opt = BayesianOptimization(
                ExtraTree_cv,
                {
                    'n_estimators': (100, 2000),
                    'max_features': (10,5000),
                    'min_samples_split': (0.01, 1),
                    'max_depth':(10,1000),
                    'min_samples_leaf': (1,20)
                }
            )
        opt.maximize(init_points=20,n_iter=100)
        opt.max
    def optSVR(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def forOptSVR( kernel,gamma,epsilon, C):
            print(kernel)
            if kernel >=1 and kernel <= 10:
                kernel = 'rbf'
            elif kernel > 11 and kernel <=20:
                kernel = 'linear'
            elif kernel >21 and kernel <=30:
                kernel = 'sigmoid'
            else:
                kernel ='poly'
            score = cross_validate(
                SVR(
                    kernel= kernel,
                    gamma = gamma,
                    epsilon = epsilon,
                    C = C,
                ),
                X2, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val

        opt = BayesianOptimization(
                forOptSVR,
                {
                    'kernel':(1,10),
                    'gamma' :(2**(-20),2**11),
                    'epsilon':(2**(-10),2**1),
                    'C' :(2 **(-5),2**11)
                }
            )
        opt.maximize(init_points=20,n_iter=100)
        opt.max

    def optRedge(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        def forOptRidge( alpha):
            score = cross_validate(
                Ridge(
                    alpha= alpha,
                ),
                X2, y,
                scoring='neg_mean_squared_error',
                cv=cv, n_jobs=-1)
            val = score['test_score'].mean()
            return val

        opt = BayesianOptimization(
            forOptRidge,{
                'alpha':(10,1000)
        }
        )
        opt.maximize(init_points=20, n_iter=100)
        opt.max
    #the value of step and stdev optimzation is very difficult
    def oftFM(self):
        from sklearn import preprocessing
        sparseX = sp.sparse.csc_matrix(X2.values)
        sparseX = preprocessing.normalize(sparseX)
        def foroptFMRegression(n_iter,rank, l2_reg_w, l2_reg_V):
            cv = KFold(n_splits=10, shuffle=True, random_state=0)
            score = cross_validate(
                    sgd.FMRegression(
                    n_iter=n_iter,
                    init_stdev=0.1,
                    rank=int(rank),
                    l2_reg_w=l2_reg_w,
                    l2_reg_V=l2_reg_V,
                    step_size=0.1,
                ),
                sparseX, y,
                scoring='neg_mean_squared_error',
                cv=cv,n_jobs=-1)
            val = score['test_score'].mean()
            return val

        opt = BayesianOptimization(
                foroptFMRegression,
                {
                    'n_iter': (1000,100000),
                    'rank' : (1,100),
                    'l2_reg_w':(0,10),
                    'l2_reg_V':(0,10),
                    }
                )
        opt.maximize(init_points=10,n_iter=100)
        opt.max


class toxPredict(object):

    os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')

    #make user evaluation method
    # def calcRMSE(real,pred):
    #   RMSE = (np.sum((pred-real.tolist())**2)/len(pred))**(1/2)
    #   return RMSE
    # def calcCorr(real,pred):
    #   corr = np.corrcoef(real, pred.flatten())[0,1]
    #   return corr
    # from sklearn.metrics import make_scorer
    # myScoreFunc = {'RMSE':make_scorer(calcRMSE),'Correlation coefficient':make_scorer(calcCorr)}

    def sepTables(xdf=None):
        try:
            if xdfdf == None:
                xdf = pd.read_csv('MorganMACCS.csv')
        except:
            pass
        try:
            #y = df['toxValue']
            #y = df['logTox']
            xdf = xdf.drop(['toxValue','logTox'],axis=1)
            xdf = xdf.set_index('CAS')
        except:
            pass
        key = 167
        MACCS = xdf.iloc[:,0:key]
        Morgan = xdf.iloc[:,key:key+512]
        descriptors = xdf.iloc[:,key+512:690]
        klekotaToth = xdf.iloc[:,690:5550]
        newFingerprint = xdf.iloc[:,5550:]
        #similarity = xdf.iloc[:,690:]
        tables = []
        # for table in [MACCS,Morgan,descriptors,newFingerprint]:
        #     table = table[table.columns[table.sum()!=0]]
        #     tables.append(table)
        #return tables[0],tables[1],tables[2],tables[3]
        return MACCS,Morgan,descriptors,klekotaToth,newFingerprint

    def calcACC(testmodel,X=X,name=None,y=y):
        def calcRMSE(real, pred):
            RMSE = (np.sum((pred - real.tolist()) ** 2) / len(pred)) ** (1 / 2)
            return RMSE

        def calcCorr(real, pred):
            corr = np.corrcoef(real, pred.flatten())[0, 1]
            return corr
        from sklearn.metrics import make_scorer
        myScoreFunc = {'RMSE': make_scorer(calcRMSE),
                       'Correlation coefficient': make_scorer(calcCorr)}
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        #Scores = cross_validate(testmodel, X, y, cv=cv, scoring=myScoreFunc,n_jobs=-1)
        Scores = cross_validate(testmodel, X, y, cv=cv, scoring=myScoreFunc)

        RMSETmp = Scores['test_RMSE'].mean()
        CORRTmP = Scores['test_Correlation coefficient'].mean()
        trainRMSETmp = Scores['train_RMSE'].mean()
        trainCORRTmP = Scores['train_Correlation coefficient'].mean()
        print(name,'test', RMSETmp, CORRTmP)
        print(name,'train',trainRMSETmp, trainCORRTmP)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=2)
    #for acuete tox predicton
    def metalPredict(self):
        rf = RandomForestRegressor(
            max_depth=20, random_state=0,
            n_estimators=134,min_samples_split=9,
            max_features=0.33
        )
        xg=xgboost.XGBRegressor(
            n_estimators=100,
            max_depth=7,
            gamma=0.0,
            colsample_bytree=1,
        )

        lgbm = LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=662,
            feature_fraction= 0.36875625810601625,
            bagging_fraction=0.39668072810414723,
            learning_rate=0.06,
            min_data_in_leaf=35,
            max_depth=27
        )

        rgf = RGFRegressor(
            max_leaf=1998,
            algorithm="RGF",
            test_interval=100,
            loss="LS",
            verbose=False,
            l2=0.187,
            min_samples_leaf=1
        )

        os.chdir(r'G:\マイドライブ\Data\tox_predict')
        #test
        #RMSE 10**1.048481123342563
        #Coor 0.6585531363438192
        #train
        #RMSE 10**0.7063184454700483
        #Coor 0.8793733707424819
        rf = RandomForestRegressor()
        #test
        #RMSE 10**1.109618572731106
        #Coor 0.6198791156669657
        #train
        #0.5647634395040055
        #0.9179571547567322


        df1 = pd.read_csv('metalMACCS.csv').set_index('CAS')
        df2 = pd.read_csv('metalECFP4.csv').set_index('CAS')
        for i,df in enumerate([df1,df2]) :
            names = ['fish_tox','daphnia_tox','Algae_tox']
            if i == 0:
                print('MACCS')
            else:
                print('ECFP4')
            for name in names:
                if name == 'fish_tox':
                    dftemp = df.drop(['daphnia_tox','Algae_tox'], axis=1)
                elif name== 'daphnia_tox':
                    dftemp = df.drop(['fish_tox','Algae_tox'], axis=1)
                elif name== 'Algae_tox':
                    dftemp = df.drop(['fish_tox','daphnia_tox'], axis=1)
                print(name)
                dftemp = dftemp.dropna()
                y = np.log10(dftemp[name])
                X = dftemp.iloc[:,0:-1]
                calcACC(rf, X=X, name=None, y=y)
    def metalPredictotherML(self):
        rf = RandomForestRegressor(
            max_depth=20, random_state=0,
            n_estimators=134,min_samples_split=9,
            max_features=0.33
        )
        xg=xgboost.XGBRegressor(
            n_estimators=9196,
            max_depth=497,
            gamma=0.0,
            colsample_bytree= 1
        )
        xg=xgboost.XGBRegressor(
            n_estimators=7534,
            max_depth=1000,
            gamma=2.0,
            colsample_bytree= 0.1
        )
        lgbm = LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=662,
            feature_fraction= 0.36875625810601625,
            bagging_fraction=0.39668072810414723,
            learning_rate=0.06,
            min_data_in_leaf=35,
            max_depth=27
        )

        rgf = RGFRegressor(
            max_leaf=1998,
            algorithm="RGF",
            test_interval=100,
            loss="LS",
            verbose=False,
            l2=0.187,
            min_samples_leaf=1
        )
        from sklearn import linear_model
        clf = linear_model.LogisticRegression(random_state=0)
        ml = LinearRegression(n_jobs=-1)
        [rf,xg,lgbm,rgf,ml]

        #test
        #RMSE 10**1.048481123342563
        #Coor 0.6585531363438192
        #train
        #RMSE 10**0.7063184454700483
        #Coor 0.8793733707424819
        rf = RandomForestRegressor()
        #test
        #RMSE 10**1.109618572731106
        #Coor 0.6198791156669657
        #train
        #0.5647634395040055
        #0.9179571547567322

        os.chdir(r'G:\マイドライブ\Data\tox_predict')
        df = pd.read_csv('metalMACCS.csv').set_index('CAS')
        dftemp = df.drop(['daphnia_tox','Algae_tox'], axis=1)
        dftemp = dftemp.dropna()
        y = np.log10(dftemp['fish_tox'])
        X = dftemp.iloc[:,0:-2]
        for i in [rf,xg,lgbm,rgf,ml]:
            calcACC(i, X=X, name=i, y=y)
    def forPredict(self):
        rf = RandomForestRegressor(
            max_depth=20, random_state=0,
            n_estimators=134,min_samples_split=9,
            max_features=0.33
        )
        os.chdir(r'G:\マイドライブ\Data\tox_predict')

        #predictCAS=['100-02-7','10102-18-8','10141-00-1','106-44-5','107-18-6','1071-83-6','108-05-4','108-46-3','110-80-5','118-96-7','122008-85-9','131-17-9','13593-03-8','141-66-2','151-50-8','177256-67-6','1918-02-1','918-16-7','1983-10-4','2425-06-1','2545-60-0','260-94-6','26027-38-3','26225-79-6','2893-78-9','300-76-5','31557-34-3','3383-96-8','34123-59-6','39515-41-8','51235-04-2','55-38-9','57-68-1','608-73-1','63-25-2','64-17-5','6515-38-4','66230-04-4','68-12-2','68951-67-7','70124-77-5','71-36-3','7173-51-5','74-83-9','74223-64-6','7446-20-0','7446-70-0','7646-79-9','7681-52-9','76930-58-0','7758-98-7','7782-50-5','7783-06-4','8003-34-7','87-65-0','877-43-0','9002-92-0','91-20-3','91465-08-6','94-09-7']

        predictCAS = ['10022-68-1', '10043-52-4', '10045-89-3', '10099-74-8', '10124-43-3', '107-64-2', '1303-96-4', '1305-62-0', '1306-23-6', '1314-98-3', '1332-40-7', '13464-37-4', '142-71-2', '14338-32-0', '144550-36-7', '16903-35-8', '23149-52-2', '24307-26-4', '25606-41-1', '301-04-2', '3653-48-3', '37262-61-6', '3766-27-6', '38641-94-0', '43222-48-6', '541-09-3', '56-35-9', '57213-69-1', '6062-26-6', '6266-23-5', '631-61-8', '6369-97-7', '6515-38-4', '66330-88-9', '69-57-8', '7173-51-5', '71751-41-2', '7439-96-5', '7439-97-6', '7440-02-0', '7631-90-5', '7646-85-7', '7733-02-0', '7758-95-4', '7773-06-0', '7782-49-2', '7783-90-6', '85-34-7']
        predictCAS = ['10102-18-8','11067-82-6','11121-48-5','121-54-0','128-44-9','1306-19-0','2164-07-0','2758-42-1','333-20-0','39148-24-8','497-19-8','52722-38-0','532-32-1','55256-32-1','56-35-9','563-68-8','7758-19-2','7784-46-5','8027-85-8','83601-83-6']
        predictCAS = ['10034-81-8', '12264-18-5', '123-46-6', '1302-42-7', '1313-27-5', '1333-82-0',
         '13826-65-8', '13939-25-8', '144-55-8', '145-73-3', '15339-36-3', '2465-27-2', '2787-91-9',
         '301-04-2', '3567-25-7', '4779-86-6', '569-64-2', '631-61-8', '6980-18-3', '7250-67-1',
         '7440-62-2', '7487-94-7', '7631-99-4', '76930-58-0', '7775-19-1', '7790-86-5', '7803-57-8',
         '79-57-2', '8065-48-3', '81510-83-0']
        df = pd.read_csv('metalMACCS.csv').set_index('CAS').drop(['canonical_smiles'],axis=1)
        names = ['fish_tox','daphnia_tox','Algae_tox']
        for name in names:
            if name == 'fish_tox':
                dftemp = df.drop(['daphnia_tox','Algae_tox'], axis=1)
            elif name== 'daphnia_tox':
                dftemp = df.drop(['fish_tox','Algae_tox'], axis=1)
            elif name== 'Algae_tox':
                dftemp = df.drop(['fish_tox','daphnia_tox'], axis=1)
            else:
                pass
            print(name)
            name ='fish_tox'
            y= np.log10(dftemp[name])
            X = dftemp.iloc[:,0:-1]

            dfTest= dftemp.loc[predictCAS]
            #dfTest = dfTest.dropna()
            test_y= dfTest[name]
            test_X = dfTest.iloc[:,0:-1]

            dfTrain = dftemp.drop(index=dfTest.index)
            dfTrain = dfTrain.dropna()
            train_y = np.log10(dfTrain[name])
            train_X = dfTrain.drop(name,axis=1)
            train_yInv = 1/ dfTrain[name]
            rf.fit(train_X,train_yInv)
            train_yNormal = dfTrain[name]
            rf.fit(train_X,train_yNormal)
            train_X = dfTrain.iloc[:,0:-1]
            rf.fit(train_X,train_y)

            predict= 10**(rf.predict(test_X))
            for i,j in zip(test_X.index,predict):
                print(i,j)
            feature = rf.feature_importances_
            f = pd.DataFrame({'number': range(0, len(feature)),
                              'feature': feature[:]})
            dpi=300
            plt.bar(f['number'], f['feature']*100,color='#377eb8')
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.savefig(r'C:\onedrive\01Doctor\論文\和文論文\kiyonorm.tiff', bbox_inches='tight', dpi=dpi)
            plt.rcParams['font.family'] = 'Times New Roman'
            igfont = {'family': 'Times New Roman'}
            plt.ylabel('Ecotoxicity [mg/L]', **igfont)
            plt.xlabel('The bit of MACCS Keys', **igfont)

            plt.show()
            f['feature'] = f['feature']*100
            f2 = f.sort_values('feature', ascending=False)
            f3 = f2.loc[:, 'number']

            featureInverse =


            morgan_turples = ((taxol, bit, bitI_morgan) for bit in list(bitI_morgan.keys())[:12])
            Draw.DrawMorganBits(morgan_turples, molsPerRow=4,
                                legends=['bit: ' + str(x) for x in list(bitI_morgan.keys())[:12]])

    def sepData(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        for train_index, test_index in cv.split(X, y):
            print(train_index,test_index)
        test_index = [1,10,12,15,21,37,75,76,78,90,96,107,135,141,144,159,186,188
,198,220,233,235,236,240,245,272,282,289,306,313,315,317,326,329,339,345
,346,351,352,354,382,387,405,414,420,430,436,469]
        dfReset = df.reset_index().dropna()
        dfTest= dfReset.loc[test_index].dropna()
        dfTest = dfTest.set_index('CAS')
        test_y= dfTest['tox_median']
        test_X = dfTest.iloc[:,0:167]

        dfTrain = df.drop(index=dfTest.index)
        train_y= np.log10(dfTrain['tox_median'])
        train_X = dfTrain.iloc[:,0:167]
        rf.fit(train_X,train_y)
        predict= 10**(rf.predict(test_X))
        for i in predict:
            print(i)
        for i in dfTest.index:
            print(i)
        for i in test_y:
            print(test_y)
    def stacklearning(self):
        class sparseNorm(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                from sklearn import preprocessing
                Y = preprocessing.normalize(sp.sparse.csc_matrix(X.values))
                return Y
        fm = sgd.FMRegression(
            n_iter=4743,
            init_stdev=0.1,
            rank=100,
            l2_reg_w=0,
            l2_reg_V=0,
            step_size=0.1,
        )
        fm = sgd.FMRegression(
            n_iter=9943,
            init_stdev=0.1,
            rank=219,
            l2_reg_w=0,
            l2_reg_V=0.06454,
            step_size=0.1,
        )
        pipe = make_pipeline(sparseNorm(), fm)
        calcACC(pipe, X=X2)

        xgb = xgboost.XGBRegressor(
                    n_estimators=100,
                    max_depth=7,
                    gamma=0,
                    colsample_bytree=0.1
                )
        lgbm = LGBMRegressor(
            boosting_type='gbdt', num_leaves=367,
            learning_rate=0.06,feature_fraction=0.14,
            max_depth=28, min_data_in_leaf=8
        )
        rgf = RGFRegressor(
            max_leaf=1211, algorithm="RGF", test_interval=100,
            loss="LS", verbose=False, l2=0.93,
            min_samples_leaf=2
        )
        rf = RandomForestRegressor(
            max_depth=20, random_state=0,
            n_estimators=56,min_samples_split=2,
            max_features=0.21
        )
        rf = RandomForestRegressor()
        ext = ExtraTreesRegressor(
            n_estimators=384,max_features= 2228,
            min_samples_split= 0.01,max_depth= 856,
            min_samples_leaf= 1
        )
        svr = SVR(
            gamma=9.5367431640625e-07,
            epsilon=0.0009765625,
            C= 2048.0
        )

        #test combination
        desNew = make_pipeline(extdescriptorNew(),rf)
        morNew = make_pipeline(extMorganNew(),rf)
        kotNew = make_pipeline(extklekotaTothNew(),rf)
        macNew = make_pipeline(extMACCSNew(),rf)

        desMac = make_pipeline(extDescriptorMACCS(),rf)
        morMac = make_pipeline(extMorganMACCS(),rf)
        kotMac = make_pipeline(extKlekotaTothMACCS(),rf)

        morKotNew = make_pipeline(extMorganKlekotaTothNew(),rf)
        des = make_pipeline(extOnlyDescriptor(),rf)
        mor = make_pipeline(extOnlyMorgan(),rf)
        kot = make_pipeline(extOnlyklekotaToth(),rf)
        mac = make_pipeline(extOnlyMACCS(),rf)
        all = make_pipeline(extAll(),rf)
        allwithoutNew = make_pipeline(extAllwithoutNew(),rf)
        allwithoutMaccs = make_pipeline(extAllwithoutMaccs(),rf)
        allwithoutDes = make_pipeline(extAllwithoutDescriptor(),rf)

        testDic = {"Desc+New":desNew,"Mor+New":morNew,"kot+New":kotNew,"MACCS+New":macNew,"Des+MAC":desMac,"Morgan+Maccs":morMac,"Kot+MACCS":kotMac,"mor+kot+New":morKotNew,
        "descriptor":des,"morgan":mor,"kot":kot,"MACCS":mac,"All":all,"All without "
                                                                      "new":allwithoutNew,
                   "All without MACCS":allwithoutMaccs,"All without Des":allwithoutDes}

        #10fold
        cv = KFold(n_splits=10, shuffle=True, random_state=0)

        #Fingerprinttest
        resultDic={}
        resultDic2={}
        for name,model in testDic.items():
            #model = StackingRegressor(regressors=[name], meta_regressor=rf,verbose=1)
            #calcACC(model,X=X,y=y2,name=name)

            Scores = cross_validate(model, X2, y2, cv=cv,scoring=myScoreFunc)
            RMSETmp = Scores['test_RMSE'].mean()
            CORRTmP = Scores['test_Correlation coefficient'].mean()
            resultDic.update({name:[RMSETmp,CORRTmP]})
            print(name,RMSETmp,CORRTmP)

        #stacking
        alldata = make_pipeline(extAll())
        # random forest
        #1.1546 0.70905
        stack = StackingRegressor(regressors=[alldata], meta_regressor=rf,verbose=1)

        # Light Gradient boosting
        # 1.160732 0.703776
        testmodel = StackingRegressor(regressors=[alldata], meta_regressor=lgbm,verbose=1)

        # XGboost
        # 1.1839805 0.689571
        testmodel = StackingRegressor(regressors=[alldata], meta_regressor=xgb,verbose=1)

        # Regularized greedily forest
        # 1.17050 0.6992
        testmodel = StackingRegressor(regressors=[alldata], meta_regressor=rgf,verbose=1)

        #pls 22.808047774809697 0.6410026452910016 i=4
        for i in np.arange(3,11,1):
            pls = PLSRegression(n_components=i)
            testmodel = StackingRegressor(regressors=[alldata], meta_regressor=pls,verbose=0)
            calcACC(testmodel)
        pls = PLSRegression(n_components=4)

        #SVR
        svr = SVR(gamma=9.5367431640625/10000000,C=1559.4918100725592,
                  epsilon=0.0009765625,)
        svr = SVR(kernel='rbf',gamma=9.5367431640625e-07,epsilon=0.0009765625,C=2048.0)

        testmodel = StackingRegressor(regressors=[alldata], meta_regressor=svr, verbose=1)
        calcACC(svr)

        #Extratree  1.157420824123527 0.7061010221224269
        testmodel = StackingRegressor(regressors=[alldata], meta_regressor=ext, verbose=1)
        calcACC(testmodel)

        #k-NN
        nbrs = KNeighborsRegressor(3)

        ##Linear regressions
        #Stochastic Gradient Descenta
        sgd = SGDRegressor(max_iter=1000)
        # Ridge
        for i in [1,10,100,1000]:
            ridge = Ridge(alpha=i)
            calcACC(ridge)
        ridge = Ridge(alpha=45.50940042350705)
        calcACC(ridge)
        # multiple linear
        lin = make_pipeline(forlinear(),LinearRegression(n_jobs=-1))
        calcACC(lin)



        #stacking
        #0.69
        testmodel = StackingRegressor(regressors=[alldata,nbrs,all], meta_regressor=rf,verbose=1)
        #1.1532 0.70926
        testmodel = StackingRegressor(regressors=[alldata,nbrs,all,xgb,lgbm,rgf], meta_regressor=rf,
                              verbose=1)
        #1.16420 0.7041
        testmodel = StackingRegressor(regressors=[alldata,alldata,all], meta_regressor=rf,verbose=1)
        #1.16379 0.7044
        stack1 = StackingRegressor(regressors=[alldata,nbrs,all,xgb,lgbm,rgf], meta_regressor=rf,verbose=1)
        testmodel  = StackingRegressor(regressors=[alldata,stack1,stack1], meta_regressor=rf,verbose=1)
        #1.1535496740699531 0.7108839199109559
        pcaFeature = make_pipeline(extPCA())
        testmodel = StackingRegressor(regressors=[pcaFeature,alldata,nbrs,rf,xgb,lgbm,rgf]
                                      ,meta_regressor=rf,verbose=1)
        #1.181801005432221 0.6889745579620922
        testmodel = StackingRegressor(regressors=[pcaFeature,alldata,nbrs,rf,xgb,lgbm,rgf]
                                      ,meta_regressor=lgbm,verbose=1)
        #0.70613
        testmodel = StackingRegressor(regressors=[pcaFeature,alldata,nbrs,rf,xgb,lgbm,rgf,ext]
                                      ,meta_regressor=xgb,verbose=1)
        #0.71641717
        testmodel = StackingRegressor(regressors=[pcaFeature,alldata,nbrs,rf,xgb,lgbm,rgf,ext]
                                      ,meta_regressor=rf,verbose=1)
        #0.7146922
        testmodel = StackingRegressor(regressors=[pcaFeature,alldata,nbrs,ridge,rf,xgb,lgbm,rgf,ext]
                                      ,meta_regressor=rf,verbose=1)

        #new features
        pcaFeature = make_pipeline(extPCA())

        #old
        pipe1 = make_pipeline(extMACCS(), rf)
        pipe2 = make_pipeline(extMorgan(), rf)
        pipe3 = make_pipeline(extDescriptor(), rf)

        pipe4 = make_pipeline(extPCA(), rgf)
        pipe7 =make_pipeline(extDescriptor(), rgf)
        pipe8 =make_pipeline(extDescriptor(), rgf)

        xgb = xgboost.XGBRegressor()
        nbrs = KNeighborsRegressor(2)
        svr = SVR(gamma='auto',kernel='linear')

        pls = PLSRegression(n_components=4)

        extMACCSdata = make_pipeline(extMACCS())

        nbrsPipe = make_pipeline(extMorgan(), nbrs)
        pipe6 = make_pipeline(extMACCS(), rgf)
        alldata = make_pipeline(extAll())
        ave = extAverage()
        withoutdesc =  make_pipeline(extMACCS())

        meta = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=400)
        #stack1 = StackingRegressor(regressors=[rgf, nbrs, alldata], meta_regressor=rgf, verbose=1)

        #0.70
        stack = StackingRegressor(regressors=[pipe1,pipe2,pipe3,xgb,lgbm,rgf,rf], meta_regressor=ave, verbose=1)

        #stack2 = StackingRegressor(regressors=[stack1,nbrs, svr,pls,rgf], meta_regressor=lgbm, verbose=1)

        #0.69######################
        stack1 = StackingRegressor(regressors=[pipe1,pipe2,pipe3], meta_regressor=rf, verbose=1)
        #0.70
        stack2 = StackingRegressor(regressors=[stack1,alldata,rgf,lgbm,xgb], meta_regressor=rf,verbose=1)

        #0.71
        stack3 = StackingRegressor(regressors=[stack2,pipe1], meta_regressor=ave, verbose=1)
        ###########################
        ###########################
        stack1 = StackingRegressor(regressors=[pipe1,pipe2,pipe3], meta_regressor=rf, verbose=1)
        stack2 = StackingRegressor(regressors=[stack1,withoutdesc,lgbm,rgf], meta_regressor=rf,verbose=1)
        stack3 = StackingRegressor(regressors=[stack2,pipe1,xgb], meta_regressor=ave, verbose=1)
        ###########################

        #stackingwithknn
        stack1 = StackingRegressor(regressors=[pipe1,pipe2,pipe3], meta_regressor=rf, verbose=1)
        stack2 = StackingRegressor(regressors=[stack1,nbrs,pipe1], meta_regressor=rf, verbose=1)


        #stack3 = StackingRegressor(regressors=[rgf, nbrs, alldata], meta_regressor=ave, verbose=1)

        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        St1Scores = cross_validate(stack1,X,y,cv=cv)
        St1Scores['test_score'].mean()**(1/2)

        St2Scores = cross_validate(stack2,X,y,cv=cv)
        St2Scores['test_score'].mean()**(1/2)

        St3Scores = cross_validate(stack3,X,y,cv=cv)
        St3Scores['test_score'].mean()**(1/2)

        stackScore = cross_validate(stack, X, y, cv=cv)
        stackScore['test_score'].mean()**(1/2)

        lgbmScores =cross_validate(lgbm,X,y,cv=cv)
        lgbmScores['test_score'].mean()**(1/2)

        rgfScores = cross_validate(rgf,X,y,cv=cv)
        rgfScores['test_score'].mean()**(1/2)

        RFScores = cross_validate(rf,X,y,cv=cv)
        RFScores['test_score'].mean()**(1/2)

        scores = cross_validate(stack2,X,y,cv=cv)
        scores['test_score'].mean()**(1/2)
        print("R^2 Score: %0.2f (+/- %0.2f) [%s]" % (scores['test_score'].mean(), scores['test_score'].std(), 'stacking'))

        stack3.fit(X, y)
        y_pred = stack3.predict(X_train)
        y_val = stack3.predict(X_test)
        #stack3.score(X_train, y_train)
        exX = preprocess(extractDf, changeList)
        valy =  (10 **(stack3.predict(exX))).tolist()
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred, y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val, y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred, y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val, y_test))

        stack1.fit(X, y)
        valy =  (10 **(stack1.predict(exX))).tolist()

        sgd.fit(X,y)
        valy =  (10 **(sgd.predict(exX))).tolist()

        rgfpipe = make_pipeline(extMACCS(), rf)
        rgf.fit(X,y)
        valy =  (10 **(rgf.predict(exX))).tolist()

        nbrs.fit(X,y)
        valy =  (10 **(nbrs.predict(exX))).tolist()

        pipe = make_pipeline(extMACCS(), rf)
        pipe.fit(X,y)
        valy =  (10 **(pipe.predict(exX))).tolist()


        rf.fit(X, y)
        y_pred = rf.predict(X_train)
        y_val = rf.predict(X_test)
        exX = preprocess(extractDf, changeList)
        valy =  (10 **(rf.predict(exX))).tolist()
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred, y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val, y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred, y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val, y_test))

        lgbm.fit(X, y)
        #y_pred = pipe1.predict(X_train)
        #y_val = pipe1.predict(X_test)
        exX = preprocess(extractDf, changeList)
        valy =  (10 **(lgbm.predict(exX))).tolist()
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred, y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val, y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred, y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val, y_test))

if __name__ == '__main__':
    ejectCAS = ['10124-36-4', '108-88-3', '111991-09-4', '116-29-0', '120-12-7', '126833-17-8',
                '13171-21-6',
                '1333-82-0', '137-30-4', '148-79-8', '1582-09-8', '1610-18-0', '2058-46-0',
                '2104-64-5',
                '21725-46-2',
                '2303-17-5', '25311-71-1', '25812-30-0', '298-00-0', '298-04-4', '314-40-9',
                '330-54-1',
                '4170-30-3',
                '4717-38-8', '50-00-0', '52645-53-1', '55406-53-6', '56-35-9', '56-38-2',
                '60207-90-1', '6051-87-2',
                '62-53-3', '6317-18-6', '69-72-7', '7440-02-0', '7447-40-7', '7722-84-1',
                '7733-02-0', '7758-94-3',
                '80844-07-1', '82657-04-3', '84852-15-3', '86-73-7', '9016-45-9', '99-35-4']

    #os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
    df =pd.read_csv(r'fishMorganMACCS.csv')
    #df2=pd.read_csv('chronicMACCSKeys_tanimoto.csv')
    #df2 = df2.drop(ejectCAS,axis=1).set_index('CAS').dropna(how='all', axis=1)
    baseDf = df
    extractDf =  df[df['CAS'].isin(ejectCAS)]
    exy = extractDf['logTox']
    exX = extractDf.drop(columns=['CAS','toxValue','logTox'])

    df = df[~df['CAS'].isin(ejectCAS)]
    #df = df.set_index('CAS')
    #df = pd.concat([df,df2],axis=1, join_axes=[df.index]).reset_index()
    y = df['logTox']
    # for name in df.columns:
    #     a = re.search('ox', name)
    #     if a != None:
    #         print(name)
    y2= df['toxValue']
    #dropList = ['CAS','toxValue','logTox','HDonor', 'HAcceptors', 'AromaticHeterocycles', 'AromaticCarbocycles', 'FractionCSP3']
    dropList = ['CAS','toxValue','logTox']
    X = df.drop(columns=dropList)
    #Normalize

    def normalize(X):
        changeList = []
        for i,name in enumerate(X.columns):
            if i <679:
                changeList.append((0,1))
            elif i > 692:
                changeList.append((0,1))
            else:
                #try:
                #name = float(name)
                #except:
                std =X[name].std()
                mean = X[name].mean()
                if std == 0:
                    pass
                else:
                    X[name] = X[name].apply(lambda x: ((x - mean) * 1 / std + 0))
                changeList.append((mean, std))
        return X, changeList

    def preprocess(extractDf,changeList):
        for i, (name, calc) in enumerate(zip(extractDf.columns,changeList)):
            #print(i, name,calc[0],calc[1])
            if calc[0] ==0:
                pass
            else:
                extractDf[name] = extractDf[name].apply(lambda x: ((x - calc[0]) * 1 / calc[1] + 0))
        dropList = ['CAS','toxValue','logTox']
        extractDf = extractDf.drop(columns=dropList)
        return extractDf