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

class(object):
    ejectCAS = ['10124-36-4', '108-88-3', '111991-09-4', '116-29-0', '120-12-7', '126833-17-8', '13171-21-6',
                        '1333-82-0', '137-30-4', '148-79-8', '1582-09-8', '1610-18-0', '2058-46-0', '2104-64-5',
                        '21725-46-2',
                        '2303-17-5', '25311-71-1', '25812-30-0', '298-00-0', '298-04-4', '314-40-9', '330-54-1',
                        '4170-30-3',
                        '4717-38-8', '50-00-0', '52645-53-1', '55406-53-6', '56-35-9', '56-38-2', '60207-90-1', '6051-87-2',
                        '62-53-3', '6317-18-6', '69-72-7', '7440-02-0', '7447-40-7', '7722-84-1', '7733-02-0', '7758-94-3',
                        '80844-07-1', '82657-04-3', '84852-15-3', '86-73-7', '9016-45-9', '99-35-4']
    os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
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

    def calcRMSE(pred,real):
      RMSE = (np.sum((pred-real.tolist())**2)/len(pred))**(1/2)
      return RMSE
    def calcCorr(pred,real):
      corr = np.corrcoef(real, pred.flatten())[0,1]
      return corr

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
    def main(self):
        os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
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

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=2)

    def stacklearning(self):
        class extAverage(BaseEstimator, TransformerMixin,RegressorMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                result = np.average(X, axis = 1)
                return self

            def transform(self, X):
                return self

            def predict(self, X):
                print(X.shape)
                result = (X[:,0]+X[:,1]*2+X[:,2]*0.5)/3.5
                #result = np.average(X, axis = 1)
                return result
        class extAll(BaseEstimator, TransformerMixin,RegressorMixin):
            def __init__(self):
                pass
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return self
            def predict(self, X):
                maccs,morgan,descriptor,klekotaToth,newFingerprint=sepTables(X)
                descriptor = pd.concat([maccs,morgan,descriptor,klekotaToth,newFingerprint],axis=1)
                return descriptor

        class extMorgan(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                _,morgan,_,klekotaToth,newFingerprint=sepTables(X)
                morgan = pd.concat([morgan,klekotaToth,newFingerprint],axis=1)
                return morgan
        class extMACCS(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                maccs,morgan,_,klekotaToth,newFingerprint=sepTables(X)
                maccs = pd.concat([maccs,morgan,klekotaToth,newFingerprint],axis=1)

                return maccs
            def predict(self, X):
                maccs,morgan,descriptor,klekotaToth,newFingerprint=sepTables(X)
                descriptor = pd.concat([maccs,morgan,klekotaToth,newFingerprint],axis=1)
                return descriptor

        class extDescriptor(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                maccs,morgan,descriptor,klekotaToth,newFingerprint=sepTables(X)
                descriptor = pd.concat([maccs,morgan,descriptor,klekotaToth,newFingerprint],axis=1)
                return descriptor

        class extPCA(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                model = PCA(n_components=64)
                _,morgan,_,_=sepTables(X)
                morgan = morgan.reset_index().drop('index', axis=1)
                W = pd.DataFrame(model.fit_transform(X))
                W = pd.concat([morgan,W],axis=1)
                return W

        lgbm = LGBMRegressor(boosting_type='gbdt', num_leaves=367,
                             learning_rate=0.06,feature_fraction=0.14,
                             max_depth=28, min_data_in_leaf=8
                             )
        rgf = RGFRegressor(max_leaf=1211, algorithm="RGF", test_interval=100,
                           loss="LS", verbose=False, l2=0.93,
                           min_samples_leaf=2
                           )
        rf = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=56,min_samples_split=2,max_features=0.21)

        pipe1 = make_pipeline(extMACCS(), rf)
        pipe2 = make_pipeline(extMorgan(), rf)
        pipe3 = make_pipeline(extDescriptor(), rf)

        pipe4 = make_pipeline(extPCA(), rgf)
        pipe7 =make_pipeline(extDescriptor(), rgf)
        pipe8 =make_pipeline(extDescriptor(), rgf)

        xgb = xgboost.XGBRegressor()
        nbrs = KNeighborsRegressor(2)
        svr = SVR(gamma='auto',kernel='linear')
        sgd = SGDRegressor(max_iter=1000)
        pls = PLSRegression(n_components=3)
        ext = ExtraTreesRegressor(n_estimators=30,max_features= 20,min_samples_split= 5,max_depth= 50, min_samples_leaf= 5)

        nbrsPipe = make_pipeline(extMorgan(), nbrs)
        pipe6 = make_pipeline(extMACCS(), rgf)
        alldata = make_pipeline(extAll())
        ave = extAverage()
        withoutdesc =  make_pipeline(extMACCS())

        meta = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=400)
        #stack1 = StackingRegressor(regressors=[rgf, nbrs, alldata], meta_regressor=rgf, verbose=1)

        stack = StackingRegressor(regressors=[pipe1,pipe2,pipe3,xgb,lgbm,rgf,rf], meta_regressor=ave, verbose=1)

        #stack2 = StackingRegressor(regressors=[stack1,nbrs, svr,pls,rgf], meta_regressor=lgbm, verbose=1)

        #0.71######################
        stack1 = StackingRegressor(regressors=[pipe1,pipe2,pipe3], meta_regressor=rf, verbose=1)
        stack2 = StackingRegressor(regressors=[stack1,alldata,rgf,lgbm,xgb], meta_regressor=rf,verbose=1)
        stack3 = StackingRegressor(regressors=[stack2,pipe1], meta_regressor=ave, verbose=1)
        ###########################
        ##########################
        stack1 = StackingRegressor(regressors=[pipe1,pipe2,pipe3], meta_regressor=rf, verbose=1)
        stack2 = StackingRegressor(regressors=[stack1,withoutdesc,lgbm,rgf], meta_regressor=rf,verbose=1)
        stack3 = StackingRegressor(regressors=[stack2,pipe1,xgb], meta_regressor=ave, verbose=1)
        ###########################


        #stack3 = StackingRegressor(regressors=[rgf, nbrs, alldata], meta_regressor=ave, verbose=1)

        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        cv = KFold(n_splits=10, shuffle=True, random_state=0)

        St2Scores = cross_validate(stack2,X,y,cv=cv)
        St2Scores['test_score'].mean()**(1/2)
        St2Scores['test_score'].mean() ** (1 / 2)
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

