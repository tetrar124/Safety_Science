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
        #similarity = xdf.iloc[:,690:]
        #print(MACCS.shape,Morgan.shape,descriptors.shape)
        return MACCS,Morgan,descriptors

    def test(self):
        os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
        df =pd.read_csv('MorganMACCS.csv')
        #df2=pd.read_csv('chronicMACCSKeys_tanimoto.csv')
        #df2 = df2.drop(ejectCAS,axis=1).set_index('CAS').dropna(how='all', axis=1)
        baseDf = df
        extractDf =  df['CAS'].isin(ejectCAS)
        df = df[~df['CAS'].isin(ejectCAS)]
        #df = df.set_index('CAS')
        #df = pd.concat([df,df2],axis=1, join_axes=[df.index]).reset_index()
        y = df['logTox']
        #dropList = ['CAS','toxValue','logTox','HDonor', 'HAcceptors', 'AromaticHeterocycles', 'AromaticCarbocycles', 'FractionCSP3']
        dropList = ['CAS','toxValue','logTox']
        X = df.drop(columns=dropList)
        #Normalize
        for i,name in enumerate(X.columns):
            if i <679:
                pass
            elif i > 692:
                pass
            else:
                try:
                    name = float(name)
                except:
                    std =X[name].std()
                    mean = X[name].mean()
                    X[name] = X[name].apply(lambda x: ((x - mean) * 1 / std + 0))
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=2)


    def stacklearning(self):
        class extAverage(BaseEstimator, TransformerMixin,RegressorMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return self

            def predict(self, X):
                result = np.average(X, axis = 1)
                return result
        class extAll(BaseEstimator, TransformerMixin,RegressorMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return self

            def predict(self, X):
                return X

        class extMorgan(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                _,morgan,_=sepTables(X)
                return morgan
        class extMACCS(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                maccs,morgan,_=sepTables(X)
                maccs = pd.concat([morgan,maccs],axis=1)

                return maccs

        class extDescriptor(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                maccs,morgan,descriptor=sepTables(X)
                descriptor = pd.concat([morgan,descriptor],axis=1)
                descriptor = pd.concat([maccs,descriptor],axis=1)
                return descriptor

        class extPCA(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                model = PCA(n_components=64)
                _,morgan,_=sepTables(X)
                morgan = morgan.reset_index().drop('index', axis=1)
                W = pd.DataFrame(model.fit_transform(X))
                W = pd.concat([morgan,W],axis=1)
                return W

        lgbm = LGBMRegressor(boosting_type='gbdt', num_leaves= 60,learning_rate=0.06)
        rgf = RGFRegressor(max_leaf=1000, algorithm="RGF",test_interval=100, loss="LS",verbose=False,l2=1.0)
        rgf1 = RGFRegressor(max_leaf=1000, algorithm="RGF",test_interval=100, loss="LS",verbose=False,l2=1.0)
        rgf2 = RGFRegressor(max_leaf=1000, algorithm="RGF",test_interval=100, loss="LS",verbose=False,l2=1.0)
        rgf3 = RGFRegressor(max_leaf=1000, algorithm="RGF",test_interval=100, loss="LS",verbose=False,l2=1.0)
        rgf4 = RGFRegressor(max_leaf=1000, algorithm="RGF",test_interval=100, loss="LS",verbose=False,l2=1.0)

        pipe1 = make_pipeline(extMACCS(), rgf)
        pipe2 = make_pipeline(extMorgan(), rgf1)
        pipe3 = make_pipeline(extDescriptor(), rgf2)
        pipe4 = make_pipeline(extPCA(), rgf3)
        pipe7 =make_pipeline(extDescriptor(), rgf4)
        pipe8 =make_pipeline(extDescriptor(), rgf4)

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

        meta = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=400)
        #stack1 = StackingRegressor(regressors=[rgf, nbrs, alldata], meta_regressor=rgf, verbose=1)
        stack = StackingRegressor(regressors=[pipe1,pipe2,pipe3,xgb,lgbm,], meta_regressor=ave, verbose=1)

        #stack2 = StackingRegressor(regressors=[stack1,nbrs, svr,pls,rgf], meta_regressor=lgbm, verbose=1)
        stack1 = StackingRegressor(regressors=[pipe1,pipe2,pipe3], meta_regressor=rgf, verbose=1)
        stack2 = StackingRegressor(regressors=[stack1,alldata,nbrsPipe], meta_regressor=ext,verbose=1)


        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        cv = KFold(n_splits=10, shuffle=True, random_state=0)


        stackScore = cross_validate(stack, X, y, cv=cv)
        stackScore['test_score'].mean()

        rgfScores = cross_validate(rgf,X,y,cv=cv)
        rgfScores['test_score'].mean()

        scores = cross_validate(stack2,X,y,cv=cv)
        scores['test_score'].mean()
        print("R^2 Score: %0.2f (+/- %0.2f) [%s]" % (scores['test_score'].mean(), scores['test_score'].std(), 'stacking'))

        stack2.fit(X_train, y_train)
        y_pred = stack2.predict(X_train)
        y_val = stack2.predict(X_test)
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred, y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val, y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred, y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val, y_test))

        rgf.fit(X_train, y_train)
        y_pred = rgf.predict(X_train)
        y_val = rgf.predict(X_test)
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred, y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val, y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred, y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val, y_test))

        pipe1.fit(X_train, y_train)
        y_pred = pipe1.predict(X_train)
        y_val = pipe1.predict(X_test)
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred, y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val, y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred, y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val, y_test))


        cols = np.arange(1,550,1).tolist()
        cols = X.columns.tolist()
        cols = [1,2,3]
        # Initializing Classifiers
        reg1 = Ridge(random_state=1)
        #reg2 = ExtraTreesRegressor()
        reg2 = ExtraTreesRegressor(n_estimators=50,max_features= 50,min_samples_split= 5,max_depth= 50, min_samples_leaf= 5)
        reg3 = SVR(gamma='auto',kernel='linear')
        reg4 = LGBMRegressor(boosting_type='gbdt', num_leaves= 60,learning_rate=0.06)
        pls = PLSRegression(n_components=3)
        pipe1 = make_pipeline(ColumnSelector(cols=cols), ExtraTreesRegressor(n_estimators=50))
        #linear =SGDRegressor(max_iter=1000)
        rgf = RGFRegressor(max_leaf=1000, algorithm="RGF",test_interval=100, loss="LS",verbose=False,l2=1.0)
        nbrs = KNeighborsRegressor(2)
        pipe2 = make_pipeline(ColumnSelector(cols=cols), KNeighborsRegressor(31))

        meta = ExtraTreesRegressor(n_estimators=50,max_features= 7,min_samples_split= 5,max_depth= 50, min_samples_leaf= 5)

        stackReg = StackingRegressor(regressors=[reg1,reg2, reg3,pipe1,pls,nbrs,rgf], meta_regressor=meta,verbose=1)
        stackReg.fit(X_train, y_train)
        y_pred = stackReg.predict(X_train)
        y_val = stackReg.predict(X_test)
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred,y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val,y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred,y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val,y_test))

        rgf.fit(X_train, y_train)
        y_pred = reg4.predict(X_train)
        y_val = reg4.predict(X_test)
        print("Root Mean Squared Error train: %.4f" % calcRMSE(y_pred,y_train))
        print("Root Mean Squared Error test: %.4f" % calcRMSE(y_val,y_test))
        print('Correlation Coefficient train: %.4f' % calcCorr(y_pred,y_train))
        print('Correlation Coefficient test: %.4f' % calcCorr(y_val,y_test))