from sklearn.impute import SimpleImputer
import pandas as pd

def train_imputer(housing, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    housing_numeric = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_numeric)
    
    return imputer
    
def impute_values(housing, imputer):
    housing_numeric = housing.drop('ocean_proximity', axis=1)
    transformed_feats = imputer.transform(housing_numeric)
    
    housing_train_pre = pd.DataFrame(transformed_feats, columns=housing_numeric.columns, index=housing_numeric.index)
    housing_train_pre['ocean_proximity'] = housing['ocean_proximity']
    
    return housing_train_pre