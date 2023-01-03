import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import dataset
dataset = pd.read_csv("./Data.csv")

# separating columns
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# handling missing numerical data
si = SimpleImputer(missing_values=np.nan, strategy='mean')
si.fit(x[:, 1:3])
x[:, 1:3] = si.transform(x[:, 1:3])

# encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# encoding dependent variables
le = LabelEncoder()
y = le.fit_transform(y)

# splitting dataset to training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
