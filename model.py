import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
# Importing dataset
df = pd.read_csv('Loan_Data.csv')
data = df
data = data.dropna()
# transfer the data to another dataframe
train = data
# clean the data to get rid of the 3+ because that will not work in the training  
def clean_Dependents(x):
    if '3+' in x:
        return '3'
    if '0' in x:
        return '0'
    if '1' in x:
        return '1'
    if '2' in x:
        return '2'

train['Dependents'] = train['Dependents'].apply(clean_Dependents)
# turning string data into floats 

le_Gender = LabelEncoder()
train['Gender'] = le_Gender.fit_transform(train['Gender'])
# train["Gender"].unique()

le_Married = LabelEncoder()
train['Married'] = le_Married.fit_transform(train['Married'])
# train["Married"].unique()

# le_Dependents = LabelEncoder()
# train['Dependents'] = le_Dependents.fit_transform(train['Dependents'])
# train["Dependents"].unique()

le_Education = LabelEncoder()
train['Education'] = le_Education.fit_transform(train['Education'])
# train["Education"].unique()

le_Self_Employed = LabelEncoder()
train['Self_Employed'] = le_Self_Employed.fit_transform(train['Self_Employed'])
# train["Self_Employed"].unique()


le_Property_Area = LabelEncoder()
train['Property_Area'] = le_Property_Area.fit_transform(train['Property_Area'])
# train["Property_Area"].unique()


le_Loan_Status = LabelEncoder()
train['Loan_Status'] = le_Loan_Status.fit_transform(train['Loan_Status'])
# train["Loan_Status"].unique()

# drop the Loan_ID we dont need it 
train = train.drop("Loan_ID", axis=1)
# turn all the type into float64
import numpy as np
train = train.astype(np.float64)
# create X and y
X = train.drop("Loan_Status", axis=1)
y = train["Loan_Status"]

# split the data into training and testing getiing ready for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, 
                                                    random_state=42)
# using the sklean linear regression
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_y_pred = linear_reg.predict(X_test)

# save the parameters and model  to use when predicting 
import pickle
data = {"model": linear_reg, "le_Gender": le_Gender, "le_Married": le_Married, "le_Education": le_Education, "le_Self_Employed": le_Self_Employed, }
# data = {"model": linear_reg, "le_Gender": le_Gender, "le_Married": le_Married,  "le_Dependents": le_Dependents, "le_Education": le_Education, "le_Self_Employed": le_Self_Employed, }
with open('model.pkl', 'wb') as file:
    pickle.dump(data, file)

# open the saved model
with open('model.pkl', 'rb') as file:
    data = pickle.load(file)

linear_reg_loaded = data["model"]
le_Gender = data["le_Gender"]
le_Married = data["le_Married"]
# le_Dependents = data["le_Dependents"]
le_Education = data["le_Education"]
le_Self_Employed = data["le_Self_Employed"]

# country, edlevel, yearscode
entry = np.array([["Male", "Yes", '1', 'Graduate','No', 4583, 1508.0, 128.0, 360.0, 1.0, 0]])
entry[:, 0] = le_Gender.transform(entry[:,0])
entry[:, 1] = le_Married.transform(entry[:,1])
# entry[:, 2] = le_Dependents.transform(entry[:,2])
entry[:, 3] = le_Education.transform(entry[:, 3])
entry[:, 4] = le_Self_Employed.transform(entry[:, 4])
# entry[:, 10] = le_Property_Area.transform(entry[:, 10])

entry[:,1]

entry = entry.astype(float)
entry.shape,type(entry), entry.dtype
y_pred = linear_reg_loaded.predict(entry)
print(y_pred)
