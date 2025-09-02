import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Read the dataset
diabetes=pd.read_csv('diabetes.csv')

cols_with_zero=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
diabetes[cols_with_zero]=diabetes[cols_with_zero].replace(0,np.nan)

# Impute missing values
imputer=SimpleImputer(strategy='median')
diabetes[cols_with_zero]=imputer.fit_transform(diabetes[cols_with_zero])

# check if class is imbalace
X=diabetes.drop('Outcome',axis=1)
y=diabetes['Outcome']

#apply smote for handle class imbalance
smote=SMOTE()
X_resampled,y_resampled=smote.fit_resample(X,y)

# Split the data
X_train,X_test,y_train,y_test=train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=2,stratify=y_resampled)

# Scale the features
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)

# Train the model
model=LogisticRegression()
model.fit(X_train_scaler,y_train)

# Evaluate the model
y_pred=model.predict(X_test_scaler)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# Save the model
with open('app/diabetes_model.pkl','wb') as f:
    pickle.dump((scaler,model),f)
print("Model saved successfully.")