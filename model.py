# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging

from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='British_Airways_Improved.log', force=True)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Load dataset
logging.info('Dataset Imported Successfully')

df = pd.read_csv("Data/customer_booking.csv", encoding="ISO-8859-1")
df = df.sample(frac=1,random_state=42)

#Segregating the Numerical Column from the dataset
numerical_columns = df.select_dtypes(include=['number'])
categorical_columns = df.select_dtypes(include=['object'])

from collections import OrderedDict

stats = []

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    stats.append(OrderedDict({
        'Feature': col,
        'Mean': df[col].mean(),
        'Median': df[col].median(),
        'Q1': df[col].quantile(0.25),
        'Q3': df[col].quantile(0.75),
        'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
        'Max': df[col].max(),
        'Min': df[col].min(),
        'Skewness': df[col].skew(),
        'Kurtosis': df[col].kurt(),
        'Variance': df[col].var()
    }))

report = pd.DataFrame(stats)

# Outlier flag
outlier_label = []
for col in report['Feature']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    LW = Q1 - 1.5 * IQR
    UW = Q3 + 1.5 * IQR
    outliers = df[(df[col] < LW) | (df[col] > UW)]
    outlier_label.append('Has Outliers' if not outliers.empty else 'No Outliers')

report['Outlier Comment'] = outlier_label
print(report)


# Feature engineering
def hour_bin(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'night'

df['stay_per_passenger'] = df['length_of_stay'] / (df['num_passengers'] + 1)
df['lead_time_ratio'] = df['purchase_lead'] / (df['length_of_stay'] + 1)
df['is_weekend_flight'] = df['flight_day'].isin(['Saturday', 'Sunday']).astype(int)
df['flight_period'] = df['flight_hour'].apply(hour_bin)

# CRITICAL: Outlier treatment BEFORE train-test split (as in original)
def replace_outliers_with_median(data, exclude_cols=['booking_complete']):
    for col in data.select_dtypes(include='number').columns:
        if col not in exclude_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            if outliers.sum() > 0:
                replacement = data[col].median()
                data.loc[outliers, col] = replacement
                print(f"Replaced {outliers.sum()} outliers in '{col}' with median")
    return data

df = replace_outliers_with_median(df)

from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if col != 'booking_complete':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Encoded column: {col}")


# Prepare features and target (same feature selection as original)
X = df.drop(columns=['booking_complete', 'route', 'booking_origin'], errors='ignore',axis = 1)
y = df['booking_complete']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=42)
X_train, y_train = smt.fit_resample(X_train, y_train)

from sklearn.preprocessing import MinMaxScaler,RobustScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

CBC = CatBoostClassifier(verbose=False).fit(X_train,y_train)
y_pred_CBC = CBC.predict(X_test)
print("The Catboost Classifier Accuracy:", round(accuracy_score(y_test, y_pred_CBC) * 100, 2), "%")

