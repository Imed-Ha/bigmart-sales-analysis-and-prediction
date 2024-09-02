import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data):
    # Copy the data to avoid modifying the original DataFrame
    data = data.copy()

    # Handle missing values based on the notebook logic
    data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
    data['Outlet_Size'] = data['Outlet_Size'].fillna('Medium')

    # Feature engineering based on the notebook logic
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'
    })
    data['New_Item_Type'] = data['Item_Identifier'].apply(lambda x: x[:2])
    data['New_Item_Type'] = data['New_Item_Type'].map({
        'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'
    })

    # Drop unnecessary columns as identified in the notebook
    data = data.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

    # Encode categorical variables based on the notebook logic
    label_encoder = LabelEncoder()
    data['Item_Fat_Content'] = label_encoder.fit_transform(data['Item_Fat_Content'])

    # One-hot encoding for other categorical variables
    data = pd.get_dummies(data, columns=['Item_Type', 'Outlet_Type'], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data
