import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Make a copy to avoid modifying the original DataFrame
    data = data.copy()

    # Handling missing values
    data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
    data['Outlet_Size'] = data['Outlet_Size'].fillna('Medium')

    # Feature engineering
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
    data['New_Item_Type'] = data['Item_Identifier'].apply(lambda x: x[:2])
    data['New_Item_Type'] = data['New_Item_Type'].map({
        'FD': 'Food',
        'NC': 'Non-Consumable',
        'DR': 'Drinks'
    })
    data['New_Item_Type'] = data['New_Item_Type'].fillna('Other')

    # Encoding categorical variables
    data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'New_Item_Type'], drop_first=True)
    
    # Normalize or scale features if necessary
    scaler = StandardScaler()
    numeric_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    return data

if __name__ == "__main__":
    train_data = pd.read_csv('../data/raw/Train.csv')
    processed_data = preprocess_data(train_data)
    processed_data.to_csv('../data/processed/processed_train.csv', index=False)
