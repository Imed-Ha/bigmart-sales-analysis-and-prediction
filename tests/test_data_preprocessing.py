from scripts.data_preprocessing import load_data, preprocess_data
import pandas as pd

def test_load_data():
    df = load_data('data/test.csv')
    assert isinstance(df, pd.DataFrame)

def test_preprocess_data():
    df = pd.DataFrame({
        'Item_Weight': [10, None, 20],
        'Outlet_Size': [None, 'Small', 'Medium']
    })
    processed_df = preprocess_data(df)
    assert processed_df['Item_Weight'].isnull().sum() == 0
    assert processed_df['Outlet_Size'].isnull().sum() == 0
