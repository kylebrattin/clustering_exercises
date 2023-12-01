# imports
import pandas as pd
import numpy as np
from env import get_db_url
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split




def get_zillow():
    '''This function imports zillow 2017 data from MySql codeup server and creates a csv
    
    argument: df
    
    returns: zillow df'''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        select * 
        from properties_2017
        left join propertylandusetype using (propertylandusetypeid)
        join predictions_2017 using (parcelid)
        left join airconditioningtype using (airconditioningtypeid)
        LEFT JOIN architecturalstyletype using (architecturalstyletypeid)
        LEFT JOIN buildingclasstype using (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype using (heatingorsystemtypeid)
        LEFT JOIN storytype using (storytypeid)
        LEFT JOIN typeconstructiontype using (typeconstructiontypeid)
        LEFT JOIN unique_properties using (parcelid);
        """
        connection = get_db_url("zillow")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df


def check_columns(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.
​
    Args:
    - df: pandas dataframe
​
    Returns:
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].isna().mean(),
                df[column].dtype
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "Proportion of Null Values",
            "dtype"
        ],
    ).sort_values(by="Number of Unique Values")

def missing_by_row(df):
    '''
    prints out a report of how many columns have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of null values, and proportion of null values
        data.append(
            [
                column,
                df[column].isna().sum(),
                df[column].isna().mean() 
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Null Values', and 'Proportion of Null Values'
    # Sort the resulting dataframe by the 'Proportion of Null Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Null Values",
            "Proportion of Null Values" 
        ],
    ).sort_values(by='Proportion of Null Values')



#Define function to drop columns/rows based on proportion of nulls
def null_dropper(df, prop_required_column, prop_required_row):
    
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    row_threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    
    return df









def split_data(df):
    '''
    split continuouse data into train, validate, test; No target variable

    argument: df

    return: train, validate, test
    '''

    train_val, test = train_test_split(df,
                                   train_size=0.8,
                                   random_state=1108,
                                   )
    train, validate = train_test_split(train_val,
                                   train_size=0.7,
                                   random_state=1108,
                                   )
    
    print(f'Train: {len(train)/len(df)}')
    print(f'Validate: {len(validate)/len(df)}')
    print(f'Test: {len(test)/len(df)}')
    

    return train, validate, test