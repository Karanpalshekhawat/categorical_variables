"""
This script modules are used to update
dataframes like filling the NaN values,
update the data type.
"""



def fill_na_with_none(df):
    """
    Fill all categorical NaN values with string type None
    so that we can use sklearn label encoding package
    Args:
        df (pd.DataFrame): training dataset

    Returns:
    """
    features = [i for i in df.columns if i not in ['id', 'target', 'kfold']]
    for feat in features:
        df.loc[:, feat] = df[feat].astype(str).fillna("NONE")

    return df