import polars as pl 


def extract_features_for_tokenizer(dataframe: pl.DataFrame, feature_column_name: str) -> pl.DataFrame:
    """
    Extracts the specified feature column from the given DataFrame for tokenization.

    Args:
        dataframe (pl.DataFrame): The input DataFrame containing raw data.
        feature_column_name (str): The name of the column to extract.

    Returns:
        pl.DataFrame: A DataFrame containing only the specified feature column.
    """
    dataframe = dataframe\
        .select(pl.col(feature_column_name), pl.col("split"))\
        .drop_nans()\
        .drop_nulls()\
        .unique()

    return dataframe