import polars as pl


def preprocess_raw_data(
        source_dataframe: pl.DataFrame,
        feature_column_name: str,
        output_column_name: str,
        random_seed: int
    ) -> pl.DataFrame:
    pl.set_random_seed(random_seed)
    
    preprocessed_dataframe = source_dataframe.with_columns(
        pl.col(feature_column_name).str.strip_chars().alias(output_column_name
    )).select(output_column_name)

    return preprocessed_dataframe