import os
import polars as pl
import numpy as np

def load_proceprocessing_raw_data(source_path: str, random_seed: int, train_test_split_ratio: float=0.0) -> pl.DataFrame:
    pl.set_random_seed(random_seed)
    
    raw_dataframe = pl.read_csv(source_path)
    raw_dataframe = _add_split_column(raw_dataframe, train_test_split_ratio)

    return raw_dataframe


def save_preprocessing_dataframe(dataframe: pl.DataFrame, output_path: str) -> None:
    """
    Save a single dataframe to the specified path.

    Args:
        dataframe: Polars DataFrame to save
        output_path: Path for saving the file
        compression_level: Zstd compression level (0-22)
    """
    # Create train and test directories
    train_path = os.path.join(output_path, "train.json")
    test_path = os.path.join(output_path, "validation.json")

    # Ensure directories exist
    os.makedirs(output_path, exist_ok=True)

    # Save train data
    train_dataframe = dataframe.filter(pl.col("split") == "train")
    _save_dataframe_partition(train_dataframe, train_path, index=0)
    
    # Save test data
    test_dataframe = dataframe.filter(pl.col("split") == "test")
    _save_dataframe_partition(test_dataframe, test_path, index=0)


def _add_split_column(dataframe: pl.DataFrame, train_test_split_ratio: float) -> pl.DataFrame:
    """
    Add a split column to the dataframe based on train_test_split ratio.
    
    Args:
        dataframe: Input Polars DataFrame
        train_test_split: Ratio for splitting data (0.0 to 1.0)
        
    Returns:
        DataFrame with added split column
    """
    return dataframe\
        .with_columns(
            pl.lit(np.random.rand(dataframe.height)).alias("random_sample")
        )\
        .with_columns(
            pl.when(pl.col.random_sample < train_test_split_ratio)\
                .then(pl.lit("test"))\
                .otherwise(pl.lit("train"))\
                .alias("split")
        )


def _save_dataframe_partition(dataframe: pl.DataFrame, file_path: str, index: int) -> None:
    """
    Save a single dataframe partition to parquet file.
    
    Args:
        dataframe: Polars DataFrame to save
        partition_path: Directory path for saving
        index: File index number
        compression_level: Zstd compression level (0-22)
    """
    if dataframe.height > 0:
        dataframe\
            .drop("split")\
            .write_ndjson(file_path)
        print(f"Dataset {index} saved to {file_path}")
