import polars as pl

from claudino_gpt.models.tokenizer_constants import SPECIAL_TOKENS
from claudino_gpt.models.tokenizer import ClaudinoGPTTokenizer


def load_training_data(source_path: str, feature_column_name: str, tokenizer: ClaudinoGPTTokenizer) -> str:
    data_frame = pl.read_csv(source_path)
    
    features_list = data_frame[feature_column_name].to_list()
    features_string = SPECIAL_TOKENS["eos_token"].join(features_list)

    string_tokenized = tokenizer.encode(features_string)

    return string_tokenized

