import polars as pl
import tensorflow as tf

from claudino_gpt.models.tokenizer_constants import SPECIAL_TOKENS
from claudino_gpt.models.tokenizer import ClaudinoGPTTokenizer

BATCH_SIZE = 16
BUFFER_SIZE = 1000


def load_training_data(source_path: str, feature_column_name: str, block_size: int, tokenizer: ClaudinoGPTTokenizer) -> tf.data.Dataset:
    examples = []

    data_frame = pl.read_csv(source_path)
    
    features_list = data_frame[feature_column_name].to_list()
    features_string = SPECIAL_TOKENS["eos_token"].join(features_list)

    tokenized_string = tokenizer.encode(features_string)
    
    for i in range(0, len(tokenized_string) - block_size + 1, block_size):
        examples.append(tokenized_string[i:i + block_size])

    inputs, labels = [], []

    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    dataset_treino = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset_treino = dataset_treino.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    return dataset_treino

