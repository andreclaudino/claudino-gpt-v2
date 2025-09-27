from typing import List, Tuple
import polars as pl
import tensorflow as tf

from claudino_gpt.models.tokenizer_constants import SPECIAL_TOKENS
from claudino_gpt.models.tokenizer import ClaudinoGPTTokenizer

BATCH_SIZE = 16
BUFFER_SIZE = 1000


def load_training_data(
    source_path: str, 
    feature_column_name: str, 
    block_size: int, 
    tokenizer: ClaudinoGPTTokenizer
) -> tf.data.Dataset:
    # Obter o ID do token <UNK> a partir do tokenizer usando a constante
    unk_token = SPECIAL_TOKENS["unk_token"]
    
    # Suponha que seu tokenizer tenha um método `token_to_id` ou um vocabulário acessível
    unk_token_id = tokenizer.encode(unk_token)[0]

    if unk_token_id is None:
        raise ValueError(f"Token UNK '{unk_token}' não encontrado no vocabulário do tokenizer.")

    # Carregar e limpar os dados
    data_frame = pl.read_ndjson(source_path)

    features_list = (
        data_frame
        .select(pl.col(feature_column_name).fill_null("").cast(pl.Utf8))
        .to_series()
        .to_list()
    )
    # Opcional: remover strings vazias
    features_list = [s for s in features_list if s.strip() != ""]

    features_string = SPECIAL_TOKENS["eos_token"].join(features_list)
    tokenized_string = tokenizer.encode(features_string)

    # 🔁 Substituir tokens inválidos (não-inteiros) por unk_token_id
    clean_tokens = []
    for i, token in enumerate(tokenized_string):
        if isinstance(token, int):
            clean_tokens.append(token)
        else:
            print(
                f"⚠️ Token inválido na posição {i}: {repr(token)} (tipo: {type(token)}). "
                f"Substituindo por UNK ('{unk_token}', ID={unk_token_id})."
            )
            clean_tokens.append(unk_token_id)

    if len(clean_tokens) < block_size:
        raise ValueError(
            f"Dados insuficientes após limpeza para block_size={block_size} "
            f"(tokens válidos: {len(clean_tokens)})"
        )

    # Criar blocos
    examples = []
    for i in range(0, len(clean_tokens) - block_size + 1, block_size):
        examples.append(clean_tokens[i:i + block_size])

    if not examples:
        raise ValueError("Nenhum exemplo válido foi gerado.")

    inputs = [ex[:-1] for ex in examples]
    labels = [ex[1:] for ex in examples]

    # Criar dataset do TensorFlow
    dataset_treino = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset_treino = dataset_treino.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    return dataset_treino