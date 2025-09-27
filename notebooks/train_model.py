#%%
train_data_source_path = "outputs/tokenizer_preprocessing/train.csv"
feature_column_name = "conteudo_noticia"
tokenizer_path = "outputs/tokenizer"
#%%
import tensorflow as tf
from transformers import GPT2Tokenizer
from claudino_gpt.persistence.model_data import load_training_data
from claudino_gpt.models.claudino_gpt import ClaudinoGPT

#%%
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
tokenized_training_string = load_training_data(train_data_source_path, feature_column_name, 128, tokenizer)

#%%
