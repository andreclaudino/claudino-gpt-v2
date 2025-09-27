#%%
vocabulary_size = 50000
tokenizer_data_source_path = "outputs/tokenizer_preprocessing/train.csv"
#%%
from claudino_gpt.models.tokenizer import ClaudinoGPTTokenizer

#%%
tokenizer = ClaudinoGPTTokenizer(vocabulary_size=50000)
tokenizer.train(tokenizer_data_source_path)
tokenizer.save("outputs/tokenizer")