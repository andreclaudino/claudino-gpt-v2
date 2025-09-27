#%%
raw_seed = 123456789
train_test_split_ratio = 0.2

source_preprocessing_path = "resources/materias_raw.csv"
raw_feature_column_name = "conteudo_noticia"
tokenizer_preprocessing_output_path = "outputs/tokenizer_preprocessing"
#%%
from claudino_gpt.feature_engineering.tokenization import extract_features_for_tokenizer
from claudino_gpt.persistence.preprocessing import load_proceprocessing_raw_data, save_preprocessing_dataframe


#%%
raw_preprocessing_dataframe = load_proceprocessing_raw_data(source_preprocessing_path, train_test_split_ratio, raw_seed)
tokenizer_features_dataframe = extract_features_for_tokenizer(raw_preprocessing_dataframe, raw_feature_column_name)
save_preprocessing_dataframe(tokenizer_features_dataframe, tokenizer_preprocessing_output_path)

#%%
