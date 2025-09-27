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
tokenized_training_string = load_training_data(train_data_source_path, feature_column_name, tokenizer)

#%%
examples = []
block_size = 512
BATCH_SIZE = 32
BUFFER_SIZE = 1000

for i in range(0, len(tokenized_training_string) - block_size + 1, block_size):
  examples.append(tokenized_training_string[i:i + block_size])

inputs, labels = [], []

for ex in examples:
  inputs.append(ex[:-1])
  labels.append(ex[1:])

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# %%
model = ClaudinoGPT(tokenizer)
# defining our optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compiling the model
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

num_epoch = 10
history = model.fit(dataset, epochs=num_epoch)