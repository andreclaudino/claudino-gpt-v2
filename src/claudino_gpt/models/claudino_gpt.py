import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

from claudino_gpt.models.tokenizer import ClaudinoGPTTokenizer


class ClaudinoGPT(TFGPT2LMHeadModel):
    def __init__(
            self,
            tokenizer: ClaudinoGPTTokenizer,
            name: str = "claudino_gpt",
            *args, **kwargs
        ):
        self._config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            *args,
            **kwargs
        )
        super().__init__(self._config, name=name)

    
    def call(self, inputs, training=False):
        return super().call(inputs, training=training)