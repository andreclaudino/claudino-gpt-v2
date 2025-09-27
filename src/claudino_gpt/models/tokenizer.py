import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from claudino_gpt.models.tokenizer_constants import SPECIAL_TOKENS



class ClaudinoGPTTokenizer(object):
    def __init__(self, vocabulary_size: int):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ]) # type: ignore
        self.tokenizer.pre_tokenizer = ByteLevel() # type: ignore
        self.tokenizer.decoder = ByteLevelDecoder() # type: ignore
        self._vocabulary_size = vocabulary_size

    def train(self, path: str):
        trainer = BpeTrainer(
            vocab_size=self._vocabulary_size, # type: ignore
            show_progress=True, # type: ignore
            inital_alphabet=ByteLevel.alphabet(), # type: ignore
            special_tokens=list(SPECIAL_TOKENS.values()) # type: ignore
        ) # type: ignore
        self.tokenizer.train([path], trainer=trainer)

    def save(self, location, prefix=None):
        os.makedirs(location, exist_ok=True)
        self.tokenizer.model.save(location, prefix)

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id(SPECIAL_TOKENS["bos_token"])
    
    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id(SPECIAL_TOKENS["eos_token"])
    