from transformers import LayoutLMv2Processor, LayoutLMv2Model
from PIL import Image
import torch
from typing import List, Tuple, Dict, Set, Union
import json
import pytesseract
import glob
import re

class LayoutLMv2():
    """ Get embeddings for LayoutLMv2
    """
    def __init__(self, vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12,\
        num_attention_heads = 12, intermediate_size = 3072, hidden_act = 'gelu', hidden_dropout_prob = 0.1,\
            attention_probs_dropout_prob = 0.1, max_position_embeddings = 512, type_vocab_size = 2,\
                initializer_range = 0.02, layer_norm_eps = 1e-12, pad_token_id = 0, \
                    max_2d_position_embeddings = 1024):

        # Initializing a LayoutLM configuration with default values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.max_2d_position_embeddings = max_2d_position_embeddings

        # Initializing a model from the configuration
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")

        self.doc_encoding = {}

    def get_outputs(self, directory) -> dict:

        #Import Image Files
        files = glob.glob(directory + '/**/*.png', recursive=True)

        #Select just one image to use -- need to do more research into whether 
        #we can use more than one page
        image_paths = [re.match(r"(.+)/pages/0/(.*)", f).group(0) for f in files \
            if re.match(r"(.+)/pages/0/(.*)", f) is not None]

        for ip in image_paths:

            #docname = re.match(r"^.+/(.*)", ip).group(1)

            image = Image.open(ip).convert("RGB")                
            
            encoding = self.processor(image, return_tensors="pt")
            
            outputs = self.model(**encoding)

            self.doc_encoding[ip] = outputs.last_hidden_state

        return self.doc_encoding

if __name__ == '__main__':

    directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/files'

    i2 = LayoutLMv2()
    encodings = i2.get_outputs(directory)

    print(encodings)
