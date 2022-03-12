from PIL import Image
import torch
from typing import List, Tuple, Dict, Set, Union
import json
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import LayoutLMv2Model, LayoutLMv2Processor

from datasets import Array2D, Array3D, Array4D, ClassLabel, Dataset, Features, Sequence, Value, concatenate_datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pytesseract
import os

from transformers import (
    LayoutLMv2Processor, 
    LayoutLMv2Model, 
    LayoutLMv2ForSequenceClassification,
    AdamW)


class LayoutLMv2:
    """Get embeddings for LayoutLMv2"""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        max_2d_position_embeddings=1024,
    ):

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

        self.encoding = pd.DataFrame()
        self.label2idx = {}
        self.int_data = pd.DataFrame()

    def reset_encodings(self):
        self.encoding = pd.DataFrame()

    def __get_encodings(self, example):
        image = Image.open(example["image_path"]).convert("RGB")
        encoding = self.processor(image, padding="max_length", truncation=True)
        example["token_type_ids"] = encoding.token_type_ids[0]
        example["attention_mask"] = encoding.attention_mask[0]
        example["bbox"] = encoding.bbox[0]
        example["image"] = np.array(encoding.image[0])
        example["input_ids"] = encoding.input_ids[0]
        return example


    def __get_hidden_states(self, example, model = None):
        image = Image.open(example["image_path"]).convert("RGB")
        outputs = self.model(**self.processor(image, padding="max_length", truncation=True, return_tensors="pt"))
        example["last_hidden_state"] = outputs.last_hidden_state[0]
        return example


    def get_outputs(self, directory, labels = False, lhs = True, model = None, outpath = None, dict_path = None, file_type = "png"):

        # Import Files
        path = Path(path)

        if path.is_dir():
            files = path.rglob("*." + file_type)
        else:
            files = [path]

        if labels:
            image_paths, label = zip(*[(str(f), re.match(r"(%s)/(.*?)/(.*?)" % str(directory), str(f)).group(2)) for f in files])
            self.encoding.at[:, "image_path"] = image_paths
            self.encoding.at[:, "label"] = label

            self.encoding = self.encoding.query("label == 'advertisement' or label == 'budget'")

            #Encode labels
            self.label2idx = { k : v for v, k in enumerate(self.encoding.label.unique())}
            self.encoding.at[:, "label_idx"] = self.encoding.apply(lambda x: self.label2idx[x.label], axis = 1)

            #Output labels dictionary
            a_file = open(dict_path, "w")
            json.dump(self.label2idx, a_file)
            a_file.close()

            #Encode via mapping
            self.encoding = Dataset.from_pandas(self.encoding)

            try:
                self.encoding = self.encoding.remove_columns("__index_level_0__")
            except:
                pass

                
            features = Features(
                {
                #"image": Sequence(Sequence(Sequence(Sequence(Value(dtype="uint8"))))),
                "image": Array3D(dtype="int64", shape=(3, 224, 224)),
                'input_ids': Sequence(Value(dtype='int64')),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "attention_mask": Sequence(Value(dtype="int64")),
                "token_type_ids": Sequence(Value(dtype="int64")),
                "image_path": Value(dtype="string"),
                "label": Value(dtype="string"),
                "label_idx": Value(dtype="int64"),
                }
                )    

            self.int_data = self.encoding.map(lambda example: self.__get_encodings(example), features=features, batch_size = 10)
        
        else:
            #Select just one image to use -- need to think about whether we want to use more than 1 page
            #Should we concatenate the two hidden states?
            image_paths = [re.match(r"(.+)/pages/0/(.*)", str(f)).group(0) for f in files \
                if re.match(r"(.+)/pages/0/(.*)", str(f)) is not None]

            #Testing a smaller number of images at the moment
            i = 0
            for ip in image_paths:
                self.encoding.at[i, "image_path"] = ip
                i += 1

            #Encode via mapping
            self.encoding = Dataset.from_pandas(self.encoding).remove_columns("__index_level_0__")
                
            features = Features(
                {
                #"image": Sequence(Sequence(Sequence(Sequence(Value(dtype="uint8"))))),
                "image": Array3D(dtype="int64", shape=(3, 224, 224)),
                'input_ids': Sequence(Value(dtype='int64')),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "attention_mask": Sequence(Value(dtype="int64")),
                "token_type_ids": Sequence(Value(dtype="int64")),
                "image_path": Value(dtype="string"),
                }
                )    

            self.int_data = self.encoding.map(lambda example: self.__get_encodings(example), features=features)

        if lhs:
            if model is None:
                model = self.model
            else:
                model = LayoutLMv2Model.from_pretrained(model, local_files_only=True)

            self.int_data = self.int_data.map(lambda example: self.__get_hidden_states(example, model))

        self.int_data.set_format(type="torch", columns=["image", "bbox", "attention_mask", "token_type_ids", "input_ids"])

        if outpath is not None:
            self.int_data.to_pandas().to_pickle(outpath)

        return self.int_data

    def fine_tune(self, input_dir, labels_dir, model_save_path, batch_size = 5, num_train_epochs = 1, save_epoch = 1):

        """
        Args:
            input_dir: Pickle with bbox, input_ids (token index), attention mask, token type ids,
                label index (numeric encoding)
            model_save_path: path to save model to
            batch_size: default 5
            num_train_epochs: default 5
            save_epoch: save model every x number of epochs, default 1
        Outputs:
            None, save models to model path
        """    
        # Bring in data
        a_file = open(labels_dir, "r")
        self.label2idx = json.loads(a_file.read())
        a_file.close()

        df = pd.read_pickle(input_dir)
        df["bbox"] = df["bbox"].apply(lambda x: np.array(x).flatten())
        df["image"] = df["image"].apply(lambda x: np.array(x).flatten())

        # Randomly sample validation data
        if len(df) <= 200:
            train, test = train_test_split(df, test_size=0.2)
        else:
            train, test = train_test_split(df, test_size=0.1)

        train_data = Dataset.from_pandas(train)
        test_data = Dataset.from_pandas(test)

        train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label_idx", "bbox", "image"]
        )
        test_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label_idx", "bbox", "image"]
        )

        train_size = len(train_data)
        test_size = len(test_data)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Put the model in training mode

        self.model = LayoutLMv2ForSequenceClassification.from_pretrained(
                "microsoft/layoutlm-base-uncased", num_labels=len(self.label2idx)
            )

        self.model.to(device)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(num_train_epochs):
            print("Training Epoch:", epoch)
            running_loss = 0.0
            total = 0
            correct = 0
            steps = 0
            self.model.train()
            for batch in tqdm(train_dataloader):
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].view(batch["bbox"].size()[0], 512, 4).to(device)
                image = batch["image"].view(batch["image"].size()[0], 3, 224, 224).to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label_idx"].view(batch["label_idx"].size()[0], -1).to(device)

                # forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    bbox=bbox,
                    image = image,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                loss = outputs.loss

                running_loss += loss.item()
                predictions = outputs.logits.argmax(-1)
                correct += (predictions == labels.view(labels.size()[1], labels.size()[0])).float().sum()

                # backward pass to get the gradients
                loss.backward()

                # update
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

            print("Loss:", running_loss / steps)

            accuracy = 100 * correct / train_size
            print("Training accuracy:", accuracy.item())

            if (epoch % save_epoch == 0) & (epoch > 0):
                new_dir = str(model_save_path) + "/epoch" + str(epoch)
                os.mkdir(new_dir)
                self.model.save_pretrained(new_dir)

            # Calculate dev set loss for training epoch

            test_running_loss = 0
            test_correct = 0
            test_steps = 0

            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device)
                    bbox = batch["bbox"].view(batch["bbox"].size()[0], 512, 4).to(device)
                    image = batch["image"].view(batch["image"].size()[0], 3, 224, 224).to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch["token_type_ids"].to(device)
                    labels = batch["label_idx"].view(batch["label_idx"].size()[0], -1).to(device)

                    # forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        bbox=bbox,
                        image = image,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )

                    loss = outputs.loss
                    test_running_loss += loss.item()
                    predictions = outputs.logits.argmax(-1)

                    test_correct += (predictions == labels.view(labels.size()[1], labels.size()[0])).float().sum()
                test_steps += 1

            print("Validation Loss:", test_running_loss / test_steps)
            accuracy = 100 * test_correct / test_size
            print("Validation accuracy:", accuracy.item())

        return
