import glob
import json
import os
import re
from distutils.version import LooseVersion
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import pytesseract
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AdamW,
    LayoutLMConfig,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
    LayoutLMModel,
    LayoutLMTokenizer,
)

from datasets import Array2D, ClassLabel, Dataset, Features, Sequence, Value, concatenate_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayoutLM:
    """
    Args:
        Vocab Size
        Hidden State
        Number of Hidden Layers
        Number of Attention Heads
        Intermediate Size
        Hidden Act
        Hidden Dropout Probability
        Attention Probability Dropout
        Maximum Position Embeddings
        Vocab Size Type
        Initializer Range
        Layer Norm Epsilon
        Pad Token ID
        Max 2d Position Embeddings
    Methods:
        process_json: Process json files with raw document information to output key information
            (words, bounding boxes, labels)
        process_images: call to process raw document images as a whole, leverages ocr
        get_encodings: get encodings for raw (words, bounding boxes, labels) data
        get_hidden_state: get hidden states by passing in encodings and model
        fine_tune: for model finetuning

    """

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

        self.configuration = LayoutLMConfig(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            initializer_range,
            layer_norm_eps,
            pad_token_id,
            max_2d_position_embeddings,
        )

        # Initializing a model from the configuration
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        # self.model.to(device)

        # For processing data embeddings
        self.pt_data = pd.DataFrame(columns=["image_path", "words", "bbox", "label"])
        self.pt_data["words"] = self.pt_data["words"].astype("object")
        self.pt_data["bbox"] = self.pt_data["bbox"].astype("object")
        self.pt_data["label"] = self.pt_data["label"].astype("object")
        self.encoding = pd.DataFrame()

        # For fine-tuning
        self.label2idx = {}
        self.ft_data = pd.DataFrame()

    def process_json(
        self, directory, word_label, position_label, label_label=None, position_processing=False, token=False
    ):
        """
        Args:
            directory: directory path of json files which you which to process
            word_label: label that words are saved under in json file
            position_label: label that bounding boxes or position coordinates are saved under in json file
            label_label: label that token labels are saved under in json file
            position_processing: positions should be saved in [top, left, bottom, right] format.
                if set to false (saved in [top, left, height, width]), it will calculate positions
            funsd: boolean for if you are using funsd dataset which requires special processing
        Outputs:
            Dataframe with image path, words, bbox, label
        """
        # Import Files

        files = Path(directory).rglob("*.json")

        i = 0
        max_files = 10

        for f in files:
            if i > max_files:
                print(f"Debugging, only loading {max_files} files!")
                break

            words = []
            positions = []
            labels = []

            with open(f, "r") as json_file:
                json_data = json.load(json_file)

                if token == True:
                    json_data = json_data["form"]
                    for data in json_data:
                        for w in data['words']:
                            words.append(w[word_label])
                            positions.append(w[position_label])
                            labels.append(data[label_label])

                for data in json_data:

                    words.append(data[word_label])

                    if position_processing == True:
                        # top-left corner is (x0, y0)
                        # bottom-right corner is (x1, y1)
                        # breakpoint()
                        positions.append(
                            [
                                # int(data[position_label]["left"] * 1000),  # x0
                                # int((data[position_label]["top"] - data[position_label]["height"]) * 1000),  # y0
                                # int((data[position_label]["left"] + data[position_label]["width"]) * 1000),  # x1
                                # int(data[position_label]["top"] * 1000),  # y1
                                int(data[position_label]["left"]) * 1000,  # x0
                                int(data[position_label]["top"]) * 1000,  # y0
                                int((data[position_label]["left"] + data[position_label]["width"])) * 1000,  # x1
                                int((data[position_label]["top"] + data[position_label]["height"])) * 1000,  # y1
                            ]
                        )
                        # TODO: Normalize coordinates.
                    else:
                        positions.append(data[position_label])

                    if label_label is not None:
                        labels.append(data[label_label])

            self.pt_data.at[i, "image_path"] = str(f)
            self.pt_data.at[i, "words"] = words.copy()
            self.pt_data.at[i, "bbox"] = positions.copy()
            self.pt_data.at[i, "label"] = labels.copy()
            i += 1

        return self.pt_data

    def __normalize_box(self, box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    def __ocr(self, datapoint):
        try:
            image = Image.open(datapoint["image_path"])
        except:
            datapoint["words"] = None
            datapoint["bbox"] = None
            return datapoint
        else:
            width, height = image.size

            # apply ocr to the image
            df = pytesseract.image_to_data(image, output_type="data.frame")
            float_cols = df.select_dtypes("float").columns
            df = df.dropna().reset_index(drop=True)
            df[float_cols] = df[float_cols].round(0).astype(int)
            df = df.replace(r"^\s*$", np.nan, regex=True)
            df = df.dropna().reset_index(drop=True)

            # get the words and actual (unnormalized) bounding boxes
            # words = [word for word in ocr_df.text if str(word) != 'nan'])
            words = list(df.text)
            words = [str(w) for w in words]
            coordinates = df[["left", "top", "width", "height"]]
            unnorm_boxes = []
            for i, row in coordinates.iterrows():
                x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
                unnorm_box = [
                    x,
                    y,
                    x + w,
                    y + h,
                ]  # we turn it into (left, top, left+width, top+height) to get the actual box
                unnorm_boxes.append(unnorm_box)

            # normalize the bounding boxes
            boxes = []

            for box in unnorm_boxes:
                boxes.append(self.__normalize_box(box, width, height))

            # add as extra columns
            assert len(words) == len(boxes)
            datapoint["words"] = words
            datapoint["bbox"] = boxes
            return datapoint

    def process_images(self, in_directory, out_directory):
        """
        Args:
            in_directory: directory path of image files which you which to process
            out_directory: directory path where you want to store pickles
        Outputs:
            Dataframe with image path, words, bbox, label
        """
        folders = [re.match(r"^.+/(.*)", f).group(1) for f in glob.glob(in_directory + "/*")]

        i = 0
        # for label in folders:
        for label in folders:

            self.ft_data = pd.DataFrame()

            files = glob.glob(in_directory + "/" + label + "/*")

            for filepath in files:
                self.ft_data.at[i, "image_path"] = filepath
                self.ft_data.at[i, "label"] = label
                i = i + 1

            self.ft_data = Dataset.from_pandas(self.ft_data)
            self.ft_data = self.ft_data.remove_columns("__index_level_0__")
            self.ft_data = self.ft_data.map(self.__ocr)
            self.ft_data = self.ft_data.filter(lambda x: x["words"] is not None)

            # Output to json file
            self.ft_data.to_pandas().to_pickle(out_directory + "/" + label + ".pkl")
        return

    def __encode_example(self, example, labels=None, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):

        words = example["words"]
        normalized_word_boxes = example["bbox"]

        assert len(words) == len(normalized_word_boxes)

        token_boxes = []
        for word, box in zip(words, normalized_word_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # Truncation of token_boxes
        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = self.tokenizer(" ".join(words), padding="max_length", truncation=True)
        input_ids = self.tokenizer(" ".join(words), truncation=True)["input_ids"]

        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        encoding["bbox"] = np.array(token_boxes)

        # if np.any(encoding["bbox"] > 1000):
        #     breakpoint()

        if labels is not None:
            if isinstance(labels, str):
                encoding["label_idx"] = self.label2idx[example["label"]]
            else:
                special_tokens_count = 2
                label_int = example["label"]
                if len(label_int) > max_seq_length - special_tokens_count:
                    label_int = label_int[: (max_seq_length - special_tokens_count)]

                encoding["label_idx"] = np.array(
                    [-100] + [self.label2idx[x] for x in label_int] + [-100] * (max_seq_length - len(label_int) - 1)
                )

        return encoding

    def get_encodings(self, outpath=None, labels=None, directory=None, max_seq_length=512):
        """
        Args:
            outpath: default None, directory path where you want to store pickles
            labels: default None, pass in dictionary of label to index key-value pairs
            directory: default None, pass in directory of encoded pickles if you instantiated instance without
                processing images or json files beforehand
            max_seq_length: default 512
        Outputs:
            Dataframe with image path, words, bbox, input_ids (token index), attention mask, token type ids,
            label index (numeric encoding)
        """
        if directory is not None:
            # Load data from pkls
            dfs = glob.glob(directory + "/*")

            i = 0
            for df in dfs:
                label = re.match(r".+/(.*).pkl", df).group(1)
                self.label2idx[label] = i
                i += 1
                self.ft_data = Dataset.from_pandas(pd.read_pickle(df))
                if "__index_level_0__" in self.ft_data.features.keys():
                    self.ft_data = self.ft_data.remove_columns("__index_level_0__")

                features = Features(
                    {
                        "input_ids": Sequence(feature=Value(dtype="int64")),
                        "bbox": Array2D(dtype="int64", shape=(max_seq_length, 4)),
                        "attention_mask": Sequence(Value(dtype="int64")),
                        "token_type_ids": Sequence(Value(dtype="int64")),
                        "label_idx": ClassLabel(
                            num_classes=len(self.label2idx.keys()), names=list(self.label2idx.values())
                        ),
                        "image_path": Value(dtype="string"),
                        "words": Sequence(feature=Value(dtype="string")),
                    }
                )

                self.encodings = self.ft_data.map(
                    lambda example: self.__encode_example(example, labels), features=features
                )
                self.encodings.set_format(
                    type="torch", columns=["input_ids", "bbox", "attention_mask", "token_type_ids"]
                )
                self.encodings.to_pandas().to_pickle(outpath + "/" + label + ".pkl")

            a_file = open(outpath + "/labels_dict.json", "w")
            json.dump(self.label2idx, a_file)
            a_file.close()

        else:
            self.pt_data = Dataset.from_pandas(self.pt_data)
            self.pt_data = self.pt_data.remove_columns("__index_level_0__")

            if labels is not None:

                self.label2idx = labels

                a_file = open(outpath + "/labels_dict.json", "w")
                json.dump(self.label2idx, a_file)
                a_file.close()

                features = Features(
                    {
                        "input_ids": Sequence(feature=Value(dtype="int64")),
                        "bbox": Array2D(dtype="int64", shape=(512, 4)),
                        "attention_mask": Sequence(Value(dtype="int64")),
                        "token_type_ids": Sequence(Value(dtype="int64")),
                        "image_path": Value(dtype="string"),
                        "words": Sequence(feature=Value(dtype="string")),
                        "label": Sequence(feature=Value(dtype="string")),
                        "label_idx": Sequence(feature=Value(dtype="int64")),
                    }
                )

            else:
                self.pt_data = self.pt_data.remove_columns("label")
                features = Features(
                    {
                        "input_ids": Sequence(feature=Value(dtype="int64")),
                        "bbox": Array2D(dtype="int64", shape=(512, 4)),
                        "attention_mask": Sequence(Value(dtype="int64")),
                        "token_type_ids": Sequence(Value(dtype="int64")),
                        "image_path": Value(dtype="string"),
                        "words": Sequence(feature=Value(dtype="string")),
                    }
                )

            self.encodings = self.pt_data.map(lambda example: self.__encode_example(example, labels), features=features)
            self.encodings.set_format(type="torch", columns=["input_ids", "bbox", "attention_mask", "token_type_ids"])

            if outpath is not None:
                self.encodings.to_pandas().to_pickle(outpath + "/enc_data.pkl")

        return self.encodings

    def __get_example_hidden_state(self, datapoint, model, model_path):
        input_ids = datapoint["input_ids"].view(1, -1).to(device)
        bbox = datapoint["bbox"].view(1, -1, 4).to(device)
        attention_mask = datapoint["attention_mask"].view(1, -1).to(device)
        token_type_ids = datapoint["token_type_ids"].view(1, -1).to(device)

        if torch.any(datapoint["bbox"] > 1000).item() or torch.any(datapoint["bbox"] < 0).item():
            print(bbox.tolist())
            raise ValueError("bbox values must be between [0, 1000]")

        outputs = model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids.view(1, -1)
        )

        datapoint["last_hidden_state"] = outputs.last_hidden_state[0]

        return datapoint

    def get_hidden_state(self, outpath=None, model_path=None):
        """
        Args:
            outpath: default None, directory path where you want to store pickles
            model_path: default None, pass in model path (a directory storing a config and bin file),
                else a vanilla LayoutLM pretrained model will be used
        Outputs:
            Dataframe with image path, words, bbox, input_ids (token index), attention mask, token type ids,
            label index (numeric encoding), and last hidden state embedding
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is not None:
            model = LayoutLMModel.from_pretrained(model_path, local_files_only=True)
        else:
            model = self.model

        model.to(device)

        dataset = self.encodings.map(lambda example: self.__get_example_hidden_state(example, model, model_path))

        if outpath is not None:
            dataset.to_pandas().to_pickle(outpath)

        return dataset

    def fine_tune(self, input_dir, model_save_path, token, batch_size=5, num_train_epochs=5, save_epoch=1):

        """
        Args:
            input_dir: Pickle with bbox, input_ids (token index), attention mask, token type ids,
                label index (numeric encoding)
            model_save_path: path to save model to
            token: boolean, sequence or token classification
            batch_size: default 5
            num_train_epochs: default 5
            save_epoch: save model every x number of epochs
        Outputs:
            None, save models to model path
        """
        # Bring in data
        dfs = glob.glob(input_dir + "/*.pkl")

        a_file = open(input_dir + "/labels_dict.json", "r")
        self.label2idx = json.loads(a_file.read())
        a_file.close()

        train_datasets = []
        test_datasets = []
        for df in dfs:

            int_df = pd.read_pickle(df)
            # Flatten each bbox first
            int_df["bbox"] = int_df["bbox"].apply(lambda x: x.tolist())
            int_df["bbox"] = int_df["bbox"].apply(lambda x: [item for sublist in x for item in sublist])
            int_df["bbox"] = int_df["bbox"].apply(lambda x: np.array(x))

            # Randomly sample validation data
            if len(int_df) < 200:
                train, test = train_test_split(int_df, test_size=0.2)
            else:
                train, test = train_test_split(int_df, test_size=0.1)

            train_datasets.append(Dataset.from_pandas(train))
            test_datasets.append(Dataset.from_pandas(test))

        train_data = concatenate_datasets(train_datasets)
        test_data = concatenate_datasets(test_datasets)

        train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label_idx", "bbox"]
        )
        test_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label_idx", "bbox"]
        )

        train_size = len(train_data)
        test_size = len(test_data)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Put the model in training mode

        if token:
            self.model = LayoutLMForTokenClassification.from_pretrained(
                "microsoft/layoutlm-base-uncased", num_labels=len(self.label2idx)
            )
        else:
            self.model = LayoutLMForSequenceClassification.from_pretrained(
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
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label_idx"].view(batch["label_idx"].size()[0], -1).to(device)

                # forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                loss = outputs.loss

                running_loss += loss.item()
                predictions = outputs.logits.argmax(-1)

                if token:
                    for i in range(0, predictions.size()[0]):
                        for j in range(0, predictions.size()[1]):
                            if labels[i][j] != -100:
                                total += 1
                                if predictions[i][j] == labels[i][j]:
                                    correct += 1

                else:
                    correct += (predictions == labels.view(labels.size()[1], labels.size()[0])).float().sum()

                # backward pass to get the gradients
                loss.backward()

                # update
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

            print("Loss:", running_loss / steps)

            if token:
                accuracy = 100 * correct / total
                print("Training accuracy:", accuracy)
            else:
                accuracy = 100 * correct / train_size
                print("Training accuracy:", accuracy.item())

            if (epoch % save_epoch == 0) & (epoch > 0):
                new_dir = model_save_path + "/epoch" + str(epoch)
                os.mkdir(new_dir)
                self.model.save_pretrained(new_dir)

            # Calculate dev set loss for training epoch

            test_running_loss = 0
            test_correct = 0
            test_total = 0
            test_steps = 0

            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device)
                    bbox = batch["bbox"].view(batch["bbox"].size()[0], 512, 4).to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch["token_type_ids"].to(device)
                    labels = batch["label_idx"].view(batch["label_idx"].size()[0], -1).to(device)

                    # forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        bbox=bbox,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )

                    loss = outputs.loss
                    test_running_loss += loss.item()
                    predictions = outputs.logits.argmax(-1)

                    if token:
                        for i in range(0, predictions.size()[0]):
                            for j in range(0, predictions.size()[1]):
                                if labels[i][j] != -100:
                                    test_total += 1
                                    if predictions[i][j] == labels[i][j]:
                                        test_correct += 1

                    else:
                        test_correct += (predictions == labels.view(labels.size()[1], labels.size()[0])).float().sum()
                test_steps += 1

            print("Validation Loss:", test_running_loss / test_steps)

            if token:
                accuracy = 100 * test_correct / test_total
                print("Validation accuracy:", accuracy)
            else:
                accuracy = 100 * test_correct / test_size
                print("Validation accuracy:", accuracy.item())

        return


if __name__ == "__main__":

    directory = "/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/rivlets"
