from transformers import LayoutLMModel, LayoutLMConfig, AdamW
from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
from PIL import Image
import torch
from typing import List, Tuple, Dict, Set, Union
import json
import pytesseract
import glob
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, concatenate_datasets
from transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayoutLM():
    """ Get embeddings for Layout LM V1
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

        self.configuration = LayoutLMConfig(vocab_size, hidden_size, num_hidden_layers,\
            num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob,\
                attention_probs_dropout_prob, max_position_embeddings, type_vocab_size,\
                    initializer_range, layer_norm_eps, pad_token_id, \
                        max_2d_position_embeddings)

        # Initializing a model from the configuration
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        #self.model.to(device)

        #For processing data embeddings
        self.pt_data = pd.DataFrame(columns = ['image_path', 'words', 'bbox'])
        self.pt_data['words'] = self.pt_data['words'].astype('object')
        self.pt_data['bbox'] = self.pt_data['bbox'].astype('object')
        self.encoding = pd.DataFrame()

        #For fine-tuning
        self.label2idx = {}
        self.ft_data = pd.DataFrame()

    def process_json(self, directory):

        #Import Files
        folders = glob.glob(directory + "/*")

        i = 0

        for folder in folders:
            f = glob.glob(folder + "/*.json")[0]
            words = []
            positions = []

            with open(f, 'r') as json_file:
                json_data = json.load(json_file)
                for data in json_data:
                    words.append(data['processed_word'])

                    positions.append([int(data['location']['left'] * 1000), \
                        int((data['location']['top'] - data['location']['height'])* 1000),\
                        int((data['location']['left'] + data['location']['width'])* 1000), \
                        int(data['location']['top']* 1000)])

            self.pt_data.at[i,'image_path'] = f
            self.pt_data.at[i,'words'] = words.copy()
            self.pt_data.at[i,'bbox'] = positions.copy()
            i+=1
        
        return self.pt_data

    def normalize_box(self, box, width, height):
        return [int(1000 * (box[0] / width)), \
                int(1000 * (box[1] / height)), \
                int(1000 * (box[2] / width)), \
                int(1000 * (box[3] / height))]

    def apply_ocr(self, datapoint):
        image = Image.open(datapoint['image_path'])
        width, height = image.size
        
        # apply ocr to the image 
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        # get the words and actual (unnormalized) bounding boxes
        #words = [word for word in ocr_df.text if str(word) != 'nan'])
        words = list(ocr_df.text)
        words = [str(w) for w in words]
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
            actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+width, top+height) to get the actual box 
            actual_boxes.append(actual_box)
        
        # normalize the bounding boxes
        boxes = []

        for box in actual_boxes:
            boxes.append(self.normalize_box(box, width, height))
        
        # add as extra columns 
        assert len(words) == len(boxes)
        datapoint['words'] = words
        datapoint['bbox'] = boxes
        return datapoint

    def process_ftdata(self, in_directory, out_directory):

        folders = [re.match(r"^.+/(.*)", f).group(1) for f in glob.glob(in_directory + '/*')]

        i=0
        for label in folders:

            self.ft_data = pd.DataFrame()

            files = glob.glob(in_directory+'/'+label+'/*') 

            sampled_files = files

            for filepath in sampled_files:
                self.ft_data.at[i,'image_path']=filepath
                self.ft_data.at[i,'label']=label
                i=i+1

            self.ft_data = Dataset.from_pandas(self.ft_data)
            self.ft_data = self.ft_data.map(self.apply_ocr)
            self.ft_data = self.ft_data.remove_columns('__index_level_0__')

            #Output to json file
            self.ft_data.to_pandas().to_pickle(out_directory+'/'+label+'.pkl')
        return

    def encode_example(self, example, finetune = False, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):

        words = example['words']
        normalized_word_boxes = example['bbox']

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
    
        encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]

        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        encoding['bbox'] = np.array(token_boxes)

        if finetune == True: 
            encoding['label'] = self.label2idx[example['label']]

        return encoding

    def get_hidden_state(self, datapoint):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = datapoint["input_ids"].to(device)
        bbox = datapoint["bbox"].to(device)
        attention_mask = datapoint["attention_mask"].to(device)
        token_type_ids = datapoint["token_type_ids"].to(device)

        outputs = self.model(input_ids=input_ids.view(1, -1), bbox=bbox.view(1, -1, 4), attention_mask=attention_mask.view(1, -1), token_type_ids=token_type_ids.view(1, -1))        

        datapoint['last_hidden_state'] = outputs.last_hidden_state[0]

        return datapoint

    def get_encodings(self, outpath, \
        finetune = False, directory = None, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):

        if finetune == True:

            #Load data from pkls
            dfs = glob.glob(directory + "/*")

            i = 0
            for df in dfs:
                label = re.match(r".+/(.*).pkl", df).group(1)
                self.label2idx[label] = i
                i += 1
                self.ft_data = Dataset.from_pandas(pd.read_pickle(df))

                features = Features({
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'label': ClassLabel(num_classes = len(self.label2idx.keys()), names = list(self.label2idx.values())),
                'image_path': Value(dtype='string'),
                'words': Sequence(feature=Value(dtype='string')),})

                self.encodings = self.ft_data.map(lambda example: self.encode_example(example, finetune), features = features)

                self.encodings.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'])

                self.encodings.to_pandas().to_pickle(outpath+'/'+label+'.pkl')

        else:

            self.pt_data = Dataset.from_pandas(self.pt_data)

            self.pt_data = self.pt_data.remove_columns('__index_level_0__')

            features = Features({
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'image_path': Value(dtype='string'),
                'words': Sequence(feature=Value(dtype='string')),})

            self.encodings = self.pt_data.map(lambda example: self.encode_example(example), features = features)

            self.encodings.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'])

            self.encodings = self.encodings.map(lambda example: self.get_hidden_state(example))

            self.encodings.to_pandas().to_pickle(outpath)

        return self.encodings

    def fine_tune(self, input_dir, model_save_path, global_step = 0, num_train_epochs = 5):

        #Bring in data
        dfs = glob.glob(input_dir + "/*")

        datasets = []
        for df in dfs:
            int_df = pd.read_pickle(df)
            int_df['bbox'] = [x.tolist() for x in int_df['bbox']]
            datasets.append(Dataset.from_pandas(int_df))
        
        train_data = concatenate_datasets(datasets)

        train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label', 'bbox'])

        train_size=len(train_data)

        dataloader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Put the model in training mode
        self.model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased", \
            num_labels=len(self.label2idx))

        self.model.to(device)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        global_step = 0
        num_train_epochs = 50
        t_total = len(train_data) * num_train_epochs # total number of training steps 

        for epoch in range(num_train_epochs):
            print("Epoch:", epoch)
            running_loss = 0.0
            correct = 0
            self.model.train()
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                bbox = batch["bbox"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label"].to(device)

                # forward pass
                outputs = self.model(input_ids=input_ids.view(1, -1), bbox=bbox.view(1, -1, 4), \
                    attention_mask=attention_mask.view(1, -1), token_type_ids=token_type_ids.view(1, -1), \
                    labels = labels)

                loss = outputs.loss

                running_loss += loss.item()
                predictions = outputs.logits.argmax(-1)
                correct += (predictions == labels).float().sum()

                # backward pass to get the gradients 
                loss.backward()

                # update
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
            print("Loss:", running_loss / batch["input_ids"].shape[0])
            accuracy = 100 * correct / train_size
            print("Training accuracy:", accuracy.item())

        #Save final model
        torch.save(self.model.state_dict(), model_save_path)
        return


if __name__ == '__main__':

    directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/rivlets'

    #Without fine-tuning, just process json
    i1 = LayoutLM()
    i1.process_json(directory)
    #Included an outpath to save the embeddings as well within the get_encodings method
    outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/encodings/layoutlm_noft_encodings.pkl'
    encodings = i1.get_encodings(outpath)
