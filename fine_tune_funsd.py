from LayoutLM import LayoutLM
import glob
import json

if __name__ == '__main__':

    directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/funsd/combined_data/annotations'
    
    #Process dataset
    i1 = LayoutLM()
    processed_data = i1.process_json(directory, 'text', 'box', 'label',  token = True)

    #Encode data
    outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/funsd_enc'
    labels_dict = {'question':0, 'answer':1, 'header':2, 'other':3}
    encoded_data = i1.get_encodings(outpath, labels = labels_dict)

    #Fine-tune dataset
    model_save_path = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/models/fine_tuned_unrelated'
    i1.fine_tune(outpath, model_save_path, token = True, save_epoch = 1, batch_size = 2, num_train_epochs=10)
