from LayoutLM import LayoutLM

if __name__ == '__main__':

    in_directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/test'
    out_directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/test_int'
    
    #Instatiate instance
    i2 = LayoutLM()

    #Process images and save pickled data (lengthy)
    i2.process_images(in_directory, out_directory)
    outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/test_enc'
    
    #Get encodings (Fast)
    i2.get_encodings(outpath, finetune = True, directory = out_directory)

    #Finetune model (~25 minutes per epoch for 40k images)
    model_save_path = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/models/'
    i2.fine_tune(outpath, model_save_path, num_train_epochs = 1)
