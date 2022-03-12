"""
Finetune model.
"""

import argparse
from lib.LayoutLM import LayoutLM
from lib.LayoutLMv2 import LayoutLMv2
from lib.path_utils import existing_directory

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        choices=[
            "finetuned_lmv1",
            "finetuned_lmv2",
        ],
        type=str,
        help="The type of model to finetune on",
    )
    parser.add_argument(
        "task",
        choices=[
            "process_image",
            "get_encodings",
            "finetune",
        ],
        type=str,
        help="Specific task - preprocessing or training",
    )

    parser.add_argument(
        "-i",
        "--images-dir",
        type=existing_directory,
        help="Path to directory containing raw images.",
        default = "datasets/"
    )
    parser.add_argument(
        "-e",
        "--embeddings-dir",
        type=existing_directory,
        help="Path to directory where hidden states will be pickled and stored.",
        default="embeddings/",
    )
    parser.add_argument(
        "-en",
        "--encodings-dir",
        type=existing_directory,
        help="Path to directory where intermediate tokenized encodings will be pickled and stored.",
        default="data/encodings/",
    )
    parser.add_argument(
        "-m",
        "--models-dir",
        type=existing_directory,
        help="Path to directory where models will be stored.",
        default="models/",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        help="Number of training epochs.",
        default=5,
    )

    return parser

def main(args: argparse.Namespace):

    input_dir = args.images_dir
    output_dir = args.embeddings_dir

    if args.model == "finetuned_lmv1":
        #Instatiate instance
        #i1 = LayoutLM()

        #Process images and save pickled data (lengthy)
        #i1.process_images(in_directory, out_directory)
        #outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/test_enc'
        
        #Get encodings (Fast)
        #i1.get_encodings(outpath, finetune = True, directory = out_directory)

        #Finetune model (~25 minutes per epoch for 40k images)
        #model_save_path = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/doc_clustering/models/'
        #i1.fine_tune(outpath, model_save_path, num_train_epochs = 1)
        pass
    
    elif args.model == "finetuned_lmv2":

        i2 = LayoutLMv2()
        
        if args.task == "get_encodings":
            outpath = args.encodings_dir / "layoutlmv2_ft_encodings.pkl"
            dict_path = args.encodings_dir / "layoutlmv2_ft_labels_dict.json"
            hidden_state = i2.get_outputs(input_dir, labels = True, lhs = False, outpath = outpath, dict_path = dict_path, file_type = "tif")
            print(hidden_state)

        elif args.task == "finetune":
            model_save_path = args.models_dir
            input_dir = args.encodings_dir / "layoutlmv2_ft_encodings.pkl"
            labels_dir = args.encodings_dir / "layoutlmv2_ft_labels_dict.json"
            i2.fine_tune(input_dir, labels_dir, model_save_path, num_train_epochs = args.epochs)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)


    


