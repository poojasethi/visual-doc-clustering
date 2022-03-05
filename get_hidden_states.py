from LayoutLM import LayoutLM
from LayoutLMv2 import LayoutLMv2
import glob
import json

if __name__ == '__main__':

    #PUT DIRECTORY OF RIVLETS HERE
    directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/rivlets'

    #OUTPUT VANILLA LM HIDDEN STATES
    i1 = LayoutLM()
    int_data = i1.process_json(directory, 'processed_word', 'location', position_processing = True)
    outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/encodings/layoutlm_noft_encodings.pkl'
    encodings = i1.get_encodings()
    hidden_state = i1.get_hidden_state(outpath = outpath)
    print(hidden_state.to_pandas())

    #OUTPUT FINE-TUNED HIDDEN STATES (RELATED TASK)
    i2 = LayoutLM()
    i2.process_json(directory, 'processed_word', 'location', position_processing = True)
    outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/encodings/layoutlm_ft_encodings.pkl'
    model_path = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/models/fine_tuned_related/epoch7'
    encodings = i2.get_encodings()
    hidden_state = i2.get_hidden_state(outpath = outpath, model_path = model_path)
    print(hidden_state.to_pandas())

    #OUTPUT FINE-TUNED HIDDEN STATES (UNRELATED TASK)
    i3 = LayoutLM()
    i3.process_json(directory, 'processed_word', 'location', position_processing = True)
    outpath = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/encodings/layoutlm_ft_ur_encodings.pkl'
    model_path = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/models/fine_tuned_unrelated/epoch15'
    encodings = i3.get_encodings()
    hidden_state = i3.get_hidden_state(outpath = outpath, model_path = model_path)
    print(hidden_state.to_pandas())

    #OUTPUT VANILLA LM V2 HIDDEN STATES
    i4 = LayoutLMv2()
    directory = '/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/files'
    hidden_state = i4.get_outputs(directory)
    print(hidden_state)
