import os
import pickle

data_dir = 'nuswide'
inp_name = 'nuswide_glove_word2vec.pkl'

inp_file = os.path.join(data_dir, inp_name)
with open(inp_file, 'rb') as f:
    inp = pickle.load(f)
print('hello')