# Peter Nguyen
# Prabhdeep Bhullar

# imports
from subprocess import call # run linux commands in python
import argparse # parse command line arguments
from neon.backends import gen_backend # backend
from neon.models import Model # model
import pickle # serialized python object
import numpy as np # numpy arrays
from neon.data.text_preprocessing import clean_string # preprocessing
from builtins import input  # this fixes input() string errors
import glob, os # files
import time # measure elapsed time

# define parameters
sentence_length = 128
vocab_size = 20000

# generate back_end
backend = gen_backend(batch_size=1)
print '\n' + str(backend) + '\n'

# vocab codes
pad_char = 0   # padding character
start = 1      # start of review marker
oov = 2        # when the word is out of vocav
index_from = 3 # index of first word in vocab

# load the vocab
vocab, rec_vocab = pickle.load(open('[0]data/imdb.vocab', 'rb'))

# start timer
start_time = time.time()

# INPUT: load the model
model_name = raw_input('Specify file name and extension--> ')

if model_name == '':
    model_name = 'imdb.p'

# INPIT: how many to validate
how_many_inputs = int(input('How many reviews to validate? (0 for all)--> '))
print ''

# Load model including layers, parameters, and weights
model = Model('[1]model/' + model_name)

# initialize model
model.initialize(dataset = (sentence_length, 1))
 
# CPU-only buffer
input_numpy = np.zeros((sentence_length, 1), dtype = np.int32)
    
correct_predictions = 0
neg_files = 0
total_filecount = 0
neg_predictions = 0
pos_predictions = 0

for file in os.listdir('[0]data/test/neg'):
    if file.endswith('.txt'):
        filename = '[0]data/test/neg/' + str(file)
        file = open(filename, 'r')
        movie_review = file.read()
        file.close()
        
        tokens = clean_string(movie_review).strip().split()

        # preprocess sentence to one hot
        sentence = [len(vocab) + 1 if t not in vocab else vocab[t] for t in tokens]
        sentence = [start] + [w + index_from for w in sentence]
        sentence = [oov if w >= vocab_size else w for w in sentence]
    
        # truncate and padding
        trunc = sentence[-sentence_length:]  # take the last sentence_length words
        input_numpy[:] = 0  # fill with zeros
        input_numpy[-len(trunc):, 0] = trunc   # place the input into the numpy array    
    
        y_pred = model.fprop(input_numpy, inference=True)  # run the forward pass through the model
    
        if (y_pred.get()[1] <= 0.5):
            neg_predictions = neg_predictions + 1
        else:
            pos_predictions = pos_predictions + 1
    
        print '[neg = ' + str(neg_predictions) + '] [pos = ' + str(pos_predictions) + ']' + '\r',

        neg_files = neg_files + 1
        if (how_many_inputs != 0 and neg_files >= how_many_inputs):
            break
        
elapsed_time = time.time() - start_time
print '[neg = ' + str(neg_predictions) + '] [pos = ' + str(pos_predictions) + ']' + ' time: ' + str(elapsed_time) + 's',
print ''
print 'negative reviews file count = ' + str(neg_files)

accuracy = neg_predictions / float(neg_predictions + pos_predictions)
print 'Accuracy negative predictions: ' + str(accuracy) + '\n'

total_filecount = total_filecount + neg_files 
pos_files = 0
correct_predictions = correct_predictions + neg_predictions
neg_predictions = 0
pos_predictions = 0

for file in os.listdir('[0]data/test/pos'):
    if file.endswith('.txt'):
        filename = '[0]data/test/pos/' + str(file)
        file = open(filename, 'r')
        movie_review = file.read()
        file.close()
        
        tokens = clean_string(movie_review).strip().split()

        # preprocess sentence to one hot
        sentence = [len(vocab) + 1 if t not in vocab else vocab[t] for t in tokens]
        sentence = [start] + [w + index_from for w in sentence]
        sentence = [oov if w >= vocab_size else w for w in sentence]
    
        # truncate and padding
        trunc = sentence[-sentence_length:]  # take the last sentence_length words
        input_numpy[:] = 0  # fill with zeros
        input_numpy[-len(trunc):, 0] = trunc   # place the input into the numpy array    
    
        y_pred = model.fprop(input_numpy, inference=True)  # run the forward pass through the model
    
        if (y_pred.get()[1] <= 0.5):
            neg_predictions = neg_predictions + 1
        else:
            pos_predictions = pos_predictions + 1
    
        print '[neg = ' + str(neg_predictions) + '] [pos = ' + str(pos_predictions) + ']' + '\r',
        
        pos_files = pos_files + 1
        if (how_many_inputs != 0 and pos_files >= how_many_inputs):
            break
        
elapsed_time = time.time() - start_time
print '[neg = ' + str(neg_predictions) + '] [pos = ' + str(pos_predictions) + ']' + ' time: ' + str(elapsed_time) + 's',
print ''
print 'positive reviews file count = ' + str(pos_files)

accuracy = pos_predictions / float(neg_predictions + pos_predictions)
print 'Accuracy positive predictions: ' + str(accuracy) + '\n'

total_filecount = total_filecount + pos_files
correct_predictions = correct_predictions + pos_predictions

overall_accuracy = correct_predictions / float(total_filecount)

print 'Correct predictions: ' + str(correct_predictions)
print 'Cumulative file count: ' + str(total_filecount)
print 'Overall accuracy: ' + str(overall_accuracy)

print '\nProgram Terminated'