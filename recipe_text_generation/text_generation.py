from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import itertools

# read in data and preprocess
# source: https://github.com/fictivekin/openrecipes
with open('recipeitems-latest.json', 'rb') as f:
    raw_dat = f.readlines()

# remove the trailing "\n" from each line
raw_dat = [x.strip() for x in raw_dat]
raw_dat = [json.loads(x.decode('utf-8')) for x in raw_dat]

# convert list of dictionnaries to dataframe
data = pd.DataFrame(raw_dat)

# reduce name of recipes to lowercase
recipes = [x.lower() for x in data['name'].tolist()]
recipes = '. '.join(recipes)
#words = list(itertools.chain.from_iterable(recipes))

# create mapping of unique chars to integers
chars = sorted(list(set(recipes)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# print out characteristics of our dataset
n_chars = len(recipes)
n_vocab = len(chars)
print("Total Characters: {}".format(n_chars))
print("Total Vocab: {}".format(n_vocab))


# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = recipes[i:i + seq_length]
    seq_out = recipes[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])


n_patterns = len(dataX)
print("Total Patterns: {}".format(n_patterns))



# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=2)

