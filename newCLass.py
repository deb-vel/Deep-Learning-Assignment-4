import numpy as np

input_file = 'text.txt'
input_file_labels = 'labels.txt'
input_test_file = 'test.txt'
input_test_labels = 'testLbls.txt'

labels = []
file = open(input_file_labels)
line = file.readline()
while line:
    strippedLine = line.strip('\n')
    if strippedLine == 'grain':
        label = 0
    elif strippedLine == 'earn':
        label = 1
    elif strippedLine == 'acq':
        label = 2
    elif strippedLine == 'crude':
        label = 3
    elif strippedLine == 'money-fx':
        label = 4
    elif strippedLine == 'interest':
        label = 5
    labels.append(label)
    line = file.readline()
file.close()
print(len(labels))


test_labels = []
file = open(input_test_labels)
line =file.readline()
while line:
    strippedLine = line.strip('\n')
    if strippedLine == 'grain':
        label = 0
    elif strippedLine == 'earn':
        label = 1
    elif strippedLine == 'acq':
        label = 1
    elif strippedLine == 'crude':
        label = 3
    elif strippedLine == 'money-fx':
        label = 4
    elif strippedLine == 'interest':
        label = 5
    test_labels.append(label)
    line = file.readline()
file.close()
print(len(test_labels))

training_data = np.array([])
sentences = []

max_len = 0
sent_lens = []
labels = []

file = open(input_file)
line=file.readline()
while line:
    strippedLine = line.strip('\n')
    words = strippedLine.split(" ")
    sentences.append(words)
    length = len(strippedLine)
    if max_len < length:
        max_len = length
    sent_lens.append(length)
    training_data= np.append(training_data,words)
    line = file.readline()
file.close()

test_data = np.array([])
test_sentences = []
test_sent_lens = []
file = open(input_test_file)
line=file.readline()
while line:
    strippedLine = line.strip('\n')
    words = strippedLine.split(" ")
    test_sentences.append(words)
    length = len(strippedLine)
    if max_len < length:
        max_len = length
    test_sent_lens.append(length)
    test_data= np.append(training_data,words)
    line = file.readline()
file.close()



unique, counts = np.unique(training_data, return_counts=True)

vocab = []
unk = 0
i = 0
for word in unique:
    if counts[i] > 3:
        vocab.append(word)
    else: unk +=1
    i+=1
vocab.append('unk')

token2index = {token: index for (index, token) in enumerate(vocab)}