import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import timeit

train_data_file= 'reuters/data/text.txt'
train_labels_file = 'reuters/data/labels.txt'
test_data_file = 'reuters/test/text.txt'
test_labels_file = 'reuters/test/labels.txt'

start_time = timeit.default_timer()
training_data = np.array([])
sentences = []

train_max_len = 0
sent_lens = []

file = open(train_data_file)
line=file.readline()
while line:
    strippedLine = line.strip('reuter 3\n')
    words = strippedLine.split(" ")
    sentences.append(words)
    length = len(words)
    if train_max_len < length:
        train_max_len = length
    sent_lens.append(length)
    training_data = np.append(training_data,words)
    line = file.readline()
file.close()

test_data = np.array([])
test_sentences = []
test_sent_lens = []
test_max_len = 0
file = open(test_data_file)
line=file.readline()
while line:
    strippedLine = line.strip('\n')
    words = strippedLine.split(" ")
    test_sentences.append(words)
    length = len(words)
    if test_max_len < length:
        test_max_len = length
    test_sent_lens.append(length)
    test_data= np.append(test_data,words)
    line = file.readline()
file.close()

training_labels = []
file = open(train_labels_file)
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
    training_labels.append(label)
    line = file.readline()
file.close()

test_labels = []
file = open(test_labels_file)
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

unique, counts = np.unique(training_data, return_counts=True)
unique= np.append(unique,'unk')

token2index = {token: index for (index, token) in enumerate(unique)}
# print("\n\n\n\n",token2index,"\n\n\n\n")
index_sents = [[token2index[token]  for token in sent] + [0 for _ in range(train_max_len - len(sent))] for sent in
               sentences]

test_index_sents = [[token2index[token] if token in token2index else token2index['unk'] for token in sent] + [0 for _ in range(test_max_len - len(sent))] for sent in
               test_sentences]

###################################

class Model(object):

    def __init__(self, vocab_size):
        init_stddev = 1e-2
        embed_size = 256
        state_size = 256
        dropoutRate = 0.5
        output_size = 6

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sents = tf.placeholder(tf.int32, [None, None], 'sents')
            self.sent_lens = tf.placeholder(tf.int32, [None], 'sent_lens')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')
            self.dropout = tf.placeholder(tf.bool, [], 'dropout')

            dropout_prob = tf.cond(self.dropout, lambda: tf.constant(1.0 - dropoutRate, tf.float32),
                                   lambda: tf.constant(1.0, tf.float32))

            self.params = []
            self.rnn_initialisers = []

            batch_size = tf.shape(self.sents)[0]

            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [vocab_size, embed_size], tf.float32,
                                                        tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([self.embedding_matrix])

                embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.sents)

            with tf.variable_scope('hidden'):
                init_state_fw = tf.get_variable('init_state_fw', [state_size], tf.float32,
                                                tf.random_normal_initializer(stddev=init_stddev))
                init_state_bw = tf.get_variable('init_state_bw', [state_size], tf.float32,
                                                tf.random_normal_initializer(stddev=init_stddev))

                batch_init_fw = tf.tile(tf.reshape(init_state_fw, [1, state_size]), [batch_size, 1])
                batch_init_bw = tf.tile(tf.reshape(init_state_bw, [1, state_size]), [batch_size, 1])

                cell_fw = tf.contrib.rnn.GRUCell(state_size)
                #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob = dropout_prob)
                cell_bw = tf.contrib.rnn.GRUCell(state_size)
                #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob = dropout_prob)

                forward_drop1 = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_prob)
                backward_drop1 = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_prob)

                (_, (self.state_fw, self.state_bw)) = tf.nn.bidirectional_dynamic_rnn(forward_drop1, backward_drop1, embedded,
                                                                                      sequence_length=self.sent_lens,
                                                                                      initial_state_fw=batch_init_fw,
                                                                                      initial_state_bw=batch_init_bw)
                self.states = tf.concat([self.state_fw, self.state_bw], axis=1)
                [W_g, b_g, W_s, b_s] = cell_fw.weights
                [W_g, b_g, W_s, b_s] = cell_bw.weights
                self.rnn_initialisers = [
                    tf.assign(W_g, tf.random_normal([state_size + embed_size, 2 * state_size], stddev=init_stddev)),
                    tf.assign(b_g, tf.zeros([2 * state_size])),
                    tf.assign(W_s, tf.random_normal([state_size + embed_size, state_size], stddev=init_stddev)),
                    tf.assign(b_s, tf.zeros([state_size])),

                    tf.assign(W_g, tf.random_normal([state_size + embed_size, 2 * state_size], stddev=init_stddev)),
                    tf.assign(b_g, tf.zeros([2 * state_size])),
                    tf.assign(W_s, tf.random_normal([state_size + embed_size, state_size], stddev=init_stddev)),
                    tf.assign(b_s, tf.zeros([state_size]))
                ]
                self.params.extend([W_g, b_g, W_s, b_s])


                # init_state1 = tf.get_variable('init_state2', [state_size], tf.float32,
                #                               tf.random_normal_initializer(stddev=init_stddev))
                # batch_init = tf.tile(tf.reshape(init_state1, [1, state_size]), [batch_size, 1])
                #
                # cell = tf.contrib.rnn.GRUCell(state_size)
                # (_, self.output1) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=self.sent_lens,
                #                                       initial_state=batch_init) # Go through the previous RNN's outputs instead of the input words.
                # [W_g, b_g, W_s, b_s] = cell.weights
                # self.rnn_initialisers = [
                #     tf.assign(W_g, tf.random_normal([state_size + embed_size, 2 * state_size], stddev=init_stddev)),
                #     tf.assign(b_g, tf.zeros([2 * state_size])),
                #     tf.assign(W_s, tf.random_normal([state_size + embed_size, state_size], stddev=init_stddev)),
                #     tf.assign(b_s, tf.zeros([state_size]))
                # ]
                # self.params.extend([W_g, b_g, W_s, b_s])

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [2*state_size, output_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [output_size], tf.float32, tf.zeros_initializer())
                self.params.extend([W, b])

                logits = tf.matmul(self.states, W) + b
                self.probs = tf.nn.softmax(logits)

            self.prediction = tf.arg_max(self.probs[0], dimension=0)
            self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits))

            self.optimiser_step = tf.train.AdamOptimizer().minimize(self.error)

            self.init = tf.global_variables_initializer()

            self.graph.finalize()

            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([self.init], {})
        self.sess.run(self.rnn_initialisers, {})

    def close(self):
        self.sess.close()

    def optimisation_step(self, sents, sent_lens, targets):
        return self.sess.run([self.optimiser_step],
                             {self.sents: sents, self.sent_lens: sent_lens, self.targets: targets})

    def get_params(self):
        return self.sess.run(self.params, {})

    def get_error(self, sents, sent_lens, targets):
        return self.sess.run([self.error], {self.sents: sents, self.sent_lens: sent_lens, self.targets: targets})[0]

    def predict(self, sents, sent_lens):
        return self.sess.run([self.probs], {self.sents: sents, self.sent_lens: sent_lens})[0]

    def get_state(self, sents, sent_lens):
        return self.sess.run([self.output1], {self.sents: sents, self.sent_lens: sent_lens})[0]


###################################

max_epochs = 80

(fig, axs) = plt.subplots(1, 1)

sent_plots = list()
sent_texts = list()

[train_error_plot] = axs.plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs.set_xlim(0, max_epochs)
axs.set_xlabel('epoch')
axs.set_ylim(0.0, 2.0)
axs.set_ylabel('XE')
axs.grid(True)
axs.set_title('Error progress')
axs.legend()

fig.tight_layout()
fig.show()

###################################

model = Model(len(unique))

model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs + 1):
    train_error = model.get_error(index_sents, sent_lens, training_labels)
    train_errors.append(train_error)

    if epoch % 10 == 0:
        print(epoch, train_error, sep='\t')

        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()

    model.optimisation_step(index_sents, sent_lens, training_labels)

print()


fig.canvas.set_window_title('Train Error')
fig.tight_layout()
fig.show()

predictions = np.argmax(model.predict(test_index_sents, test_sent_lens),axis=1)
confusionMatrix = confusion_matrix(test_labels,predictions)
print(confusionMatrix)

accuracy = np.sum(predictions == test_labels) / len(test_labels)
num_params = sum(p.size for p in model.get_params())
duration = round((timeit.default_timer() - start_time) / 60, 1)
print("Accuracy: ",accuracy)

(fig, ax) = plt.subplots(1, 1)
ax.matshow(confusionMatrix, cmap='bwr')
ax.set_xlabel('output')
ax.set_ylabel('target')
ax.text(1.0, 0.5, 'Accuracy: {:.2%}\nDuration: {}min\nParams: {}'.format(accuracy, duration, num_params),
        dict(fontsize=10, ha='left', va='center', transform=ax.transAxes))

fig.canvas.set_window_title('Confusion matrix')
fig.tight_layout()
fig.show()

model.close()
input()