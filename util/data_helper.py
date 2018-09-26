# -*- coding: utf-8 -*-
import codecs
import json
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf

def load_data(file_path, min_frequency=0, max_length=0, vocab_processor=None, shuffle=True):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        print('Building dataset ...')
        data = json.load(f)
        sentences = []
        raw_labels = []
        for api in data:
            sentences.append(api['desc'])
            raw_labels.append(api['label'])
        # labels
        labels_set = set(raw_labels)
        labels_dict = dict(zip(labels_set, np.arange(len(labels_set))))
        labels = []
        for label in raw_labels:
            labels.append(labels_dict[label])
        labels = np.array(labels)
        # lengths
        lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in sentences])))
        if max_length == 0:
            max_length = max(lengths)
        # idx_sentences & vocab_processor (Extract vocabulary from sentences and map words to indices)
        if vocab_processor is None:
            vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
            idx_sentences = np.array(list(vocab_processor.fit_transform(sentences)))
        else:
            idx_sentences = np.array(list(vocab_processor.transform(sentences)))
        # data_size
        data_size = len(idx_sentences)
        # shuffle
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            idx_sentences = idx_sentences[shuffle_indices]
            labels = labels[shuffle_indices]
            lengths = lengths[shuffle_indices]
    print('Dataset has been built successfully.')
    print('Number of sentences: {}'.format(data_size))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
    print('Max document length: {}\n'.format(vocab_processor.max_document_length))
    return idx_sentences, labels, lengths, vocab_processor

def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for epoch in range(num_epochs):
        print("New epoch! Epoch " + str(epoch+1))
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length