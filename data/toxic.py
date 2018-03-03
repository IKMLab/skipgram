import glovar
import os
import pandas as pd
import nltk
import pickling
import string
import collections


def get_contexts(m):
    if not pickling.exists(glovar.DATA_DIR, 'contexts_%s' % m):
        vocab = get_vocab()
        contexts = {k: set() for k in range(len(vocab))}
        for x in get_token_lists():
            for i, center in enumerate(x):
                center = vocab[center]
                left_context = [vocab[t] for t in x[max(0, i - m):i - 1]]
                right_context = [vocab[t] for t in x[i + 1: min(len(x), i + m)]]
                contexts[center].update(left_context + right_context)
        pickling.save(contexts, glovar.DATA_DIR, 'contexts_%s' % m)
        return contexts
    else:
        return pickling.load(glovar.DATA_DIR, 'contexts_%s' % m)


def get_frequencies():
    if not pickling.exists(glovar.DATA_DIR, 'frequencies'):
        frequencies = collections.Counter()
        vocab = get_vocab()
        for x in get_token_lists():
            frequencies.update([vocab[t] for t in x])
        pickling.save(frequencies, glovar.DATA_DIR, 'frequencies')
    else:
        return pickling.load(glovar.DATA_DIR, 'frequencies')


def get_full_train_test():
    train_file_path = os.path.join(glovar.DATA_DIR, 'train_full.csv')
    test_file_path = os.path.join(glovar.DATA_DIR, 'test_full.csv')
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    return train, test


def get_token_lists():
    if not pickling.exists(glovar.DATA_DIR, 'token_lists'):
        train, test = get_train_test()
        token_lists = [nltk.word_tokenize(c)
                       for c in train['comment_text'].values] \
                    + [nltk.word_tokenize(c)
                       for c in test['comment_text'].values]
        pickling.save(token_lists, glovar.DATA_DIR, 'token_lists')
        return token_lists
    else:
        return pickling.load(glovar.DATA_DIR, 'token_lists')


def get_token_set():
    if not pickling.exists(glovar.DATA_DIR, 'token_set'):
        train, test = get_train_test()
        data = train.append(test)
        comments = data['comment_text'].values
        token_set = set([t for tokens
                         in [nltk.word_tokenize(c) for c in comments]
                        for t in tokens])
        pickling.save(token_set, glovar.DATA_DIR, 'token_set')
        return token_set
    else:
        return pickling.load(glovar.DATA_DIR, 'token_set')


def get_train_test():
    train_file_path = os.path.join(glovar.DATA_DIR, 'train.csv')
    test_file_path = os.path.join(glovar.DATA_DIR, 'test.csv')
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    return train, test


def get_vocab():
    if not pickling.exists(glovar.DATA_DIR, 'vocab'):
        token_set = get_token_set()
        vocab = dict(zip(token_set, range(1, len(token_set) + 1)))
        pickling.save(vocab, glovar.DATA_DIR, 'vocab')
        return vocab
    else:
        return pickling.load(glovar.DATA_DIR, 'vocab')


def get_wanted_dfs():
    train, test = get_full_train_test()
    train = train[train['comment_text'].apply(wanted)]
    test = test[test['comment_text'].apply(wanted)]
    return train, test


def prepare():
    """Prepares all background data."""
    print('Preparing necessary preliminaries...')
    get_token_set()
    get_vocab()
    get_token_lists()
    get_contexts(5)
    get_frequencies()
    print('Success.')


def wanted(comment):
    wanted_chars = string.ascii_letters + '?!".,*' + "'"
    unwanted_substrs = ['www.', '.com', 'http']
    return all((all(c in wanted_chars for c in token)
                and len(token) < 15
                and all(s not in token for s in unwanted_substrs))
               for token in nltk.word_tokenize(comment))
