'''
Summary a source file using a summarization model.
'''
import argparse
import theano
import numpy
import cPickle as pkl
import os
from data_iterator import TextIterator

from main import (build_model, pred_probs, prepare_data, pred_acc, load_params, init_params, init_tparams)

def main():
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    model = '../../models/{}.npz'.format(model_name)
    valid_datasets   = ['../../data/sequence_and_features/premise_snli_1.0_dev_token.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_dev_token.txt',
                        '../../data/sequence_and_features/premise_snli_1.0_dev_lemma.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_dev_lemma.txt',
                        '../../data/sequence_and_features/label_snli_1.0_dev.txt']
    test_datasets    = ['../../data/sequence_and_features/premise_snli_1.0_test_token.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_test_token.txt',
                        '../../data/sequence_and_features/premise_snli_1.0_test_lemma.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_test_lemma.txt',
                        '../../data/sequence_and_features/label_snli_1.0_test.txt']
    dictionary       = ['../../data/sequence_and_features/vocab_cased.pkl',
                        '../../data/sequence_and_features/vocab_cased_lemma.pkl']
    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    print options
    # load dictionary and invert
    with open(dictionary[0], 'rb') as f:
        word_dict = pkl.load(f)

    print 'Loading knowledge base ...'
    kb_dicts = options['kb_dicts']
    with open(kb_dicts[0], 'rb') as f:
        kb_dict = pkl.load(f)

    n_words = options['n_words']
    valid_batch_size = options['valid_batch_size']

    valid = TextIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2], valid_datasets[3], valid_datasets[4],
                         dictionary[0], dictionary[1],
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
    test = TextIterator(test_datasets[0], test_datasets[1], test_datasets[2], test_datasets[3], test_datasets[4],
                         dictionary[0], dictionary[1],
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)

    # allocate model parameters
    params = init_params(options, word_dict)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x1, x1_mask, x1_kb, x2, x2_mask, x2_kb, kb_att, y, \
        opt_ret, \
        cost, \
        f_pred, \
        f_probs = \
        build_model(tparams, options)

    use_noise.set_value(0.)
    valid_acc = pred_acc(f_pred, prepare_data, options, valid, kb_dict)
    test_acc = pred_acc(f_pred, prepare_data, options, test, kb_dict)

    print 'valid accuracy', valid_acc
    print 'test accuracy', test_acc

    predict_labels_valid = pred_label(f_pred, prepare_data, options, valid, kb_dict)
    predict_labels_test = pred_label(f_pred, prepare_data, options, test, kb_dict)

    with open('predict_gold_samples_valid.txt', 'w') as fw:
        with open(valid_datasets[0], 'r') as f1:
            with open(valid_datasets[1], 'r') as f2:
                with open(valid_datasets[-1], 'r') as f3:
                    for a, b, c, d in zip(predict_labels_valid, f3, f1, f2):
                        fw.write(str(a) + '\t' + b.rstrip() + '\t' + c.rstrip() + '\t' + d.rstrip() + '\n')

    with open('predict_gold_samples_test.txt', 'w') as fw:
        with open(test_datasets[0], 'r') as f1:
            with open(test_datasets[1], 'r') as f2:
                with open(test_datasets[-1], 'r') as f3:
                    for a, b, c, d in zip(predict_labels_test, f3, f1, f2):
                        fw.write(str(a) + '\t' + b.rstrip() + '\t' + c.rstrip() + '\t' + d.rstrip() + '\n')

    print 'Done'

def pred_label(f_pred, prepare_data, options, iterator, kb_dict):
    labels = []
    for x1, x2, x1_lemma, x2_lemma, y in iterator:
        x1, x1_mask, x1_kb, x2, x2_mask, x2_kb, kb_att, y = prepare_data(x1, x2, x1_lemma, x2_lemma, y, options, kb_dict)
        preds = f_pred(x1, x1_mask, x1_kb, x2, x2_mask, x2_kb, kb_att)
        labels = labels + preds.tolist()

    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
