import numpy
import os
import sys
from main import train

if __name__ == '__main__':
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    train(
    saveto           = '../../models/{}.npz'.format(model_name),
    reload_          = False,
    dim_word         = 300,
    dim              = 300,
    patience         = 7,
    n_words          = 110497,
    n_words_lemma    = 100360,
    decay_c          = 0.,
    clip_c           = 10.,
    lrate            = 0.0004,
    optimizer        = 'adam', 
    maxlen           = 100,
    batch_size       = 32,
    valid_batch_size = 32,
    dispFreq         = 100,
    validFreq        = int(549367/32+1),
    saveFreq         = int(549367/32+1),
    use_dropout      = True,
    verbose          = False,
    datasets         = ['../../data/sequence_and_features/premise_snli_1.0_train_token.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_train_token.txt',
                        '../../data/sequence_and_features/premise_snli_1.0_train_lemma.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_train_lemma.txt',
                        '../../data/sequence_and_features/label_snli_1.0_train.txt'],
    valid_datasets   = ['../../data/sequence_and_features/premise_snli_1.0_dev_token.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_dev_token.txt',
                        '../../data/sequence_and_features/premise_snli_1.0_dev_lemma.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_dev_lemma.txt',
                        '../../data/sequence_and_features/label_snli_1.0_dev.txt'],
    test_datasets    = ['../../data/sequence_and_features/premise_snli_1.0_test_token.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_test_token.txt',
                        '../../data/sequence_and_features/premise_snli_1.0_test_lemma.txt', 
                        '../../data/sequence_and_features/hypothesis_snli_1.0_test_lemma.txt',
                        '../../data/sequence_and_features/label_snli_1.0_test.txt'],
    dictionary       = ['../../data/sequence_and_features/vocab_cased.pkl',
                        '../../data/sequence_and_features/vocab_cased_lemma.pkl'],
    kb_dicts         = ['../../data/sequence_and_features/pair_features.pkl'],
    embedding        = '../../data/glove/glove.840B.300d.txt',
    dim_kb           = 5,
    kb_inference     = True,
    kb_composition   = False,
    attention_lambda = 0,
    )

