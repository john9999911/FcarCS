def get_config():
    conf = {
        'workdir': './data/github/',
        'data_params': {
            # training data
            'train_methname': 'train.methodname.pkl',
            'train_apiseq': 'train.apiseq.pkl',
            'train_tokens': 'train.tokens.pkl',
            'train_sbt': 'train.staTree.pkl',
            'train_desc': 'train.desc.pkl',
            # valid data
            'valid_methname': 'test.methodname.pkl',
            'valid_apiseq': 'test.apiseq.pkl',
            'valid_tokens': 'test.tokens.pkl',
            'valid_sbt': 'test.staTree.pkl',
            'valid_desc': 'test.desc.pkl',
            # use data (computing code vectors)
            'use_codebase': 'test_source.txt',  # 'use.rawcode.h5'
            'use_methname': 'use.methname.h5',
            'use_apiseq': 'use.apiseq.h5',
            'use_tokens': 'use.tokens.h5',
            # results data(code vectors)
            'use_codevecs': 'use.codevecs.normalized.h5',  # 'use.codevecs.h5',

            # parameters
            'methname_len': 6,
            'apiseq_len': 30,
            'tokens_len': 50,
            'sbt_len': 150,
            'desc_len': 30,
            'n_methodname_words': 27177,
            'n_desc_words': 30001,  # len(vocabulary) + 1
            'n_tokens_words': 46145,
            'n_api_words': 52892,
            'n_sbt_words': 257507,  # ast 46223 sbt 89 staTree 257507 tokTree 340876 astTree 49191
            # vocabulary info
            'vocab_methname': 'vocab.methodname.pkl',
            'vocab_apiseq': 'vocab.apiseq.pkl',
            'vocab_tokens': 'vocab.tokens.pkl',
            'vocab_sbt': 'vocab.staTree.pkl',
            'vocab_desc': 'vocab.desc.pkl',
        },
        'training_params': {
            # 'batch_size': 128,
            'batch_size': 1024,
            'chunk_size': 100000,
            # 'nb_epoch': 500,
            'nb_epoch': 300,
            'validation_split': 0.1,
            # 'optimizer': 'adam',
            # 'optimizer': Adam(clip_norm=0.1),
            'valid_every': 1,
            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'save_every': 1,
            # 'reload': 0,  # that the model is reloaded from . If reload=0, then train from scratch
            'reload': 184,
        },

        'model_params': {
            'model_name': 'JointEmbeddingModel',
            'n_embed_dims': 100,
            'n_hidden': 400,  # number of hidden dimension of code/desc representation
            # recurrent
            'n_lstm_dims': 200,  # * 2
            'init_embed_weights_methname': None,  # 'word2vec_100_methname.h5',
            'init_embed_weights_tokens': None,  # 'word2vec_100_tokens.h5',
            'init_embed_weights_sbt': None,
            'init_embed_weights_desc': None,  # 'word2vec_100_desc.h5',
            'init_embed_weights_api': None,
            'margin': 0.05,
            'sim_measure': 'cos',  # similarity measure: gesd, cosine, aesd
        }
    }
    return conf
