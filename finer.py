import itertools
import logging
import os
import time
import re
import datasets
import numpy as np
import tensorflow as tf
import wandb

from copy import deepcopy
from tqdm import tqdm
from gensim.models import KeyedVectors
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoTokenizer
from wandb.keras import WandbCallback

from configurations import Configuration
from data import DATA_DIR, VECTORS_DIR
from models import BiLSTM, Transformer, TransformerBiLSTM
from models.callbacks import ReturnBestEarlyStopping, F1MetricCallback

LOGGER = logging.getLogger(__name__)


class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, dataset, vectorize_fn, batch_size=8, max_length=128, shuffle=False):
        self.dataset = dataset
        self.vectorize_fn = vectorize_fn
        self.batch_size = batch_size
        if Configuration['general_parameters']['debug']:
            self.indices = np.arange(Configuration['general_parameters']['debug'])
        else:
            self.indices = np.arange(len(dataset))
        self.max_length = max_length
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Denotes the numbers of batches per epoch"""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + targets
        samples = self.dataset[indices]

        x_batch, y_batch = self.vectorize_fn(samples=samples, max_length=self.max_length)
        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class FINER:
    def __init__(self):
        self.general_params = Configuration['general_parameters']
        self.train_params = Configuration['train_parameters']
        self.hyper_params = Configuration['hyper_parameters']
        self.eval_params = Configuration['evaluation']
        self.tag2idx, self.idx2tag = FINER.load_dataset_tags()
        self.n_classes = len(self.tag2idx)

        if Configuration['task']['mode'] == 'train':
            display_name = Configuration['task']['log_name']
            if Configuration['task']['model'] == 'transformer':
                display_name = f"{display_name}_{self.train_params['model_name']}".replace('/', '-')
            elif Configuration['task']['model'] == 'bilstm':
                display_name = f"{display_name}_bilstm_{self.train_params['embeddings']}"
            wandb.init(
                entity=self.general_params['wandb_entity'],
                project=self.general_params['wandb_project'],
                id=Configuration['task']['log_name'],
                name=display_name
            )

        shape_special_tokens_path = os.path.join(DATA_DIR, 'shape_special_tokens.txt')
        with open(shape_special_tokens_path) as fin:
            self.shape_special_tokens = [shape.strip() for shape in fin.readlines()]
        self.shape_special_tokens_set = set(self.shape_special_tokens)

        if Configuration['task']['model'] == 'bilstm':
            if 'subword' in self.train_params['embeddings']:
                self.train_params['token_type'] = 'subword'
            else:
                self.train_params['token_type'] = 'word'

            word_vector_path = os.path.join(VECTORS_DIR, self.train_params['embeddings'])
            if not os.path.exists(word_vector_path):
                import wget
                url = f"https://zenodo.org/record/6571000/files/{self.train_params['embeddings']}"
                wget.download(url=url, out=word_vector_path)
                if not os.path.exists(word_vector_path):
                    raise Exception(f"Unable to download {self.train_params['embeddings']} embeddings")

            if word_vector_path.endswith('.vec') or word_vector_path.endswith('.txt'):
                word2vector = KeyedVectors.load_word2vec_format(word_vector_path, binary=False)
            else:
                word2vector = KeyedVectors.load_word2vec_format(word_vector_path, binary=True)

            if self.train_params['token_type'] == 'subword':
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w') as tmp:
                    vocab_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]'] + list(word2vector.index_to_key)
                    tmp.write('\n'.join(vocab_tokens))

                    additional_special_tokens = []
                    if 'num' in self.train_params['embeddings']:
                        additional_special_tokens.append('[NUM]')
                    elif 'shape' in self.train_params['embeddings']:
                        additional_special_tokens.append('[NUM]')
                        additional_special_tokens.extend(self.shape_special_tokens)
                    # TODO: Check AutoTokenizer
                    self.tokenizer = BertTokenizer(
                        vocab_file=tmp.name,
                        use_fast=self.train_params['use_fast_tokenizer']
                    )
                    if additional_special_tokens:
                        self.tokenizer.additional_special_tokens = additional_special_tokens

            if self.train_params['token_type'] == 'word':
                self.word2index = {'[PAD]': 0, '[UNK]': 1}
                self.word2index.update({word: i + 2 for i, word in enumerate(word2vector.index_to_key)})

                self.word2vector_weights = np.concatenate(
                    [
                        np.mean(word2vector.vectors, axis=0).reshape((1, word2vector.vectors.shape[-1])),
                        word2vector.vectors
                    ],
                    axis=0
                )
                self.word2vector_weights = np.concatenate(
                    [
                        np.zeros((1, self.word2vector_weights.shape[-1]), dtype=np.float32),
                        self.word2vector_weights
                    ],
                    axis=0
                )

            if self.train_params['token_type'] == 'subword':
                self.word2index = {'[PAD]': 0}
                self.word2index.update({word: i + 1 for i, word in enumerate(word2vector.index_to_key)})

                self.word2vector_weights = np.concatenate(
                    [
                        np.zeros((1, word2vector.vectors.shape[-1]), dtype=np.float32),
                        word2vector.vectors
                    ],
                    axis=0
                )

            self.index2word = {v: k for k, v in self.word2index.items()}

        elif Configuration['task']['model'] == 'transformer':
            additional_special_tokens = []
            if self.train_params['replace_numeric_values']:
                additional_special_tokens.append('[NUM]')
            if self.train_params['replace_numeric_values'] == 'SHAPE':
                additional_special_tokens.extend(self.shape_special_tokens)

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.train_params['model_name'],
                additional_special_tokens=additional_special_tokens,
                use_fast=self.train_params['use_fast_tokenizer']
            )

    @staticmethod
    def load_dataset_tags():

        dataset = datasets.load_dataset('nlpaueb/finer-139', split='train', streaming=True)
        dataset_tags = dataset.features['ner_tags'].feature.names
        tag2idx = {tag: int(i) for i, tag in enumerate(dataset_tags)}
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}

        return tag2idx, idx2tag

    def is_numeric_value(self, text):
        digits, non_digits = 0, 0

        for char in str(text):
            if char.isdigit():
                digits = digits + 1
            else:
                non_digits += 1

        return (digits + 1) > non_digits

    def vectorize(self, samples, max_length):
        if Configuration['task']['model'] == 'bilstm' and self.train_params['token_type'] == 'word':

            sample_tokens = [
                [
                    token.lower()
                    for token in sample
                ]
                for sample in samples['tokens']
            ]

            if 'word.num' in self.train_params['embeddings']:
                sample_tokens = [
                    [
                        '[NUM]' if re.fullmatch(r'(\d+[\d,.]*)|([,.]\d+)', token)
                        else token
                        for token in sample
                    ]
                    for sample in sample_tokens
                ]

            elif 'word.shape' in self.train_params['embeddings']:
                for sample_idx, _ in enumerate(sample_tokens):
                    for token_idx, _ in enumerate(sample_tokens[sample_idx]):
                        if re.fullmatch(r'(\d+[\d,.]*)|([,.]\d+)', sample_tokens[sample_idx][token_idx]):
                            shape = '[' + re.sub(r'\d', 'X', sample_tokens[sample_idx][token_idx]) + ']'
                            if shape in self.shape_special_tokens_set:
                                sample_tokens[sample_idx][token_idx] = shape
                            else:
                                sample_tokens[sample_idx][token_idx] = '[NUM]'

            word_indices = [
                [
                    self.word2index[token]
                    if token in self.word2index
                    else self.word2index['[UNK]']
                    for token in sample
                ]
                for sample in sample_tokens
            ]

            word_indices = pad_sequences(
                sequences=word_indices,
                maxlen=max_length,
                padding='post',
                truncating='post'
            )
            x = word_indices

        elif Configuration['task']['model'] == 'transformer' \
                or (Configuration['task']['model'] == 'bilstm' and self.train_params['token_type'] == 'subword'):

            sample_tokens = samples['tokens']

            sample_labels = samples['ner_tags']

            batch_token_ids, batch_tags, batch_subword_pooling_mask = [], [], []

            for sample_idx in range(len(sample_tokens)):

                sample_token_ids, sample_tags, subword_pooling_mask = [], [], []

                sample_token_idx = 1  # idx 0 is reserved for [CLS]
                for token_idx in range(len(sample_tokens[sample_idx])):

                    if (Configuration['task']['model'] == 'transformer' and self.train_params['model_name'] == 'nlpaueb/sec-bert-num') \
                            or (Configuration['task']['model'] == 'bilstm' and 'subword.num' in self.train_params['embeddings']):
                        if re.fullmatch(r'(\d+[\d,.]*)|([,.]\d+)', sample_tokens[sample_idx][token_idx]):
                            sample_tokens[sample_idx][token_idx] = '[NUM]'

                    if (Configuration['task']['model'] == 'transformer' and self.train_params['model_name'] == 'nlpaueb/sec-bert-shape') \
                            or (Configuration['task']['model'] == 'bilstm' and 'subword.shape' in self.train_params['embeddings']):
                        if re.fullmatch(r'(\d+[\d,.]*)|([,.]\d+)', sample_tokens[sample_idx][token_idx]):
                            shape = '[' + re.sub(r'\d', 'X', sample_tokens[sample_idx][token_idx]) + ']'
                            if shape in self.shape_special_tokens_set:
                                sample_tokens[sample_idx][token_idx] = shape
                            else:
                                sample_tokens[sample_idx][token_idx] = '[NUM]'

                    if self.train_params['replace_numeric_values']:
                        if self.is_numeric_value(sample_tokens[sample_idx][token_idx]):
                            if re.fullmatch(r'(\d+[\d,.]*)|([,.]\d+)', sample_tokens[sample_idx][token_idx]):
                                if self.train_params['replace_numeric_values'] == 'NUM':
                                    sample_tokens[sample_idx][token_idx] = '[NUM]'
                                elif self.train_params['replace_numeric_values'] == 'SHAPE':
                                    shape = '[' + re.sub(r'\d', 'X', sample_tokens[sample_idx][token_idx]) + ']'
                                    if shape in self.shape_special_tokens_set:
                                        sample_tokens[sample_idx][token_idx] = shape
                                    else:
                                        sample_tokens[sample_idx][token_idx] = '[NUM]'

                    token = sample_tokens[sample_idx][token_idx]

                    # Subword pooling (As in BERT or Acs et al.)
                    if 'subword_pooling' in self.train_params:
                        label_to_assign = self.idx2tag[sample_labels[sample_idx][token_idx]]
                        if self.train_params['subword_pooling'] == 'all':  # First token is B-, rest are I-
                            if label_to_assign.startswith('B-'):
                                remaining_labels = 'I' + label_to_assign[1:]
                            else:
                                remaining_labels = label_to_assign
                        elif self.train_params['subword_pooling'] in ['first', 'last']:
                            remaining_labels = 'O'
                        else:
                            raise Exception(f'Choose a valid subword pooling ["all", "first" and "last"] in the train parameters.')

                    # Assign label to all (multiple) generated tokens, if any
                    token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
                    sample_token_idx += len(token_ids)
                    sample_token_ids.extend(token_ids)

                    for i in range(len(token_ids)):

                        if self.train_params['subword_pooling'] in ['first', 'all']:
                            if i == 0:
                                sample_tags.append(label_to_assign)
                                subword_pooling_mask.append(1)
                            else:
                                if self.train_params['subword_pooling'] == 'first':
                                    subword_pooling_mask.append(0)
                                sample_tags.append(remaining_labels)
                        elif self.train_params['subword_pooling'] == 'last':
                            if i == len(token_ids) - 1:
                                sample_tags.append(label_to_assign)
                                subword_pooling_mask.append(1)
                            else:
                                sample_tags.append(remaining_labels)
                                subword_pooling_mask.append(0)

                if Configuration['task']['model'] == 'transformer':  # if 'bert' in self.general_params['token_type']:
                    CLS_ID = self.tokenizer.vocab['[CLS]']
                    SEP_ID = self.tokenizer.vocab['[SEP]']
                    PAD_ID = self.tokenizer.vocab['[PAD]']
                    sample_token_ids = [CLS_ID] + sample_token_ids + [SEP_ID]
                    sample_tags = ['O'] + sample_tags + ['O']
                    subword_pooling_mask = [1] + subword_pooling_mask + [1]

                # Append to batch_token_ids & batch_tags
                batch_token_ids.append(sample_token_ids)
                batch_tags.append(sample_tags)
                batch_subword_pooling_mask.append(subword_pooling_mask)

            if Configuration['task']['model'] == 'bilstm' and self.train_params['token_type'] == 'subword':
                for sent_idx, _ in enumerate(batch_token_ids):
                    for tok_idx, _ in enumerate(batch_token_ids[sent_idx]):
                        token_subword = self.tokenizer.convert_ids_to_tokens(
                            batch_token_ids[sent_idx][tok_idx], skip_special_tokens=True)
                        batch_token_ids[sent_idx][tok_idx] = self.word2index[token_subword] \
                            if token_subword in self.word2index else self.word2index['[UNK]']

            # Pad, truncate and verify
            # Returns an np.array object of shape ( len(batch_size) x max_length ) that contains padded/truncated gold labels
            batch_token_ids = pad_sequences(
                sequences=batch_token_ids,
                maxlen=max_length,
                padding='post',
                truncating='post'
            )

            # Replace last column with SEP special token if it's not PAD
            if Configuration['task']['model'] == 'transformer':
                batch_token_ids[np.where(batch_token_ids[:, -1] != PAD_ID)[0], -1] = SEP_ID

            x = batch_token_ids

        else:
            x = None

        if Configuration['task']['model'] == 'bilstm' and self.train_params['token_type'] == 'word':

            y = pad_sequences(
                sequences=samples['ner_tags'],
                maxlen=max_length,
                padding='post',
                truncating='post'
            )

        elif Configuration['task']['model'] == 'transformer' \
                or (Configuration['task']['model'] == 'bilstm' and self.train_params['token_type'] == 'subword'):

            batch_tags = [[self.tag2idx[tag] for tag in sample_tags] for sample_tags in batch_tags]

            # Pad/Truncate the rest tags/labels
            y = pad_sequences(
                sequences=batch_tags,
                maxlen=max_length,
                padding='post',
                truncating='post'
            )

            if Configuration['task']['model'] == 'transformer':
                y[np.where(x[:, -1] != PAD_ID)[0], -1] = 0

        if self.train_params['subword_pooling'] in ['first', 'last']:
            batch_subword_pooling_mask = pad_sequences(
                sequences=batch_subword_pooling_mask,
                maxlen=max_length,
                padding='post',
                truncating='post'
            )

            return [np.array(x), batch_subword_pooling_mask], y
        else:
            return np.array(x), y

    def build_model(self, train_params=None):
        if Configuration['task']['model'] == 'bilstm':
            model = BiLSTM(
                n_classes=self.n_classes,
                n_layers=train_params['n_layers'],
                n_units=train_params['n_units'],
                dropout_rate=train_params['dropout_rate'],
                crf=train_params['crf'],
                word2vectors_weights=self.word2vector_weights,
            )

        elif Configuration['task']['model'] == 'transformer':
            model = Transformer(
                model_name=train_params['model_name'],
                n_classes=self.n_classes,
                dropout_rate=train_params['dropout_rate'],
                crf=train_params['crf'],
                tokenizer=self.tokenizer if self.train_params['replace_numeric_values'] else None,
                subword_pooling=self.train_params['subword_pooling']
            )
        elif Configuration['task']['model'] == 'transformer_bilstm':
            model = TransformerBiLSTM(
                model_name=train_params['model_name'],
                n_classes=self.n_classes,
                dropout_rate=train_params['dropout_rate'],
                crf=train_params['crf'],
                n_layers=train_params['n_layers'],
                n_units=train_params['n_units'],
                tokenizer=self.tokenizer if self.train_params['replace_numeric_values'] else None,
            )

        else:
            raise Exception(f"The model type that you entered isn't a valid one.")

        return model

    def get_monitor(self):
        monitor_metric = self.general_params['loss_monitor']
        if monitor_metric == 'val_loss':
            monitor_mode = 'min'
        elif monitor_metric in ['val_micro_f1', 'val_macro_f1']:
            monitor_mode = 'max'
        else:
            raise Exception(f'Unrecognized monitor: {self.general_params["loss_monitor"]}')

        return monitor_metric, monitor_mode

    def train(self):

        train_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='train')
        train_generator = DataLoader(
            dataset=train_dataset,
            vectorize_fn=self.vectorize,
            batch_size=self.general_params['batch_size'],
            max_length=self.train_params['max_length'],
            shuffle=True
        )

        validation_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='validation')
        validation_generator = DataLoader(
            dataset=validation_dataset,
            vectorize_fn=self.vectorize,
            batch_size=self.general_params['batch_size'],
            max_length=self.train_params['max_length'],
            shuffle=False
        )

        test_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='test')
        test_generator = DataLoader(
            dataset=test_dataset,
            vectorize_fn=self.vectorize,
            batch_size=self.general_params['batch_size'],
            max_length=self.train_params['max_length'],
            shuffle=False
        )

        train_params = deepcopy(self.train_params)
        train_params.update(self.hyper_params)

        # Build model
        model = self.build_model(train_params=train_params)
        LOGGER.info('Model Summary')
        model.print_summary(print_fn=LOGGER.info)

        optimizer = tf.keras.optimizers.Adam(learning_rate=train_params['learning_rate'], clipvalue=5.0)

        if train_params['crf']:
            model.compile(
                optimizer=optimizer,
                loss=model.crf_layer.loss,
                run_eagerly=self.general_params['run_eagerly']
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                run_eagerly=self.general_params['run_eagerly']
            )

        monitor, monitor_mode = self.get_monitor()

        # Init callbacks
        callbacks = []

        f1_metric = F1MetricCallback(
            train_params=train_params,
            idx2tag=self.idx2tag,
            validation_generator=validation_generator,
            subword_pooling=self.train_params['subword_pooling'],
            calculate_train_metric=False
        )
        callbacks.append(f1_metric)

        callbacks.append(
            ReturnBestEarlyStopping(
                monitor=monitor,
                mode=monitor_mode,
                patience=self.general_params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            )
        )

        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=monitor_mode,
                factor=0.5,
                cooldown=self.general_params['reduce_lr_cooldown'],
                patience=self.general_params['reduce_lr_patience'],
                verbose=1
            )
        )

        if Configuration['task']['model'] == 'transformer':
            wandb.config.update(
                {
                    'model': 'transformer',
                    'model_name': self.train_params['model_name'],
                }
            )
        elif Configuration['task']['model'] == 'bilstm':
            wandb.config.update(
                {
                    'model': 'bilstm',
                    'embedddings': self.train_params['embeddings'],
                }
            )

        wandb.config.update(
            {
                'max_length': self.train_params['max_length'],
                'replace_numeric_values': self.train_params['replace_numeric_values'],
                'subword_pooling': self.train_params['subword_pooling'],
                'epochs': self.general_params['epochs'],
                'batch_size': self.general_params['batch_size'],
                'loss_monitor': self.general_params['loss_monitor'],
                'early_stopping_patience': self.general_params['early_stopping_patience'],
                'reduce_lr_patience': self.general_params['reduce_lr_patience'],
                'reduce_lr_cooldown': self.general_params['reduce_lr_cooldown']
            }
        )
        wandb.config.update(self.hyper_params)

        callbacks.append(
            WandbCallback(
                monitor=monitor,
                mode=monitor_mode,

            )
        )

        # Train model
        start = time.time()
        history = model.fit(
            x=train_generator,
            validation_data=validation_generator,
            callbacks=callbacks,
            epochs=self.general_params['epochs'],
            workers=self.general_params['workers'],
            max_queue_size=self.general_params['max_queue_size'],
            use_multiprocessing=self.general_params['use_multiprocessing']
        )

        # Loss Report
        self.loss_report(history.history)

        # Save model
        weights_save_path = os.path.join(Configuration['experiment_path'], 'model', 'weights.h5')
        LOGGER.info(f'Saving model weights to {weights_save_path}')
        model.save_weights(filepath=weights_save_path)

        # Evaluate
        self.evaluate(model, validation_generator, split_type='validation')
        self.evaluate(model, test_generator, split_type='test')

        training_time = time.time() - start
        training_days = int(training_time / (24 * 60 * 60))
        if training_days:
            LOGGER.info(f'Training time: {training_days} days {time.strftime("%H:%M:%S", time.gmtime(training_time))} sec\n')
        else:
            LOGGER.info(f'Training time: {time.strftime("%H:%M:%S", time.gmtime(training_time))} sec\n')

    def evaluate(self, model, generator, split_type):
        """
        :param model: the trained TF model
        :param generator: the generator for the split type to evaluate on
        :param split_type: validation or test
        :return:
        """

        LOGGER.info(f'\n{split_type.capitalize()} Evaluation\n{"-" * 30}\n')
        LOGGER.info('Calculating predictions...')

        y_true, y_pred = [], []

        for x_batch, y_batch in tqdm(generator, ncols=100):

            if self.train_params['subword_pooling'] in ['first', 'last']:
                pooling_mask = x_batch[1]
                x_batch = x_batch[0]
                y_prob_temp = model.predict(x=[x_batch, pooling_mask])
            else:
                pooling_mask = x_batch
                y_prob_temp = model.predict(x=x_batch)

            # Get lengths and cut results for padded tokens
            lengths = [len(np.where(x_i != 0)[0]) for x_i in x_batch]

            if model.crf:
                y_pred_temp = y_prob_temp.astype('int32')
            else:
                y_pred_temp = np.argmax(y_prob_temp, axis=-1)

            for y_true_i, y_pred_i, l_i, p_i in zip(y_batch, y_pred_temp, lengths, pooling_mask):

                if Configuration['task']['model'] == 'transformer':
                    if self.train_params['subword_pooling'] in ['first', 'last']:
                        y_true.append(np.take(y_true_i, np.where(p_i != 0)[0])[1:-1])
                        y_pred.append(np.take(y_pred_i, np.where(p_i != 0)[0])[1:-1])
                    else:
                        y_true.append(y_true_i[1:l_i - 1])
                        y_pred.append(y_pred_i[1:l_i - 1])

                elif Configuration['task']['model'] == 'bilstm':
                    if self.train_params['subword_pooling'] in ['first', 'last']:
                        y_true.append(np.take(y_true_i, np.where(p_i != 0)[0]))
                        y_pred.append(np.take(y_pred_i, np.where(p_i != 0)[0]))
                    else:
                        y_true.append(y_true_i[:l_i])
                        y_pred.append(y_pred_i[:l_i])

        # Indices to labels in one flattened list
        seq_y_pred_str = []
        seq_y_true_str = []

        for y_pred_row, y_true_row in zip(y_pred, y_true):  # For each sequence
            seq_y_pred_str.append(
                [self.idx2tag[idx] for idx in y_pred_row.tolist()])  # Append list with sequence tokens
            seq_y_true_str.append(
                [self.idx2tag[idx] for idx in y_true_row.tolist()])  # Append list with sequence tokens

        flattened_seq_y_pred_str = list(itertools.chain.from_iterable(seq_y_pred_str))
        flattened_seq_y_true_str = list(itertools.chain.from_iterable(seq_y_true_str))
        assert len(flattened_seq_y_true_str) == len(flattened_seq_y_pred_str)

        # TODO: Check mode (strict, not strict) and scheme
        cr = classification_report(
            y_true=[flattened_seq_y_true_str],
            y_pred=[flattened_seq_y_pred_str],
            zero_division=0,
            mode=None,
            digits=3,
            scheme=IOB2
        )
        LOGGER.info(cr)

    def evaluate_pretrained_model(self):

        train_params = deepcopy(self.train_params)
        train_params.update(self.hyper_params)

        # Build model and load weights manually
        model = self.build_model(train_params=train_params)

        # Fake forward pass to get variables
        LOGGER.info('Model Summary')
        model.print_summary(print_fn=LOGGER.info)

        # Load weights by checkpoint
        model.load_weights(os.path.join(self.eval_params['pretrained_model_path'], 'weights.h5'))

        validation_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='validation')
        validation_generator = DataLoader(
            dataset=validation_dataset,
            vectorize_fn=self.vectorize,
            batch_size=self.general_params['batch_size'],
            max_length=self.train_params['max_length'],
            shuffle=False
        )

        self.evaluate(model=model, generator=validation_generator, split_type='validation')

        test_dataset = datasets.load_dataset(path='nlpaueb/finer-139', split='test')
        test_generator = DataLoader(
            dataset=test_dataset,
            vectorize_fn=self.vectorize,
            batch_size=self.general_params['batch_size'],
            max_length=self.train_params['max_length'],
            shuffle=False
        )

        self.evaluate(model=model, generator=test_generator, split_type='test')

    def loss_report(self, history):
        """
        Prints the loss report of the trained model
        :param history: The history dictionary that tensorflow returns upon completion of fit function
        """

        best_epoch_by_loss = np.argmin(history['val_loss']) + 1
        n_epochs = len(history['val_loss'])
        val_loss_per_epoch = '- ' + ' '.join('-' if history['val_loss'][i] < np.min(history['val_loss'][:i])
                                             else '+' for i in range(1, len(history['val_loss'])))
        report = f'\nBest epoch by Val Loss: {best_epoch_by_loss}/{n_epochs}\n'
        report += f'Val Loss per epoch: {val_loss_per_epoch}\n\n'

        loss_dict = {
            'loss': 'Loss',
            'val_loss': 'Val Loss',
            'val_micro_f1': 'Val Micro F1',
            'val_macro_f1': 'Val Macro F1'
        }
        monitor_metric, monitor_mode = self.get_monitor()
        if monitor_metric != 'val_loss':
            argmin_max_fn = np.argmin if monitor_mode == 'min' else np.argmax
            min_max_fn = np.min if monitor_mode == 'min' else np.max
            best_epoch_by_monitor = argmin_max_fn(history[monitor_metric]) + 1
            val_monitor_per_epoch = '- ' if monitor_mode == 'min' else '+ ' + ' '.join(
                '-' if history[monitor_metric][i] < min_max_fn(history[monitor_metric][:i])
                else '+' for i in range(1, len(history[monitor_metric])))
            monitor_metric_str = " ".join([s.capitalize() for s in monitor_metric.replace('val_', '').split("_")])
            val_monitor_metric_str = " ".join([s.capitalize() for s in monitor_metric.split("_")])
            report += f'Best epoch by {val_monitor_metric_str}: {best_epoch_by_monitor}/{n_epochs}\n'
            report += f'{val_monitor_metric_str} per epoch: {val_monitor_per_epoch}\n\n'
            # loss_dict[monitor_metric.replace('val_', '')] = monitor_metric_str
            # loss_dict[monitor_metric] = val_monitor_metric_str
            report += f"Loss & {monitor_metric_str} Report\n{'-' * 100}\n"
        else:
            report += f"Loss Report\n{'-' * 100}\n"

        report += f"Loss Report\n{'-' * 120}\n"
        report += 'Epoch       | '
        report += ' | '.join([f"{loss_nick:<17}" for loss_name, loss_nick in loss_dict.items() if loss_name in history])
        report += ' | Learning Rate' + '\n'

        for n_epoch in range(len(history['loss'])):
            report += f'Epoch #{n_epoch + 1:3.0f}  | '
            for loss_name in loss_dict.keys():
                if loss_name in history:
                    report += f'{history[loss_name][n_epoch]:1.6f}' + ' ' * 10
                report += '| '
            report += f'{history["lr"][n_epoch]:.3e}' + '\n'

        LOGGER.info(report)
