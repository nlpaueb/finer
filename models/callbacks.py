import logging
import numpy as np
import itertools

from tqdm import tqdm

from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
from tensorflow.keras.callbacks import Callback, EarlyStopping

from configurations import Configuration

LOGGER = logging.getLogger(__name__)


class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


class F1MetricCallback(Callback):
    def __init__(
            self,
            train_params,
            idx2tag,
            train_generator=None,
            validation_generator=None,
            subword_pooling='all',
            calculate_train_metric=False
    ):
        super(F1MetricCallback, self).__init__()

        if validation_generator is None:
            raise Exception(f'F1MetricCallback: Please provide a validation generator')

        if calculate_train_metric and train_generator is None:
            raise Exception(f'F1MetricCallback: Please provide a train generator')

        self.train_params = train_params
        self.idx2tag = idx2tag
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.subword_pooling = subword_pooling
        self.calculate_train_metric = calculate_train_metric

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if self.calculate_train_metric:
            train_micro_precision, train_micro_recall, train_micro_f1, \
            train_macro_precision, train_macro_recall, train_macro_f1, train_support = \
                self.evaluate(generator=self.train_generator)
            logs[f'micro_precision'] = train_micro_precision
            logs[f'micro_recall'] = train_micro_recall
            logs[f'micro_f1'] = train_micro_f1
            logs[f'macro_precision'] = train_macro_precision
            logs[f'macro_recall'] = train_macro_recall
            logs[f'macro_f1'] = train_macro_f1

        val_micro_precision, val_micro_recall, val_micro_f1, \
        val_macro_precision, val_macro_recall, val_macro_f1, val_support = \
            self.evaluate(generator=self.validation_generator)

        logs[f'val_micro_precision'] = val_micro_precision
        logs[f'val_micro_recall'] = val_micro_recall
        logs[f'val_micro_f1'] = val_micro_f1
        logs[f'val_macro_precision'] = val_macro_precision
        logs[f'val_macro_recall'] = val_macro_recall
        logs[f'val_macro_f1'] = val_macro_f1

    def evaluate(self, generator):

        y_true, y_pred = [], []
        for x_batch, y_batch in tqdm(generator, ncols=100):

            if self.subword_pooling in ['first', 'last']:
                pooling_mask = x_batch[1]
                x_batch = x_batch[0]
                y_prob_temp = self.model.predict(x=[x_batch, pooling_mask])
            else:
                pooling_mask = x_batch
                y_prob_temp = self.model.predict(x=x_batch)

            # Get lengths and cut results for padded tokens
            lengths = [len(np.where(x_i != 0)[0]) for x_i in x_batch]

            if self.model.crf:
                y_pred_temp = y_prob_temp.astype('int32')
            else:
                y_pred_temp = np.argmax(y_prob_temp, axis=-1)

            for y_true_i, y_pred_i, l_i, p_i in zip(y_batch, y_pred_temp, lengths, pooling_mask):

                if Configuration['task']['model'] == 'transformer':
                    if self.subword_pooling in ['first', 'last']:
                        y_true.append(np.take(y_true_i, np.where(p_i != 0)[0])[1:-1])
                        y_pred.append(np.take(y_pred_i, np.where(p_i != 0)[0])[1:-1])
                    else:
                        y_true.append(y_true_i[1:l_i - 1])
                        y_pred.append(y_pred_i[1:l_i - 1])

                elif Configuration['task']['model'] == 'bilstm':
                    if self.subword_pooling in ['first', 'last']:
                        y_true.append(np.take(y_true_i, np.where(p_i != 0)[0]))
                        y_pred.append(np.take(y_pred_i, np.where(p_i != 0)[0]))
                    else:
                        y_true.append(y_true_i[:l_i])
                        y_pred.append(y_pred_i[:l_i])

        # Indices to labels list of lists
        seq_y_pred_str = []
        seq_y_true_str = []

        for y_pred_row, y_true_row in zip(y_pred, y_true):
            seq_y_pred_str.append([self.idx2tag[idx] for idx in y_pred_row.tolist()])
            seq_y_true_str.append([self.idx2tag[idx] for idx in y_true_row.tolist()])

        flattened_seq_y_pred_str = list(itertools.chain.from_iterable(seq_y_pred_str))
        flattened_seq_y_true_str = list(itertools.chain.from_iterable(seq_y_true_str))
        assert len(flattened_seq_y_true_str) == len(flattened_seq_y_pred_str)

        precision_micro, recall_micro, f1_micro, support = precision_recall_fscore_support(
            y_true=[flattened_seq_y_true_str],
            y_pred=[flattened_seq_y_pred_str],
            average='micro',
            warn_for=('f-score',),
            beta=1,
            zero_division=0
        )

        precision_macro, recall_macro, f1_macro, support = precision_recall_fscore_support(
            y_true=[flattened_seq_y_true_str],
            y_pred=[flattened_seq_y_pred_str],
            average='macro',
            warn_for=('f-score',),
            beta=1,
            zero_division=0
        )

        return precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, support
