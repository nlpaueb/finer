import tensorflow as tf
import numpy as np
from tf2crf import CRF


class BiLSTM(tf.keras.Model):
    def __init__(
            self,
            n_classes,
            n_layers=1,
            n_units=128,
            dropout_rate=0.1,
            crf=False,
            word2vectors_weights=None,
            subword_pooling='all'
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.crf = crf
        self.subword_pooling = subword_pooling

        self.embeddings = tf.keras.layers.Embedding(
            input_dim=len(word2vectors_weights),
            output_dim=word2vectors_weights.shape[-1],
            weights=[word2vectors_weights],
            trainable=False,
            mask_zero=True
        )

        self.bilstm_layers = []
        for i in range(n_layers):
            self.bilstm_layers.append(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=n_units,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        return_sequences=True,
                        name=f'BiLSTM_{i+1}'
                    )
                )
            )

        if self.crf:
            self.classifier = tf.keras.layers.Dense(
                units=n_classes,
                activation=None
            )
            # Pass logits to a custom CRF Layer
            self.crf_layer = CRF(output_dim=n_classes, mask=True)
        else:
            self.classifier = tf.keras.layers.Dense(
                units=n_classes,
                activation='softmax'
            )

    def call(self, inputs, training=None, mask=None):

        if self.subword_pooling in ['first', 'last']:
            pooling_mask = inputs[1]
            inputs = inputs[0]

        inner_inputs = self.embeddings(inputs)

        for i, bilstm_layer in enumerate(self.bilstm_layers):
            encodings = bilstm_layer(inner_inputs)

            if i != 0:
                inner_inputs = tf.keras.layers.add([inner_inputs, encodings])
            else:
                inner_inputs = encodings

            inner_inputs = tf.keras.layers.SpatialDropout1D(
                rate=self.dropout_rate
            )(inner_inputs, training=training)

        outputs = self.classifier(inner_inputs)

        if self.crf:
            outputs = self.crf_layer(outputs, mask=tf.not_equal(inputs, 0))

        if self.subword_pooling in ['first', 'last']:
            outputs = tf.cast(tf.expand_dims(pooling_mask, axis=-1), dtype=tf.float32) * outputs

        return outputs

    def print_summary(self, line_length=None, positions=None, print_fn=None):
        # Fake forward pass to build graph
        batch_size, sequence_length = 1, 32
        inputs = np.ones((batch_size, sequence_length), dtype=np.int32)

        if self.subword_pooling in ['first', 'last']:
            pooling_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
            inputs = [inputs, pooling_mask]

        self.predict(inputs)
        self.summary(line_length=line_length, positions=positions, print_fn=print_fn)


if __name__ == '__main__':
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Init random seeds
    np.random.seed(1)
    tf.random.set_seed(1)

    # Build test model
    word2vectors_weights = np.random.random((30000, 200))
    model = BiLSTM(
        n_classes=10,
        n_layers=2,
        n_units=128,
        dropout_rate=0.1,
        crf=True,
        word2vectors_weights=word2vectors_weights,
        subword_pooling='all'
    )

    inputs = pad_sequences(np.random.randint(0, 30000, (5, 32)), maxlen=64, padding='post', truncating='post')
    outputs = pad_sequences(np.random.randint(0, 10, (5, 32)), maxlen=64, padding='post', truncating='post')

    if model.crf:
        model.compile(optimizer='adam', loss=model.crf_layer.loss, run_eagerly=True)
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', run_eagerly=True)

    model.print_summary(line_length=150)

    model.fit(x=inputs, y=outputs, batch_size=2)
    predictions = model.predict(inputs, batch_size=2)
    print(predictions)