import tensorflow as tf
import keras
from keras import Input, Model, layers, Layer
from keras import ops as K
import numpy as np
import os

@tf.keras.utils.register_keras_serializable(package='Q-Aligner')
class AlignmentLayer(Layer):
    def __init__(self, **kwargs):
        super(AlignmentLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="alignment_weight", shape=(input_shape[1][-1], 1),
            initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="alignment_bias", shape=(input_shape[1][1],),
            initializer="zeros", trainable=True)
        super(AlignmentLayer, self).build(input_shape)

    def call(self, x):
        embed_q, embed_p = x
        # Multiplying by weights
        weighted_p = K.dot(embed_p, self.W)
        weighted_q = K.dot(embed_q, self.W)
        # Alignment scores
        scores = tf.einsum('ijk,ilk->il', weighted_q, weighted_p)+self.b
        # Compute the weights
        alpha = K.softmax(scores)
        context = tf.einsum('ijk,il->ilk', embed_q, alpha)
        return context

@tf.keras.utils.register_keras_serializable(package='Q-Attender')
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
            initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1),
            initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        scores = K.tanh(K.dot(x, self.W)+self.b)
        # Remove dimension of size 1
        scores = K.squeeze(scores, axis=-1)
        # Compute the weights
        alpha = K.softmax(scores)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

@tf.keras.utils.register_keras_serializable(package='S-Predictor')
class StartPredictor(Layer):
    def __init__(self, **kwargs):
        super(StartPredictor, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="start_prob_weight", shape=(input_shape[0][-1], 1),
            initializer="random_normal", trainable=True)
        super(StartPredictor, self).build(input_shape)
    
    def call(self, x):
        p_matrix, q_vector = x
        product = K.dot(p_matrix, self.W)
        product = tf.einsum('ijk,il->ijl', product, q_vector)
        return product

@tf.keras.utils.register_keras_serializable(package='E-Predictor')
class EndPredictor(Layer):
    def __init__(self, **kwargs):
        super(EndPredictor, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="end_prob_weight", shape=(input_shape[0][-1], 1),
            initializer="random_normal", trainable=True)
        super(EndPredictor, self).build(input_shape)
    
    def call(self, x):
        p_matrix, q_vector = x
        product = K.dot(p_matrix, self.W)
        product = tf.einsum('ijk,il->ijl', product, q_vector)
        return product

@tf.keras.utils.register_keras_serializable(package='Ind-Formatter')
class IndFormatter(Layer):
    def __init__(self, **kwargs):
        super(IndFormatter, self).__init__(**kwargs)
    
    def call(self, x):
        result = tf.exp(tf.squeeze(x, axis=-1))
        return result

class Aligner(object):
    def __init__(self, embed_dim: int, max_q: int = -1, max_p: int = -1, lr: float = 0.001) -> None:
        super(Aligner, self).__init__()
        self.embed_dim = embed_dim
        self.feature_dim = 6
        self.max_q = max_q
        self.max_p = max_p
        self.q_aligner = self.create_question_aligner(self.max_q, self.max_p)
        self.q_encoder = self.create_question_encoder(self.max_q)
        self.start_pred = self.create_start_predictor(self.max_p)
        self.end_pred = self.create_end_predictor(self.max_p)
        self.qa_optimizer = keras.optimizers.Adam(lr)
        self.qe_optimizer = keras.optimizers.Adam(lr)
        self.sp_optimizer = keras.optimizers.Adam(lr)
        self.ep_optimizer = keras.optimizers.Adam(lr)
        self.span_limit = 30
    
    def predict(self, query_embedding: np.ndarray, paragraph_embeddings: np.ndarray, feature_matrix: np.ndarray) -> np.ndarray:
        query_vector = self.q_encoder.predict(query_embedding)
        aligned_paragraph_matrix = self.q_aligner.predict([query_embedding, paragraph_embeddings])
        paragraph_vectors = np.concatenate([aligned_paragraph_matrix, paragraph_embeddings, feature_matrix], axis=-1)

        start_matrix = self.start_pred.predict([paragraph_vectors, query_vector])
        end_matrix = self.end_pred.predict([paragraph_vectors, query_vector])

        start_prod = self._prefix_product(start_matrix, end_matrix, True)
        end_prod = self._prefix_product(end_matrix, start_matrix, False)

        start_argmax = np.argmax(start_prod, axis=-1)
        end_argmax = np.argmax(end_prod, axis=-1)

        max_start = start_prod[np.arange(start_argmax.shape[0]), start_argmax]
        max_end = end_prod[np.arange(end_argmax.shape[0]), end_argmax]

        start_ind_x = np.argmax(max_start)
        start_ind = np.array([start_ind_x, start_argmax[start_ind_x]])
        end_ind_x = np.argmax(max_end)
        end_ind = np.array([end_ind_x, end_argmax[end_ind_x]])

        return start_ind, end_ind, start_argmax, end_argmax

    @tf.function
    def train_step(self, query_embedding: tf.Tensor, paragraph_embeddings: tf.Tensor, feature_matrix: tf.Tensor, start_spans: tf.Tensor, end_spans: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as qe_tape, tf.GradientTape() as qa_tape, tf.GradientTape() as sp_tape, tf.GradientTape() as ep_tape:
            query_vector = self.q_encoder(query_embedding, training=True)
            aligned_paragraph_matrix = self.q_aligner([query_embedding, paragraph_embeddings], training=True)
            paragraph_vectors = tf.concat([aligned_paragraph_matrix, paragraph_embeddings, feature_matrix], axis=-1)
            print("paragraph vectors:", paragraph_vectors)

            start_matrix = self.start_pred([paragraph_vectors, query_vector], training=True)
            end_matrix = self.end_pred([paragraph_vectors, query_vector], training=True)

            losses = tf.nn.softmax_cross_entropy_with_logits(tf.stack([start_spans, end_spans]), tf.stack([start_matrix, end_matrix]))
            losses = tf.reduce_mean(losses, axis=1)
            print("losses:", losses)
            loss_avg = tf.reduce_mean(losses, axis=0)
            loss_1, loss_2 = tf.split(losses, 2, axis=0)
        
        qa_gradients = qa_tape.gradient(loss_avg, self.q_aligner.trainable_variables)
        qe_gradients = qe_tape.gradient(loss_avg, self.q_encoder.trainable_variables)
        sp_gradients = sp_tape.gradient(loss_1, self.start_pred.trainable_variables)
        ep_gradients = ep_tape.gradient(loss_2, self.end_pred.trainable_variables)
        
        self.qa_optimizer.apply_gradients(zip(qa_gradients, self.q_aligner.trainable_variables))
        self.qe_optimizer.apply_gradients(zip(qe_gradients, self.q_encoder.trainable_variables))
        self.sp_optimizer.apply_gradients(zip(sp_gradients, self.start_pred.trainable_variables))
        self.ep_optimizer.apply_gradients(zip(ep_gradients, self.end_pred.trainable_variables))

        return loss_avg, losses

    def _prefix_product(self, matrix_1: tf.Tensor, matrix_2: tf.Tensor, reverse) -> tf.Tensor:
        matrix_2_cmax = self._cumulative_max(matrix_2, reverse=reverse)
        return np.multiply(matrix_1, matrix_2_cmax)
    
    def _tf_while_condition(self, x, loop_counter, reverse):
        return tf.not_equal(loop_counter, self.span_limit - 1)

    def _tf_while_body(self, x, loop_counter, reverse):
        loop_counter += 1
        y = tf.cond(reverse,
                   lambda: tf.concat((x[:, 1:], tf.expand_dims(x[:, -1], axis=1)), axis=1),
                   lambda: tf.concat((tf.expand_dims(x[:, 0], axis=1), x[:, :-1]), axis=1))
        new_x = tf.maximum(x, y)
        return new_x, loop_counter, reverse

    def _cumulative_max(self, matrix: np.ndarray, reverse: bool) -> np.ndarray:
        new_x = np.copy(matrix)
        for i in range(self.span_limit):
            if reverse:
                y = np.concatenate((new_x[:, 1:], np.expand_dims(new_x[:, -1], axis=1)), axis=1)
            else:
                y = np.concatenate((np.expand_dims(new_x[:, 0], axis=1), new_x[:, :-1]), axis=1)
            new_x = np.maximum(new_x, y)
        return new_x
        # cumulative_max, _, _ = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond=self._tf_while_condition, 
        #                           body=self._tf_while_body, 
        #                           loop_vars=(matrix, 0, reverse)))
        # return cumulative_max
    
    def create_question_aligner(self, max_q: int, max_p: int) -> Model:
        if max_q == -1 or max_p == -1:
            return None
        x_q = Input(shape=(max_q, self.embed_dim))
        x_p = Input(shape=(max_p, self.embed_dim))
        alignment_layer = AlignmentLayer()([x_q, x_p])
        model = Model([x_q, x_p], alignment_layer, name="Question_Aligner")
        assert model.output_shape == (None, max_p, self.embed_dim)
        return model

    def create_question_encoder(self, max_q: int, hidden_units: int = 20, activation: str = "tanh") -> Model:
        if max_q == -1:
            return None
        x = Input(shape=(max_q, self.embed_dim))
        LSTM_layer = layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, activation=activation, dropout=0.2))(x)
        attention_layer = Attention()(LSTM_layer)
        model = Model(x, attention_layer, name="Question_Encoder")
        assert model.output_shape == (None, hidden_units * 2)
        self.q_units = hidden_units * 2
        return model
    
    def create_start_predictor(self, max_p: int, hidden_units: int = 20, activation: str = "tanh") -> Model:
        if not hasattr(self, 'q_units') or max_p == -1:
            return None
        x_p = Input(shape=(max_p, self.embed_dim * 2 + self.feature_dim))
        x_q = Input(shape=(self.q_units,))
        start_predictor = StartPredictor()([x_p, x_q])
        LSTM_layer = layers.Bidirectional(layers.LSTM(hidden_units, activation=activation, return_sequences=True, dropout=0.2))(start_predictor)
        Dense_layer = layers.Dense(1, activation=activation)(LSTM_layer)
        ind_format = IndFormatter()(Dense_layer)
        model = Model([x_p, x_q], ind_format, name="Start_Predictor")
        assert model.output_shape == (None, max_p)
        return model
    
    def create_end_predictor(self, max_p: int, hidden_units: int = 20, activation: str = "tanh") -> Model:
        if not hasattr(self, 'q_units') or max_p == -1:
            return None
        x_p = Input(shape=(max_p, self.embed_dim * 2 + self.feature_dim))
        x_q = Input(shape=(self.q_units,))
        end_predictor = EndPredictor()([x_p, x_q])
        LSTM_layer = layers.Bidirectional(layers.LSTM(hidden_units, activation=activation, return_sequences=True, dropout=0.2))(end_predictor)
        Dense_layer = layers.Dense(1, activation=activation)(LSTM_layer)
        ind_format = IndFormatter()(Dense_layer)
        model = Model([x_p, x_q], ind_format, name="End_Predictor")
        assert model.output_shape == (None, max_p)
        return model
    
    def restore_checkpoint(self, checkpoint_dir: str) -> None:
        print(tf.train.latest_checkpoint(checkpoint_dir))
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save_checkpoint(self) -> None:
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(q_aligner=self.q_aligner,
                                    q_encoder=self.q_encoder,
                                    start_pred=self.start_pred,
                                    end_pred=self.end_pred)
