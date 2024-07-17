import tensorflow as tf
import keras
from keras import Input, Model, layers, Layer
#from keras.layers import Layer, SimpleRNN
from keras import ops as K
#import keras.backend as K
import numpy as np
import os

@tf.keras.utils.register_keras_serializable(package='Q-Aligner')
class AlignmentLayer(Layer):
    def __init__(self, **kwargs):
        super(AlignmentLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="alignment_weight", shape=(input_shape[0][-1], 1),
            initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="alignment_bias", shape=(input_shape[0][1], 1),
            initializer="zeros", trainable=True)
        super(AlignmentLayer, self).build(input_shape)

    def call(self, x):
        embed_q, embed_p = x
        #embed_p = K.reshape(embed_p, (embed_p.shape[0], 1))
        # print("align input", embed_q.shape, embed_p.shape)
        # print("weights and bias", self.W.shape, self.b.shape)
        # Multiplying by weights
        weighted_p = K.dot(embed_p, self.W)+self.b
        # print(weighted_p.shape)
        # print("bad stuff?")
        weighted_q = K.dot(embed_q, self.W)
        # print(weighted_q.shape)
        weighted_q = weighted_q + self.b
        # print("weighted shapes")
        # print(weighted_p.shape, weighted_q.shape)
        # print("hmmm, actually did it")
        tf.matmul
        # Alignment scores
        #print(K.permute_dimensions(weighted_q, (1, 0)).shape)
        #scores = K.dot(weighted_q, K.permute_dimensions(weighted_p, (0, 2, 1)))
        scores = tf.einsum('ijk,ljk->lj', weighted_q, weighted_p)
        #print("cuh scores", scores.shape)
        #print("cuh")
        #scores = K.sum(scores, axis=1)
        #print("scores", scores.shape)
        # Compute the weights
        alpha = K.softmax(scores)
        alpha = K.expand_dims(alpha, axis=-1)
        # print("alpha.shape", alpha.shape)
        # print("embed shape", embed_q.shape)
        #context = embed_q * alpha
        context = tf.einsum('ijk,ljk->ljk', embed_q, alpha)
        # print("context", context.shape)
        # context = K.sum(context, axis=0)
        # print("new context", context.shape)
        # context = K.squeeze(context, axis=-1)

        # print("newest", context.shape)
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
        #print("input", x.shape)
        # print("weights and bias", self.W.shape, self.b.shape)
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
        #print("context shape", context.shape)
        context = K.sum(context, axis=1)
        # Extra stuff
        #print("new context shape", context.shape)
        new_context = K.expand_dims(K.sum(context, axis=0), axis=-1)
        #print("even newer context shape", new_context.shape)
        summation = K.dot(x, new_context)
        #print("old summation shape", summation.shape)
        summation = K.sum(summation, axis=0, keepdims=True)
        #print("new summation shape", summation.shape)
        return summation

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
        #print("input shapes:", p_matrix.shape, q_vector.shape)
        product = K.dot(p_matrix, self.W)
        #print("product shape", product.shape)
        #product = K.dot(product, K.permute_dimensions(q_vector, (2, 0, 1)))
        product = tf.einsum('ijk,lmk->ik', product, q_vector)
        product = tf.expand_dims(product, axis=-1)
        #product = K.softplus(product)
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
        #print("input shapes:", p_matrix.shape, q_vector.shape)
        product = K.dot(p_matrix, self.W)
        #print("product shape", product.shape)
        #product = K.dot(product, K.permute_dimensions(q_vector, (2, 0, 1)))
        product = tf.einsum('ijk,lmk->ik', product, q_vector)
        product = tf.expand_dims(product, axis=-1)
        #product = K.softplus(product)
        return product

@tf.keras.utils.register_keras_serializable(package='Ind-Formatter')
class IndFormatter(Layer):
    def __init__(self, **kwargs):
        super(IndFormatter, self).__init__(**kwargs)
    
    def call(self, x):
        result = tf.exp(tf.squeeze(x, axis=-1))
        return result

class Aligner(object):
    def __init__(self, embed_dim: int, feature_dim: int, lr: float = 0.0002) -> None:
        super(Aligner, self).__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.q_aligner = self.create_question_aligner()
        self.q_encoder = self.create_question_encoder()
        self.start_pred = self.create_start_predictor()
        self.end_pred = self.create_end_predictor()
        self.qa_optimizer = keras.optimizers.Adam(lr)
        self.qe_optimizer = keras.optimizers.Adam(lr)
        self.sp_optimizer = keras.optimizers.Adam(lr)
        self.ep_optimizer = keras.optimizers.Adam(lr)
        self.span_limit = 30
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 300, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 300, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 6, 1], dtype=tf.float32)])
    def predict(self, query_embedding: tf.Tensor, paragraph_embeddings: tf.Tensor, feature_matrix: tf.Tensor) -> tf.Tensor:
        #with tf.GradientTape() as qa_tape, tf.GradientTape() as qe_tape, tf.GradientTape() as sp_tape, tf.GradientTape() as ep_tape:
        query_vector = self.q_encoder(query_embedding, training=False)
        aligned_paragraph_matrix = self.q_aligner([query_embedding, paragraph_embeddings], training=False)
        paragraph_vectors = tf.concat([aligned_paragraph_matrix, paragraph_embeddings, feature_matrix], axis=1)

        start_matrix = self.start_pred([paragraph_vectors, query_vector], training=False)
        end_matrix = self.end_pred([paragraph_vectors, query_vector], training=False)
        #print(start_matrix)
        # start_ind = self._softargmax(start_matrix)
        # end_ind = self._softargmax(end_matrix, offset=tf.cast(start_ind, tf.int32))
        start_prod = self._prefix_product(start_matrix, end_matrix, tf.cast(True, dtype=tf.bool))
        end_prod = self._prefix_product(end_matrix, start_matrix, tf.cast(False, dtype=tf.bool))

        start_ind = tf.argmax(start_prod)
        end_ind = tf.argmax(end_prod)+1
        print(start_ind)
        print(end_ind)

        return start_ind, end_ind, start_prod, end_prod

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 300, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 300, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 6, 1], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def train_step(self, query_embedding: tf.Tensor, paragraph_embeddings: tf.Tensor, feature_matrix: tf.Tensor, start_spans: tf.Tensor, end_spans: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as qe_tape, tf.GradientTape() as qa_tape, tf.GradientTape() as sp_tape, tf.GradientTape() as ep_tape:
        #with tf.GradientTape() as tape:
            query_vector = self.q_encoder(query_embedding, training=True)
            aligned_paragraph_matrix = self.q_aligner([query_embedding, paragraph_embeddings], training=True)
            paragraph_vectors = tf.concat([aligned_paragraph_matrix, paragraph_embeddings, feature_matrix], axis=1)

            start_matrix = self.start_pred([paragraph_vectors, query_vector], training=True)
            end_matrix = self.end_pred([paragraph_vectors, query_vector], training=True)
            #print(start_matrix)
            # start_ind = self._softargmax(start_matrix)
            # end_ind = self._softargmax(end_matrix, offset=tf.cast(start_ind, tf.int32))+1
            #print("end matrix:", end_matrix)
            # start_prod = self._prefix_product(start_matrix, end_matrix, tf.constant(True, dtype=tf.bool))
            # end_prod = self._prefix_product(end_matrix, start_matrix, tf.constant(False, dtype=tf.bool))
            losses = tf.nn.softmax_cross_entropy_with_logits(tf.stack([start_spans, end_spans]), tf.stack([start_matrix, end_matrix]))
            #print("losses:", losses)
            loss_avg = tf.reduce_mean(losses, axis=-1)
            loss_1, loss_2 = tf.split(losses, 2)
            # start_ind = self._softargmax(start_prod)
            # end_ind = self._softargmax(end_prod)+1

            # loss = tf.reduce_min(tf.map_fn(lambda span: self._calculate_loss(start_ind, end_ind, span[0], span[1], tf.cast(tf.shape(start_matrix)[-1], dtype=tf.float32)), elems=answer_spans))
        #print("loss:", loss)
        
        #variables = self.q_aligner.trainable_variables + self.q_encoder.trainable_variables + self.start_pred.trainable_variables + self.end_pred.trainable_variables
        qa_gradients = qa_tape.gradient(loss_avg, self.q_aligner.trainable_variables)
        qe_gradients = qe_tape.gradient(loss_avg, self.q_encoder.trainable_variables)
        sp_gradients = sp_tape.gradient(loss_1, self.start_pred.trainable_variables)
        ep_gradients = ep_tape.gradient(loss_2, self.end_pred.trainable_variables)
        #print("grads:", qa_gradients, qe_gradients, sp_gradients, ep_gradients)
        
        self.qa_optimizer.apply_gradients(zip(qa_gradients, self.q_aligner.trainable_variables))
        self.qe_optimizer.apply_gradients(zip(qe_gradients, self.q_encoder.trainable_variables))
        self.sp_optimizer.apply_gradients(zip(sp_gradients, self.start_pred.trainable_variables))
        self.ep_optimizer.apply_gradients(zip(ep_gradients, self.end_pred.trainable_variables))

        #print("applied gradients")
        return loss_avg, losses



    def _prefix_product(self, matrix_1: tf.Tensor, matrix_2: tf.Tensor, reverse) -> tf.Tensor:
        matrix_2_cmax = self._cumulative_max(matrix_2, reverse=reverse)
        return tf.multiply(matrix_1, matrix_2_cmax)
    
    def tf_while_condition(self, x, loop_counter, reverse):
        return tf.not_equal(loop_counter, self.span_limit - 1)

    def tf_while_body(self, x, loop_counter, reverse):
        loop_counter += 1
        y = tf.cond(reverse,
                   lambda: tf.concat((x[1:], [x[-1]]), axis=0),
                   lambda: tf.concat(([x[0]], x[:-1]), axis=0))
        #print(y)
        new_x = tf.maximum(x, y)
        return new_x, loop_counter, reverse

    def _cumulative_max(self, matrix: tf.Tensor, reverse: tf.Tensor) -> tf.Tensor:
        #return tf.scan(lambda a, b: tf.maximum(a, b), matrix, reverse=reverse, initializer=tf.reduce_min(matrix))
        cumulative_max, _, _ = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond=self.tf_while_condition, 
                                  body=self.tf_while_body, 
                                  loop_vars=(matrix, 0, reverse)))
        #print("cumulative max:", cumulative_max)
        return cumulative_max

    def gumbel_softmax(self, logits, temperature, eps=1e-20):
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits)) + eps) + eps)
        y = logits + gumbel_noise
        return tf.nn.softmax(y / temperature)

    def _softargmax(self, x, temp=0.01) -> tf.Tensor:
        x_range = tf.range(tf.shape(x)[-1], dtype=x.dtype)
        probs = self.gumbel_softmax(x, temp)
        return tf.reduce_sum(probs * x_range, axis=-1)
    
    def _calculate_loss(self, pred_start: tf.Tensor, pred_end: tf.Tensor, answer_start: int, answer_end: int, size: tf.Tensor, smooth: float = 0.01) -> tf.Tensor:
        inter = tf.reduce_min(tf.stack([pred_end, answer_end])) - tf.reduce_max(tf.stack([pred_start, answer_start]))
        union = tf.reduce_max(tf.stack([pred_end, answer_end])) - tf.reduce_min(tf.stack([pred_start, answer_start]))
        return 1 - ((tf.reduce_max(tf.stack([inter, 0])) + smooth) / (union + tf.reduce_min(tf.stack([inter, 0])) + smooth)) + tf.abs(tf.reduce_min(tf.stack([inter, 0])) / size)
        # return tf.cond(tf.greater(pred_start, pred_end)
        #         , lambda: 2.0 * smooth
        #         , lambda: (1 - ((tf.reduce_max(tf.stack([inter, 0])) + smooth) / (union + tf.reduce_min(tf.stack([inter, 0])) + smooth))) * smooth)


    def create_question_aligner(self) -> Model:
        x_q = Input(shape=(self.embed_dim, 1))
        x_p = Input(shape=(self.embed_dim, 1))
        alignment_layer = AlignmentLayer()([x_q, x_p])
        model = Model([x_q, x_p], alignment_layer, name="Question_Aligner")
        # print("expected", (None, self.embed_dim, 1))
        # print("actual", model.output_shape)
        assert model.output_shape == (None, self.embed_dim, 1)
        return model

    def create_question_encoder(self, hidden_units: int = 20, activation: str = "tanh") -> Model:
        x = Input(shape=(self.embed_dim, 1))
        LSTM_layer = layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, activation=activation))(x)
        attention_layer = Attention()(LSTM_layer)
        model = Model(x, attention_layer, name="Question_Encoder")
        # print("input", input_shape)
        # print("expected", (1, input_shape[0]))
        # print("actual", model.output_shape)
        assert model.output_shape == (1, self.embed_dim, 1)
        return model
    
    def create_start_predictor(self, hidden_units: int = 20, activation: str = "tanh") -> Model:
        x_p = Input(shape=(self.embed_dim * 2 + self.feature_dim, 1))
        x_q = Input(shape=(self.embed_dim, 1))
        start_predictor = StartPredictor()([x_p, x_q])
        LSTM_layer = layers.Bidirectional(layers.LSTM(hidden_units, activation=activation))(start_predictor)
        Dense_layer = layers.Dense(1, activation=activation)(LSTM_layer)
        ind_format = IndFormatter()(Dense_layer)
        model = Model([x_p, x_q], ind_format, name="Start_Predictor")
        #print("output shape:", model.output_shape)
        assert model.output_shape == (None,)
        return model
    
    def create_end_predictor(self, hidden_units: int = 20, activation: str = "tanh") -> Model:
        x_p = Input(shape=(self.embed_dim * 2 + self.feature_dim, 1))
        x_q = Input(shape=(self.embed_dim, 1))
        end_predictor = EndPredictor()([x_p, x_q])
        LSTM_layer = layers.Bidirectional(layers.LSTM(hidden_units, activation=activation))(end_predictor)
        Dense_layer = layers.Dense(1, activation=activation)(LSTM_layer)
        ind_format = IndFormatter()(Dense_layer)
        model = Model([x_p, x_q], ind_format, name="End_Predictor")
        assert model.output_shape == (None,)
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
