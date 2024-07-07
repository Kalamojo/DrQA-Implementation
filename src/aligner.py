import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Layer, SimpleRNN
import keras.backend as K
import numpy as np
import os

@tf.keras.utils.register_keras_serializable(package='Aligner')
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
        # print("or worse stuff?")
        # print(weighted_p.shape, weighted_q.shape)
        # print("hmmm, actually did it")
        # Alignment scores
        #print(K.permute_dimensions(weighted_q, (1, 0)).shape)
        #scores = K.dot(weighted_q, K.permute_dimensions(weighted_p, (0, 2, 1)))
        scores = tf.einsum('ijk,ljk->lij', weighted_q, weighted_p)
        # print("cuh scores", scores.shape)
        #print("cuh")
        scores = K.sum(scores, axis=1)
        # print("scores", scores.shape)
        # Compute the weights
        alpha = K.softmax(scores)
        alpha = K.expand_dims(alpha, axis=-1)
        # print("alpha.shape", alpha.shape)
        #context = embed_q * alpha
        context = tf.einsum('ijk,ljk->ljk', embed_q, alpha)
        # print("context", context.shape)
        # context = K.sum(context, axis=0)
        # print("new context", context.shape)
        # context = K.squeeze(context, axis=-1)

        # print("newest", context.shape)
        return context

@tf.keras.utils.register_keras_serializable(package='Attender')
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
        product = product
        #print("new product shape", product.shape)
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
        product = product
        #print("new product shape", product.shape)
        return product

class Aligner(object):
    def __init__(self, embed_dim: int, feature_dim: int, lr: float = 0.001) -> None:
        super(Aligner, self).__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.q_aligner = self.create_question_aligner()
        self.q_encoder = self.create_question_encoder()
        self.start_pred = self.create_start_predictor()
        self.end_pred = self.create_end_predictor()
        self.lr = lr
        #self.optimizer = keras.optimizers.Adam(lr)
        #print("Trainable variables:", self.trainable_variables)

    def calculate_loss(self, pred, answer):
        #print("Trainable variables:", self.trainable_variables)
        if pred[0] > pred[1]:
            return 2
        inter = min(pred[1], answer[1]) - max(pred[0], answer[0])
        union = max(pred[1], answer[1]) - min(pred[0], answer[0])
        return 1 - (max(inter, 0) / (union + min(inter, 0)))

    def create_question_aligner(self) -> Model:
        x_q = Input(shape=(self.embed_dim, 1))
        x_p = Input(shape=(self.embed_dim, 1))
        alignment_layer = AlignmentLayer()([x_q, x_p])
        model = Model([x_q, x_p], alignment_layer, name="Question_Aligner")
        #print("expected", (1, input_shape[0][0]))
        # print("actual", model.output_shape)
        assert model.output_shape == (None, self.embed_dim, 1)
        return model

    def create_question_encoder(self, hidden_units: int = 20, activation: str = "tanh") -> Model:
        x = Input(shape=(self.embed_dim, 1))
        RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
        attention_layer = Attention()(RNN_layer)
        model = Model(x, attention_layer, name="Question_Encoder")
        # print("input", input_shape)
        # print("expected", (1, input_shape[0]))
        # print("actual", model.output_shape)
        assert model.output_shape == (1, self.embed_dim, 1)
        return model
    
    def create_start_predictor(self) -> Model:
        x_p = Input(shape=(self.embed_dim * 2 + self.feature_dim, 1))
        x_q = Input(shape=(self.embed_dim, 1))
        start_predictor = StartPredictor()([x_p, x_q])
        model = Model([x_p, x_q], start_predictor, name="Start_Predictor")
        assert model.output_shape == (None, 1)
        return model
    
    def create_end_predictor(self) -> Model:
        x_p = Input(shape=(self.embed_dim * 2 + self.feature_dim, 1))
        x_q = Input(shape=(self.embed_dim, 1))
        start_predictor = EndPredictor()([x_p, x_q])
        model = Model([x_p, x_q], start_predictor, name="End_Predictor")
        assert model.output_shape == (None, 1)
        return model
    
    def load_checkpoint(self, checkpoint_dir: str = None) -> tf.train.Checkpoint:
        if checkpoint_dir is None:
            return None
        
        #optimizer = keras.optimizers.Adam(self.lr)
        q_aligner_optimizer = keras.optimizers.Adam(self.lr)
        q_encoder_optimizer = keras.optimizers.Adam(self.lr)
        start_pred_optimizer = keras.optimizers.Adam(self.lr)
        end_pred_optimizer = keras.optimizers.Adam(self.lr)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        return tf.train.Checkpoint(q_aligner_optimizer=q_aligner_optimizer,
                                    q_encoder_optimizer=q_encoder_optimizer,
                                    start_pred_optimizer=start_pred_optimizer,
                                    end_pred_optimizer=end_pred_optimizer,
                                    q_aligner=self.q_aligner,
                                    q_encoder=self.q_encoder,
                                    start_pred=self.start_pred,
                                    end_pred=self.end_pred)
        # return tf.train.Checkpoint(optimizer=optimizer,
        #                             q_aligner=self.q_aligner,
        #                             q_encoder=self.q_encoder,
        #                             start_pred=self.start_pred,
        #                             end_pred=self.end_pred)
