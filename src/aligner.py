import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Layer, SimpleRNN
import keras.backend as K
import os

@tf.keras.utils.register_keras_serializable(package='Aligner')
class AlignmentLayer(Layer):
    def __init__(self, **kwargs):
        super(AlignmentLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[0][-1], 1),
            initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[0][1], 1),
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
        scores = tf.einsum('ijk,kjk->ij', weighted_q, weighted_p)
        # print(scores.shape)
        # print("cuh")
        scores = K.sum(scores, axis=0)
        # print("scores", scores.shape)
        # Compute the weights
        alpha = K.softmax(scores)
        alpha = K.expand_dims(alpha, axis=-1)
        # print("alpha.shape", alpha.shape)
        context = embed_q * alpha
        # print("context", context.shape)
        context = K.sum(context, axis=0)
        context = K.squeeze(context, axis=-1)
        # print("new context", context.shape)
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
        context = K.sum(context, axis=1)
        return context

class Aligner(object):
    def __init__(self, embed_dim: int, q_encoder_shape: tuple = None) -> None:
        self.embed_dim = embed_dim
        self.create_question_aligner()
        self.create_question_encoder(input_shape=q_encoder_shape)

    def create_question_aligner(self) -> None:
        x_q = Input(shape=(self.embed_dim, 1))
        x_p = Input(shape=(self.embed_dim, 1))
        alignment_layer = AlignmentLayer()([x_q, x_p])
        model = Model([x_q, x_p], alignment_layer, name="Question_Aligner")
        #print("expected", (1, input_shape[0][0]))
        # print("actual", model.output_shape)
        assert model.output_shape == (self.embed_dim,)
        self.q_aligner = model

    def create_question_encoder(self, input_shape: tuple = None, hidden_units: int = 20, activation: str = "tanh") -> None:
        if input_shape is None:
            return;
        
        x = Input(shape=input_shape)
        RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
        attention_layer = Attention()(RNN_layer)
        model = Model(x, attention_layer, name="Question_Encoder")
        # print("input", input_shape)
        # print("expected", (1, input_shape[0]))
        # print("actual", model.output_shape)
        assert model.output_shape == (None, hidden_units)
        self.q_encoder = model
    
    def load_checkpoint(self, checkpoint_dir: str = None) -> tf.train.Checkpoint:
        if checkpoint_dir is None:
            return None
        
        q_aligner_optimizer = keras.optimizers.Adam(0.001)
        q_encoder_optimizer = keras.optimizers.Adam(0.001)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        return tf.train.Checkpoint(q_aligner_optimizer=q_aligner_optimizer,
                                    q_encoder_optimizer=q_encoder_optimizer,
                                    q_aligner=self.q_aligner,
                                    q_encoder=self.q_encoder)
