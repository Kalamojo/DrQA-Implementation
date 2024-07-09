import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Layer, SimpleRNN
import keras.backend as K
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
        #print("prod:", product.shape)
        product = K.squeeze(K.argmax(product, axis=0), axis=-1)
        #print("prod arg:", product.shape)
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
        product = K.squeeze(K.argmax(product, axis=0), axis=-1) + 1
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

    @tf.function
    def train_step(self, query_embedding: np.ndarray, paragraph_embeddings: np.ndarray, feature_matrix: np.ndarray, answer_spans: list[tuple[int, int]], optimizer, train: bool = True):
        # q_aligner_optimizer = keras.optimizers.Adam(self.lr)
        # q_encoder_optimizer = keras.optimizers.Adam(self.lr)
        # start_pred_optimizer = keras.optimizers.Adam(self.lr)
        # end_pred_optimizer = keras.optimizers.Adam(self.lr)
        #optimizer = keras.optimizers.Adam(self.lr)

        # with tf.GradientTape(persistent=True) as qe_tape:
        #     qe_tape.watch(query_embedding)
        #     query_vector = self.q_encoder(query_embedding, training=train)
        # with tf.GradientTape(persistent=True) as qa_tape:
        #     qa_tape.watch([query_embedding, paragraph_embeddings])
        #     aligned_paragraph_matrix = self.q_aligner([query_embedding, paragraph_embeddings], training=train)
        
        # paragraph_vectors = tf.concat([aligned_paragraph_matrix, paragraph_embeddings, feature_matrix], axis=1)
        # print("paragraph vec", paragraph_vectors.shape)
        # with tf.GradientTape(persistent=True) as sp_tape, tf.GradientTape(persistent=True) as ep_tape:
        #     print("start pred")
        #     sp_tape.watch([paragraph_vectors, query_vector])
        #     start_matrix = self.start_pred([paragraph_vectors, query_vector], training=train)
        #     print("end pred")
        #     ep_tape.watch([paragraph_vectors, query_vector])
        #     end_matrix = self.end_pred([paragraph_vectors, query_vector], training=train)

        #     prediction = tf.range(tf.argmax(start_matrix)[0], tf.argmax(end_matrix)[0]+1, dtype=tf.int64)
        #     print("prediction_span:", prediction)
        #     loss = self.calculate_loss(prediction, tf.range(answer_spans[0][0], answer_spans[0][1]+1, dtype=tf.int64))


        q_embed = tf.Variable(tf.zeros_like(query_embedding))
        p_embed = tf.Variable(tf.zeros_like(paragraph_embeddings))
        f_matrix = tf.Variable(tf.zeros_like(feature_matrix))
        answer_start = tf.Variable(tf.zeros_like(answer_spans[0][0], dtype=tf.int64))
        answer_end = tf.Variable(tf.zeros_like(answer_spans[0][1], dtype=tf.int64))
        #with tf.GradientTape() as qa_tape, tf.GradientTape() as qe_tape, tf.GradientTape() as sp_tape, tf.GradientTape() as ep_tape:
        with tf.GradientTape() as tape:
            tape.watch(q_embed)
            tape.watch(p_embed)
            tape.watch(f_matrix)
            tape.watch(answer_start)
            tape.watch(answer_end)
            q_embed.assign(query_embedding)
            print(query_embedding.shape, query_embedding.dtype)
            print(q_embed)
            #qe_tape.watch(query_embedding)
            
            query_vector = self.q_encoder(q_embed, training=train)
            
            p_embed.assign(paragraph_embeddings)
            #qa_tape.watch([query_embedding, paragraph_embeddings])
            aligned_paragraph_matrix = self.q_aligner([p_embed, paragraph_embeddings], training=train)
            
            f_matrix.assign(feature_matrix)
            paragraph_vectors = tf.concat([aligned_paragraph_matrix, p_embed, f_matrix], axis=1)
            print("paragraph vec", paragraph_vectors.shape)

            print("start pred")
            #sp_tape.watch([paragraph_vectors, query_vector])
            start_matrix = self.start_pred([paragraph_vectors, query_vector], training=train)
            print(start_matrix)
            #print(tf.argmax(start_matrix))
            # print(start_matrix)
            # print(start_matrix.shape)
            # print("start index:", np.argmax(start_matrix))
            print("end pred")
            #ep_tape.watch([paragraph_vectors, query_vector])
            end_matrix = self.end_pred([paragraph_vectors, query_vector], training=train)
            print(end_matrix)
            # print(end_matrix)
            # print(end_matrix.shape)
            # print("end index:", np.argmax(end_matrix))
            #prediction = tf.stack([tf.argmax(start_matrix)[0], tf.argmax(end_matrix)[0]])
            
            # prediction = tf.Variable(tf.zeros([], dtype=tf.int64))

            # start_val = tf.Variable(tf.argmax(start_matrix)[0])
            # end_val = tf.Variable(tf.argmax(end_matrix)[0]+1)
            # if start_val > end_val:
            #     #prediction.assign(tf.range(start_val, end_val, dtype=tf.int64))
            #     start_val.assign(0)
            #     end_val.assign(0)

            #prediction = tf.Variable(tf.cast(tf.range(tf.argmax(start_matrix)[0], tf.argmax(end_matrix)[0]+1), dtype=tf.float32))
            #print("prediction_span:", prediction)
            # with tf.compat.v1.Session() as sess:
            #     print(sess.run(prediction))

            #answers = tf.Variable(tf.cast(tf.range(answer_spans[0][0], answer_spans[0][1]+1), dtype=tf.float32))
            answer_start.assign(answer_spans[0][0])
            answer_end.assign(answer_spans[0][1])
            #tape.watch(prediction)
            #tape.watch(answers)
            print("stuff")
            print(answer_start)
            print(start_matrix)
            #loss = self.calculate_loss(start_matrix, end_matrix, answer_start, answer_end)
            loss = answer_start * 4
            # for i in range(1, len(answer_spans)):
            #     print("ans span:", answer_spans[i])
            #     #loss = tf.reduce_min(tf.stack([loss, self.calculate_loss(prediction, tf.constant(answer_spans[i], dtype=tf.int64))]))
            #     loss = tf.reduce_min(tf.stack([loss, self.calculate_loss(prediction, tf.range(answer_spans[i][0], answer_spans[i][1]+1, dtype=tf.int64))]))
            #     #loss = min(loss, self.calculate_loss(prediction, tf.constant(answer, dtype=tf.int64)))
            #loss = tf.reshape(loss, [1, 1])
        print("loss:", loss)
        
        # variables = self.q_aligner.trainable_variables + self.q_encoder.trainable_variables + self.start_pred.trainable_variables + self.end_pred.trainable_variables
        # gradients = tape.gradient(loss, variables)
        # gradients_of_q_aligner = qa_tape.gradient(loss, self.q_aligner.trainable_variables)
        # gradients_of_q_encoder = qe_tape.gradient(loss, self.q_encoder.trainable_variables)
        # gradients_of_start_pred = sp_tape.gradient(loss, self.start_pred.trainable_variables)
        # gradients_of_end_pred = ep_tape.gradient(loss, self.end_pred.trainable_variables)

        print("grad stuff")
        print(tape.gradient(loss, answer_start))
        # print(gradients)
        # print(variables)
        # print(gradients_of_q_aligner)
        # print(self.q_aligner.trainable_variables)
        # print(gradients_of_q_encoder, gradients_of_start_pred, gradients_of_end_pred)

        # optimizer.apply_gradients(zip(gradients, variables))
        # q_aligner_optimizer.apply_gradients(zip(gradients_of_q_aligner, self.q_aligner.trainable_variables))
        # q_encoder_optimizer.apply_gradients(zip(gradients_of_q_encoder, self.q_encoder.trainable_variables))
        # start_pred_optimizer.apply_gradients(zip(gradients_of_start_pred, self.start_pred.trainable_variables))
        # end_pred_optimizer.apply_gradients(zip(gradients_of_end_pred, self.end_pred.trainable_variables))
        return loss

    def calculate_loss(self, pred_start: tf.Tensor, pred_end: tf.Tensor, answer_start: tf.Tensor, answer_end: tf.Tensor, smooth: float = 1.) -> tf.Tensor:
        pred = tf.cast(tf.range(pred_start, pred_end), tf.float32)
        answer = tf.cast(tf.range(answer_start, answer_end), tf.float32)
        intersection = K.sum(K.abs(answer * pred), axis=-1)
        sum_ = K.sum(K.abs(answer) + K.abs(pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

        # smooth = 1.
        # y_true_f = K.cast(K.flatten(answer), tf.float32)
        # y_pred_f = K.cast(K.flatten(pred), tf.float32)
        # intersection = K.sum(y_true_f * y_pred_f)
        # score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        # return 1 - score
    

        # tp = tf.reduce_sum(tf.matmul(answer, pred), 1)
        # fn = tf.reduce_sum(tf.matmul(answer, 1-pred), 1)
        # fp = tf.reduce_sum(tf.matmul(1-answer, pred), 1)
        # return 1 - (tp / (tp + fn + fp))
    
        # # print(pred, answer)
        # if pred_start > pred_end:
        #     return tf.constant(2, dtype=tf.float64)
        # #inter = min(pred[1], answer[1]) - max(pred[0], answer[0])
        # inter = tf.reduce_min(tf.stack([pred_end, answer_end])) - tf.reduce_max(tf.stack([pred_start, answer_start]))
        # #union = max(pred[1], answer[1]) - min(pred[0], answer[0])
        # union = tf.reduce_max(tf.stack([pred_end, answer_end])) - tf.reduce_min(tf.stack([pred_start, answer_start]))
        # #return 1 - (max(inter, 0) / (union + min(inter, 0)))
        # #tf.divide(tf.reduce_max(tf.stack([inter, 0])), union + tf.reduce_min(tf.stack([inter, 0])))
        # # return 1 - (tf.reduce_max(tf.stack([inter, 0])) / (union + tf.reduce_min(tf.stack([inter, 0]))))
        # #return tf.cast(1 - tf.divide(tf.reduce_max(tf.stack([inter, 0])), union + tf.reduce_min(tf.stack([inter, 0]))))
        # return 1 - (tf.reduce_max(tf.stack([inter, 0])) / (union + tf.reduce_min(tf.stack([inter, 0]))))
        
        
        # """ Calculates mean of Jaccard distance as a loss function """
        # intersection = tf.reduce_sum(answer * pred, axis=(1,2))
        # sum_ = tf.reduce_sum(answer + pred, axis=(1,2))
        # jac = (intersection + smooth) / (sum_ - intersection + smooth)
        # jd =  (1 - jac) * smooth
        # return tf.reduce_mean(jd)

        # #flatten label and prediction tensors
        # pred = K.flatten(pred)
        # answer = K.expand_dims(answer, -1)
        # print(pred)
        # print(answer)
        
        # intersection = K.sum(K.dot(answer, pred))
        # total = K.sum(answer) + K.sum(pred)
        # union = total - intersection
        
        # IoU = (intersection + smooth) / (union + smooth)
        # return 1 - IoU

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
        #print("output shape:", model.output_shape)
        assert model.output_shape == ()
        return model
    
    def create_end_predictor(self) -> Model:
        x_p = Input(shape=(self.embed_dim * 2 + self.feature_dim, 1))
        x_q = Input(shape=(self.embed_dim, 1))
        start_predictor = EndPredictor()([x_p, x_q])
        model = Model([x_p, x_q], start_predictor, name="End_Predictor")
        assert model.output_shape == ()
        return model
    
    def load_checkpoint(self, checkpoint_dir: str) -> tf.train.Checkpoint:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must not be 'None'. Please provide a path")
        
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
