import numpy as np
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dense, RepeatVector, TimeDistributed
from keras.utils import plot_model

encoding_dim = 2
sequence = np.array([[1, 3.1, 4, 4, 2.1, 3.5], [1.6, 3.1, 0, 4, 2.1, 8.5]])
#sequence = np.array([1, 3.1, 4, 4, 2.1, 3.5])
n_in = sequence.shape[0]
input_dim = sequence.shape[1]
print(sequence)
print(sequence.shape)
sequence = sequence.reshape((n_in, input_dim, 1))
print(sequence)

model = Sequential()
model.add(LSTM(encoding_dim, activation='relu', input_shape=(input_dim, 1)))
model.add(RepeatVector(input_dim))
model.add(LSTM(encoding_dim, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

model.fit(sequence, sequence, epochs=300, verbose=0)

plot_model(model, show_shapes=True, to_file='lstm_autoencoder.png')

yhat = model.predict(sequence, verbose=0)
print(yhat)

encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)

print(len(model.get_weights()), [model.get_weights()[i].shape for i in range(len(model.get_weights()))])

yhat2 = encoder.predict(sequence)
print(yhat2.shape)
print(yhat2)

print(model.layers)