import numpy as np
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dense, RepeatVector, TimeDistributed
from keras.utils import plot_model

encoding_dim = 100
sequence = np.array([1, 3.1, 4, 4, 2.1, 3.5])
n_in = sequence.shape[0]
sequence = sequence.reshape((1, n_in, 1))
print(sequence)

model = Sequential()
model.add(LSTM(encoding_dim, activation='relu', input_shape=(n_in, 1)))
model.add(RepeatVector(n_in))
model.add(LSTM(encoding_dim, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

model.fit(sequence, sequence, epochs=300, verbose=0)

#plot_model(model, show_shapes=True, to_file='lstm_autoencoder.png')

yhat = model.predict(sequence, verbose=0)
print(yhat)

encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)

yhat2 = encoder.predict(sequence)
print(yhat2.shape)
print(yhat2)

print(model.layers)