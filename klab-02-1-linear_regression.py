from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

# prints summary of the model to the terminal
model.summary()

from keras.callbacks import History
history = History()

model.fit(x_train, y_train, nb_epoch=100, callbacks=[history])
y_predict = model.predict(np.array([4]))
print(y_predict)

print history.history

import os
# serialize model to JSON
model_json = model.to_json()
with open(os.path.basename(__file__)+".model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.basename(__file__)+".model.h5")
print("Saved model to disk")


from keras.utils import plot_model
plot_model(model, to_file=os.path.basename(__file__)+'.png', show_shapes=True)

#import matplotlib
import matplotlib.pyplot as plt0
import matplotlib.pyplot as plt1

plt0.plot(x_train,y_train,'ro', x_train, y_train)
plt0.axis([0, 5, -4, 0])
plt0.xlabel('x train')
plt0.ylabel('y train')
plt0.text(2, 1.5, r'$y =A x + b$', fontsize=15)
#plt0.text(2, 1, r'$A = 1$, $b = 0$', fontsize=15)
plt0.show()

plt1.plot(range(len(history.history['loss'])), history.history['loss'])
plt1.xlabel('step')
plt1.ylabel('loss')

plt1.show()

