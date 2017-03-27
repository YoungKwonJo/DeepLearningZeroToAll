from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd',
              lr=0.1, metrics=['accuracy'])

from keras.callbacks import History
history = History()

model.summary()
#model.fit(x_data, y_data, nb_epoch=50000)
model.fit(x_data, y_data, nb_epoch=50000, callbacks=[history])


print history.history

print(model.predict_classes(x_data))

score = model.evaluate(x_data, y_data, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

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


#import matplotlib.pyplot as plt0
import matplotlib.pyplot as plt1

#plt1.plot(range(len(history.history['loss'])), history.history['loss'])
plt1.figure(1)
plt1.subplot(211)

plt1.ylabel('loss')
plt1.plot(range(len(history.history['loss'])), history.history['loss'])

plt1.subplot(212)
plt1.ylabel('acc')
plt1.plot(range(len(history.history['acc'])), history.history['acc'])
plt1.xlabel('step')


plt1.show()
