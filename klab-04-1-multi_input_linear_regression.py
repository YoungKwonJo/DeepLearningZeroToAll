from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

x_data = [[73., 80., 75.], [93., 88., 93.], [
    89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=3))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop',  lr=1e-10)

from keras.callbacks import History
history = History()

model.fit(x_data, y_data, nb_epoch=1000, callbacks=[history])

print history.history

y_predict = model.predict(np.array([[95., 100., 80]]))
print(y_predict)


x0,x1,x2,x3,y0 = [],[],[],[],[]
import matplotlib.pyplot as plt0
for i in range(len(x_data)):
    x0.append(i)
    x1.append(x_data[i][0])
    x2.append(x_data[i][1])
    x3.append(x_data[i][2])
    y= model.predict(np.array([x_data[i]]))
    y0.append(y[0][0])

#print x_data
print "--------------------------"

#print x1
#print x2
#print x3
print model.summary()

plt0.figure(1)
plt0.subplot(411)
plt0.plot(x0,x1)
plt0.subplot(412)
plt0.plot(x0,x2)
plt0.subplot(413)
plt0.plot(x0,x3)
plt0.subplot(414)
plt0.plot(x0,y0)

plt0.show()


import matplotlib.pyplot as plt1

plt1.plot(range(len(history.history['loss'])), history.history['loss'])
plt1.xlabel('step')
plt1.ylabel('loss')

plt1.show()
