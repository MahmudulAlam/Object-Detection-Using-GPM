import tensorflow as tf
from network import model
from generator import train_generator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


def loss_function(y_true, y_pred):
    loss = - tf.math.xlogy(y_true, y_pred) - tf.math.xlogy((1 - y_true), (1 - y_pred))
    loss = tf.reduce_sum(loss)
    return loss


# Creating the model
model = model()
model.summary()
model.load_weights('weights/weights_001.h5', by_name=True, skip_mismatch=True)

# Compile
adam = Adam(lr=1e-7, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
model.compile(optimizer=adam, loss=loss_function, metrics=None)

# Train
epochs = 1
batch_size = 24
train_set_size = 118287

train_gen = train_generator(batch_size=batch_size)

checkpoints = ModelCheckpoint('weights/weights_{epoch:03d}.h5', save_weights_only=True, save_freq=500)
history = model.fit(train_gen, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
                    steps_per_epoch=train_set_size / batch_size, callbacks=[checkpoints], max_queue_size=32)

print('Model Saved')

with open('weights/history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')
