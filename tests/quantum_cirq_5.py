# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    def __init__(self, units=32) -> None:
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.initializers.constant(value=1),
            trainable=True,
            name="w"
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer=tf.initializers.constant(value=0),
            trainable=True,
            name="b"
        )
        self.n_calls = self.add_weight(
            shape=(),
            initializer=tf.initializers.constant(value=121),
            trainable=False,
        )

    def call(self, inputs):
        self.n_calls = self.n_calls.assign_add(inputs.shape[0])
        return inputs @ self.w + self.b
        # return  tf.einsum("ij,j", inputs, self.w) + self.b


X = np.matrix([[1, 2, 3]], dtype=np.float32)

# %%
layer = Linear(units=1)
c_1 = layer(X)
print(c_1)
# %%
x = tf.Variable(1.0)


def f(x):
    return x**2 + 2 * x - 5


f(x)
# %%
w = tf.Variable(tf.constant([1, 2, 3], tf.float32), trainable=True, name="w")
b = tf.Variable(tf.constant([[1, 1, 1]], tf.float32), trainable=True, name="b")
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)
with tf.GradientTape(persistent=True) as tape:
    y = tf.einsum("ij,j->i", x, w) + b
    loss = tf.reduce_mean(tf.abs(y), 0)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])

# %%
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams["figure.figsize"] = [9, 6]

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)


def f(x):
    y = x**2 + 2 * x - 5
    return y


y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), ".", label="Data")
plt.plot(x, f(x), label="Ground truth")
plt.legend()
# %%


class Model(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.Linear = Linear(units=units)
        self.final_layer = Linear(1)

    def call(self, x, training=True):
        x = x[:, tf.newaxis]
        x = self.Linear(x)
        x = self.final_layer(x)
        return x


model = Model(32)
# %%
plt.plot(x.numpy(), y.numpy(), ".", label="data")
plt.plot(x, f(x), label="Ground truth")
plt.plot(x, model(x), label="Untrained predictions")
plt.title("Before training")
plt.legend()
# %%
variables = model.trainable_variables
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for step in range(1000):
    with tf.GradientTape() as tape:
        prediction = model(x)
        error = (y - prediction) ** 2
        mean_error = tf.reduce_mean(error)
    gradient = tape.gradient(mean_error, variables)
    optimizer.apply_gradients(zip(gradient, variables))

    if step % 100 == 0:
        print(f"Mean Squared error: {mean_error.numpy():0.3f}")

# %%
plt.plot(x.numpy(), y.numpy(), ".", label="data")
plt.plot(x, f(x), label="Ground truth")
plt.plot(x, model(x), label="Trained predictions")
plt.title("After training")
plt.legend()
# %%
