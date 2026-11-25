#!/usr/bin/python3

# Muon optimizer, convolution kernels included, for Keras 3.0 (Jean Zay, tensorflow-gpu/py3/2.16.1) and later.
#
# Muon (K. Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/) replaces the update of a hidden weight matrix
# by the nearest orthogonal matrix to its momentum: same singular vectors, all singular values close to 1 (five
# Newton-Schulz iterations). No direction of the matrix dominates the step, as the few largest ones do with a plain
# gradient. A convolution kernel (kh, kw, in, out) is the weight matrix of a dense layer applied to every kh x kw x in
# patch, so it is flattened to (kh*kw*in, out) for this step, as in Jordan's CIFAR-10 runs. The update is scaled by
# 0.2 * sqrt(max(rows, columns)), which gives it the size of an AdamW update (Liu et al. 2025, arXiv:2502.16982), so
# the AdamW learning rate and weight decay can be kept.
#
# Biases, batch normalisation parameters and the layers named in exclude_layers (meant for the first convolution and
# the output layer) get the AdamW update instead. Decoupled weight decay, as in AdamW, applies to every weight.
#
# Adapted from keras.optimizers.Muon (Keras >= 3.10, Apache 2.0 licence), which does not exist in Keras 3.0 and
# leaves every convolution kernel to AdamW.

import math

import tensorflow as tf

ops = tf.keras.ops


@tf.keras.utils.register_keras_serializable(package="Neural_network")
class Muon(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, weight_decay=0.004, momentum=0.95, nesterov=True, ns_steps=5, rms_rate=0.2,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-7, exclude_layers=(), name="muon", **kwargs):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, name=name, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.rms_rate = rms_rate
        self.beta_1 = beta_1 # beta_1, beta_2 and epsilon: AdamW update, Keras defaults
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_layers = list(exclude_layers) # layer names, e.g. ["conv1_conv", model.layers[-1].name]

    def uses_muon(self, variable):
        # Dense kernels (in, out) and convolution kernels (kh, kw, in, out), outside the excluded layers
        if len(variable.shape) not in (2, 4):
            return False
        return not any(layer in variable.path.split("/") for layer in self.exclude_layers)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._momentums = [self.add_variable_from_reference(v, name="momentum") for v in var_list]
        # Second moments for the AdamW update only; None marks a Muon weight
        self._velocities = [None if self.uses_muon(v) else self.add_variable_from_reference(v, name="velocity") for v in var_list]

    def update_step(self, gradient, variable, learning_rate):
        i = self._get_variable_index(variable)
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        if self._velocities[i] is None:
            self._muon_step(gradient, variable, lr, self._momentums[i])
        else:
            self._adamw_step(gradient, variable, lr, self._momentums[i], self._velocities[i])

    def _muon_step(self, gradient, variable, lr, m):
        self.assign(m, ops.add(ops.multiply(m, self.momentum), gradient))
        g = ops.add(gradient, ops.multiply(m, self.momentum)) if self.nesterov else m
        shape = tuple(variable.shape)
        rows, columns = math.prod(shape[:-1]), shape[-1]
        update = self._orthogonalize(ops.reshape(g, (rows, columns)))
        update = ops.multiply(update, lr * self.rms_rate * math.sqrt(max(rows, columns)))
        self.assign_sub(variable, ops.reshape(update, shape))

    def _orthogonalize(self, g):
        # Quintic Newton-Schulz iteration of Jordan et al.: pushes every singular value of g towards 1 (roughly into
        # [0.7, 1.2], which works as well as the exact U V^T of the SVD) and is cheap on the GPU
        a, b, c = 3.4445, -4.7750, 2.0315
        transpose = g.shape[0] > g.shape[1] # iterate on the wide form, so that x x^T is the smaller Gram matrix
        x = ops.transpose(g) if transpose else g
        x = ops.divide(x, ops.add(ops.sqrt(ops.sum(ops.square(x))), 1e-7)) # Frobenius norm >= largest singular value
        for _ in range(self.ns_steps):
            gram = ops.matmul(x, ops.transpose(x))
            x = ops.add(ops.multiply(x, a), ops.matmul(ops.add(ops.multiply(gram, b), ops.multiply(ops.matmul(gram, gram), c)), x))
        return ops.transpose(x) if transpose else x

    def _adamw_step(self, gradient, variable, lr, m, v):
        # Same update as tf.keras.optimizers.Adam (the weight decay of AdamW is applied by the base class)
        t = ops.cast(self.iterations + 1, variable.dtype)
        alpha = lr * ops.sqrt(1 - ops.power(ops.cast(self.beta_2, variable.dtype), t)) / (1 - ops.power(ops.cast(self.beta_1, variable.dtype), t))
        self.assign_add(m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1))
        self.assign_add(v, ops.multiply(ops.subtract(ops.square(gradient), v), 1 - self.beta_2))
        self.assign_sub(variable, ops.divide(ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)))

    def get_config(self):
        config = super().get_config()
        config.update({"momentum": self.momentum, "nesterov": self.nesterov, "ns_steps": self.ns_steps,
                       "rms_rate": self.rms_rate, "beta_1": self.beta_1, "beta_2": self.beta_2, "epsilon": self.epsilon,
                       "exclude_layers": self.exclude_layers})
        return config
