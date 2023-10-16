import numpy as np
import ot
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group

import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from opt_einsum import contract
from functools import partial
from itertools import combinations

K = tc.set_backend('tensorflow')
tc.set_dtype('complex64')

def backCircuit(input, params, n_tot, L):
    '''
    the backward denoise parameteric quantum circuits,
    designed following the hardware-efficient ansatz
    output is the state before measurmeents on ancillas
    Args:
    input: input quantum state of n_tot qubits
    params: the parameters of the circuit
    n_tot: number of qubits in the circuits
    L: layers of circuit
    '''
    c = tc.Circuit(n_tot, inputs=input)
    for l in range(L):
        for i in range(n_tot):
            c.rx(i, theta=params[2*n_tot*l+i])
            c.ry(i, theta=params[2*n_tot*l+n_tot+i])
        for i in range(n_tot//2):
            c.cz(2*i, 2*i+1)
        for i in range((n_tot-1)//2):
            c.cz(2*i+1, 2*i+2)
    return c.state()

class QDT():
    def __init__(self, n, na, L):
        '''
        the QDT model: backward process only work on cpu
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        L: layers of circuit
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), 
                                             vectorized_argnums=0))

    def set_diffusionSet(self, states_diff):
        self.states_diff = tf.cast(tf.convert_to_tensor(states_diff), dtype=tf.complex64)

    @tf.function
    def randomMeasure(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        n_batch = inputs.shape[0]
        m_probs = tf.abs(tf.reshape(inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
        m_probs = tf.reduce_sum(m_probs, axis=2)
        m_res = tfp.distributions.Categorical(probs=m_probs).sample(1)
        indices = 2 ** self.n * tf.reshape(m_res, [-1, 1]) + tf.range(2 ** self.n)
        post_state = tf.gather(inputs, indices, batch_dims=1)
        
        return tf.linalg.normalize(post_state, axis=1)[0]
    
    def backwardOutput_t(self, inputs, params):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_vmap(inputs, params) 
        # perform measurement
        output_t = self.randomMeasure(output_full)

        return output_t
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = [inputs_T]
        input_tplus1 = tf.concat([inputs_T, tf.zeros(shape=(Ndata, 2**self.n_tot - 2**self.n), 
                                                          dtype=tf.complex64)], axis=1)
        params_tot = tf.cast(tf.convert_to_tensor(params_tot), dtype=tf.float32)
        
        output = self.backwardOutput_t(input_tplus1, params_tot)
        states.append(output)
        states = tf.stack(states)[::-1]
        return states