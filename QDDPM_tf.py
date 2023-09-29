from functools import partial
from itertools import combinations

import ot
import numpy as np
import scipy as sp
from scipy.stats import unitary_group

import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config

import tensorcircuit as tc

np_config.enable_numpy_behavior()

# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import StepLR
# from torch.linalg import matrix_power
from opt_einsum import contract

K = tc.set_backend('tensorflow')
tc.set_dtype('complex64')

@tf.function
def scrambleCircuitOneQubit(input, phis):
    '''
    obtain the state through diffusion step t
    Args:
    t: diffusion step
    input: the input quantum state
    phis: the single-qubit rotation angles in diffusion circuit
    gs: the angle of RZZ gates in diffusion circuit when n>=2
    '''
    # input, phis = params
    t = phis.shape[0] // 3
    c = tc.Circuit(1, inputs=input)

    for s in range(t):
        # single qubit rotations
        c.rz(0, theta=phis[3 * s])
        c.ry(0, theta=phis[3 * s + 1])
        c.rz(0, theta=phis[3 * s + 2])

    return c.state()

def setDiffusionDataOneQubit(inputs, diff_hs):
    '''
    obtain the quantum data set for 1 qubit through diffusion step t
    Args:
    t: diffusion step
    inputs: the input quantum data set
    diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
    '''
    t = diff_hs.shape[0]
    Ndata = inputs.shape[0]
    diff_hs = tf.repeat(diff_hs, 3)

    # set single-qubit rotation angles
    tf.random.set_seed(t)
    phis = tf.random.uniform((Ndata, 3 * t)) * np.pi / 4. - np.pi / 8.
    phis *= diff_hs

    # states = tf.vectorized_map(partial(self.scrambleCircuit_t, t=t), (inputs, phis))
    states = K.vmap(scrambleCircuitOneQubit, vectorized_argnums=(0, 1))(inputs, phis)

    return states

class OneQubitDiffusionModel():
    def __init__(self, T, Ndata):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.t = 0
        self.T = T
        self.Ndata = Ndata
    
    def HaarSampleGeneration(self, Ndata, seed):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(seed)
        states_T = unitary_group.rvs(dim=2, size=Ndata)[:,:,0]

        return tf.cast(tf.convert_to_tensor(states_T), dtype=tf.complex64)
    

class MultiQubitDiffusionModel():
    def __init__(self, n, T, Ndata):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.n = n
        self.T = T
        self.Ndata = Ndata
    
    def HaarSampleGeneration(self, Ndata, seed):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(seed)
        states_T = unitary_group.rvs(dim=2 ** self.n, size=Ndata)[:,:,0]

        return tf.cast(tf.convert_to_tensor(states_T), dtype=tf.complex64)
    
    def scrambleCircuit_t(self, params, t):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        input, phis, gs = params
        c = tc.Circuit(self.n, inputs=input)
        for s in range(t):
            # single qubit rotations
            for i in range(self.n):
                c.rz(i, theta=phis[3 * self.n * s + i])
                c.ry(i, theta=phis[3 * self.n * s + self.n + i])
                c.rz(i, theta=phis[3 * self.n * s + 2*self.n + i])

            # homogenous RZZ on every pair of qubits (n>=2)
            for i, j in combinations(range(self.n), 2):
                c.rzz(i, j, theta=gs[s] / (2 * self.n ** 0.5))

        return c.state()
        
    def set_diffusionDataMulti_t(self, t, inputs, diff_hs, seed):
        '''
        obtain the quantum data set for multiple qubit through diffusion step t
        Args:
        t: diffusion step
        inputs: the input quantum data set
        diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
        '''
        # set single-qubit rotation angles
        tf.random.set_seed(seed)
        phis = tf.random.uniform((self.Ndata, 3 * self.n * t)) * np.pi / 4. - np.pi / 8.
        phis *= tf.repeat(diff_hs, 3 * self.n)

        # set homogenous RZZ gate angles
        gs = tf.random.uniform((self.Ndata, t)) * 0.2 + 0.4
        gs *= diff_hs
        
        states = tf.vectorized_map(partial(self.scrambleCircuit_t, t=t), (inputs, phis, gs))

        return states

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
            c.rx(i, theta=params[2* n_tot * l + i])
            c.ry(i, theta=params[2* n_tot* l + n_tot + i])

        for i in range(n_tot // 2):
            c.cz(2 * i, 2 * i + 1)

        for i in range((n_tot-1) // 2):
            c.cz(2 * i + 1, 2 * i + 2)

    return c.state()

# '''
# class QDDPM_cpu():
#     def __init__(self, n, na, T, L):
#         '''
#         the QDDPM model: backward process only work on cpu
#         Args:
#         n: number of data qubits
#         na: number of ancilla qubits
#         T: number of diffusion steps
#         L: layers of circuit in each backward step
#         '''
#         super().__init__()
#         self.n = n
#         self.na = na
#         self.n_tot = n + na
#         self.T = T
#         self.L = L
#         # embed the circuit to a vectorized pytorch neural network layer
#         self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), vectorized_argnums=0))

#     def set_diffusionSet(self, states_diff):
#         self.states_diff = torch.from_numpy(states_diff).cfloat()

#     def randomMeasure(self, inputs):
#         '''
#         Given the inputs on both data & ancilla qubits before measurmenets,
#         calculate the post-measurement state.
#         The measurement and state output are calculated in parallel for data samples
#         Args:
#         inputs: states to be measured, first na qubit is ancilla
#         '''
#         n_batch = inputs.shape[0]
#         m_probs = tf.abs(tf.reshape(inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
#         m_probs = tf.reduce_sum(m_probs, axis=2)
#         m_res = tfp.distributions.Categorical(probs=m_probs).sample(1)
#         indices = 2 ** self.n * tf.reshape(m_res, [-1, 1]) + tf.range(2 ** self.n)
#         post_state = tf.gather(inputs, indices, batch_dims=1)
        
#         return tf.linalg.normalize(post_state, axis=1)

#     def backwardOutput_t(self, inputs, params):
#         '''
#         Backward denoise process at step t
#         Args:
#         inputs: the input data set at step t
#         '''
#         # outputs through quantum circuits before measurement
#         output_full = self.backCircuit_vmap(inputs, params) 
#         # perform measurement
#         output_t = self.randomMeasure(output_full)

#         return output_t
    
#     def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
#         '''
#         prepare the input samples for step t
#         Args:
#         inputs_T: the input state at the beginning of backward
#         params_tot: all circuit parameters till step t+1
#         '''
#         self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot)).cfloat()
#         self.input_tplus1[:,:2**self.n] = inputs_T
#         params_tot = torch.from_numpy(params_tot).float()
#         with torch.no_grad():
#             for tt in range(self.T-1, t, -1):
#                 self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tot[tt])

#         return self.input_tplus1
    
#     def backDataGeneration(self, inputs_T, params_tot, Ndata):
#         '''
#         generate the dataset in backward denoise process with training data set
#         '''
#         states = torch.zeros((self.T+1, Ndata, 2**self.n_tot)).cfloat()
#         states[-1, :, :2**self.n] = inputs_T
#         params_tot = torch.from_numpy(params_tot).float()
#         with torch.no_grad():
#             for tt in range(self.T-1, -1, -1):
#                 states[tt, :, :2**self.n] = self.backwardOutput_t(states[tt+1], params_tot[tt])

#         return states
# '''


class QDDPM():
    def __init__(self, n, na, T, L):
        '''
        the QDDPM model: backward process only work on cpu
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit in each backward step
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.jit(K.vmap(partial(backCircuit, n_tot=self.n_tot, L=self.L), vectorized_argnums=0))

    def set_diffusionSet(self, states_diff):
        self.states_diff = tf.convert_to_tensor(states_diff)

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
    
    def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
        '''
        prepare the input samples for step t
        Args:
        inputs_T: the input state at the beginning of backward
        params_tot: all circuit parameters till step t+1
        '''
        zero_tensor = tf.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), dtype=tf.complex64)
        self.input_tplus1 = tf.concat([inputs_T, zero_tensor], axis=1)
        params_tot = tf.constant(params_tot, dtype=tf.float32)
        for tt in range(self.T-1, t, -1):
            output = self.backwardOutput_t(self.input_tplus1, params_tot[tt])
            self.input_tplus1 = tf.concat([output, zero_tensor], axis=1)

        return self.input_tplus1
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = [inputs_T]
        input_tplus1 = tf.concat([inputs_T, tf.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                          dtype=tf.complex64)], axis=1)
        params_tot = tf.cast(tf.convert_to_tensor(params_tot), dtype=tf.float32)
        for tt in range(self.T-1, -1, -1):
            output = self.backwardOutput_t(input_tplus1, params_tot[tt])
            input_tplus1 = tf.concat([output, tf.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                            dtype=tf.complex64)], axis=1)
            states.append(output)
        states = tf.stack(states)[::-1]
        return states

@tf.function
def naturalDistance(Set1, Set2):
    '''
        a natural measure on the distance between two sets of quantum states
        definition: 2*d - r1-r2
        d: mean of inter-distance between Set1 and Set2
        r1/r2: mean of intra-distance within Set1/Set2
    '''
    # a natural measure on the distance between two sets, according to trace distance
    r11 = 1. - tf.reduce_mean(tf.abs(contract('mi,ni->mn', tf.math.conj(Set1), Set1))**2)
    r22 = 1. - tf.reduce_mean(tf.abs(contract('mi,ni->mn', tf.math.conj(Set2), Set2))**2)
    r12 = 1. - tf.reduce_mean(tf.abs(contract('mi,ni->mn', tf.math.conj(Set1), Set2))**2)
    
    return 2 * r12 - r11 - r22


def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - tf.abs(tf.math.conj(Set1) @ tf.transpose(Set2))**2.
    emt = tf.constant([], dtype=tf.float32)
    Wass_dis = ot.emd2(emt, emt, M=D)

    return Wass_dis

def sinkhornDistance(Set1, Set2, reg=0.005, eps=1e-4, log=False):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    D = 1. - tf.abs(tf.math.conj(Set1) @ tf.transpose(Set2)) ** 2.
    emt = tf.constant([], dtype=tf.float32)
    if log == True:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, stopThr=eps, method='sinkhorn_stabilized')
    else:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, stopThr=eps)
        
    return sh_dis

def diffusionDistance(Set1, Set2, band_width=0.05, q=None):
    '''
        diffusion distance to measure the distance between two sets of quantum states
        bandwidth: band width of the RBF kernel
        q: number of diffusion steps 
    '''
    # calculate distance matrix
    Set = tf.concat([Set1, Set2], 0)
    S = tf.abs(contract('mi, ni->mn', tf.math.conj(Set), Set)) ** 2.0
    Kn = 1. - S

    Ndata1 = Set1.shape[0]
    if q is None:
        q = 2 * int(S.shape[0] ** 1.1)

    # compute the kernel matrix
    Kn /= 2.0 * band_width ** 2.0
    Kn = tf.exp(-Kn)

    Dinv = tf.reduce_sum(Kn, axis=1) # diagonal of inverse degree
    Dinv = 1.0 / Dinv
    P = tf.transpose(Dinv * tf.transpose(Kn))

    # matrix power by eigendecomposition
    D, Q = tf.linalg.eig(P)
    P = Q @ tf.transpose(tf.pow(D, q) * tf.transpose(tf.linalg.inv(Q)))
    P = tf.math.real(P)

    A = P * Dinv # affinity matrix

    return tf.reduce_mean(A[:Ndata1, :Ndata1]) + tf.reduce_mean(A[Ndata1:, Ndata1:]) - 2 * tf.reduce_mean(A[:Ndata1, Ndata1:])
