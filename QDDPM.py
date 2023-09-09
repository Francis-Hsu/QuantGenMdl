import numpy as np
import ot
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.linalg import matrix_power
from opt_einsum import contract
from functools import partial
from itertools import combinations

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

class DiffusionModel(nn.Module):
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
    
    def scrambleCircuit_t(self, t, input, phis, gs=None):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        c = tc.Circuit(self.n, inputs=input)
        for tt in range(t):
            # single qubit rotations
            for i in range(self.n):
                c.rz(i, theta=phis[3*self.n*tt+i])
                c.ry(i, theta=phis[3*self.n*tt+self.n+i])
                c.rz(i, theta=phis[3*self.n*tt+2*self.n+i])
            # homogenous RZZ on every pair of qubits (n>=2)
            if self.n >= 2:
                for i, j in combinations(range(self.n), 2):
                    c.rzz(i, j, theta=gs[tt]/(2*np.sqrt(self.n)))
        return c.state()
    
    def set_diffusionData_t(self, t, inputs, diff_hs, seed):
        '''
        obtain the quantum data set through diffusion step t
        Args:
        t: diffusion step
        inputs: the input quantum data set
        diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
        '''
        # set single-qubit rotation angles
        np.random.seed(seed)
        phis = torch.rand(self.Ndata, 3*self.n*t)*np.pi/4. - np.pi/8.
        phis = phis*(diff_hs.repeat(3*self.n))
        if self.n > 1:
            # set homogenous RZZ gate angles
            gs = torch.rand(self.Ndata, t)*0.2 + 0.4
            gs *= diff_hs
        states = torch.zeros((self.Ndata, 2**self.n)).cfloat()
        for i in range(self.Ndata):
            if self.n > 1:
                states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i], gs[i])
            else:
                states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i])
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
            c.rx(i, theta=params[2*n_tot*l+i])
            c.ry(i, theta=params[2*n_tot*l+n_tot+i])
        for i in range(n_tot//2):
            c.cz(2*i, 2*i+1)
        for i in range((n_tot-1)//2):
            c.cz(2*i+1, 2*i+2)
    return c.state()


class QDDPM(nn.Module):
    def __init__(self, n, na, T, L):
        super().__init__()
        '''
        the QDDPM model: backward process
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit in each backward step
        '''
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.qclayer = tc.TorchLayer(partial(backCircuit, n_tot=self.n_tot, L=self.L), weights_shape=[2*self.n_tot*self.L],
                                     use_vmap=True, vectorized_argnums=0)

    def set_diffusionSet(self, states_diff):
        self.states_diff = torch.from_numpy(states_diff).cfloat()

    def randomSampleGeneration(self, Ndata):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(22)
        states_T = unitary_group.rvs(dim=2**self.n, size=Ndata)[:,:,0]
        return states_T

    def randomMeasure(self, input):
        '''
        Perform random meausurement on ancilla qubits in computational basis,
        return the output post-measuremenet state on data qubits.
        Currently only work on cpu
        '''
        q_idx = list(range(self.n_tot))
        c = tc.Circuit(self.n_tot, inputs=input)
        # the measurement result of ancillas
        zs, _ = c.measure_reference(*q_idx[:self.na])
        for i in range(self.na):
            c.post_select(i, keep=int(zs[i]))
            if int(zs[i]) == 1:
                c.x(i) # re-set every ancilla to be |0>
        post_state = c.state()[:2**self.n]
        normal_const = K.sqrt(K.real(post_state.conj() @ post_state))
        return post_state*(1./normal_const)
    
    def randomMeasureParallel(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Currently only work for one ancilla qubit.
        Args:
        inputs: states to be measured, first qubit is ancilla
        '''
        m_probs = torch.sum(torch.abs(inputs[:,:2**self.n])**2, axis=1) # the probability of measure ancilla |0>
        m_probs = torch.vstack((m_probs, 1.-m_probs)).T
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze() # measurment results
        post_state = torch.vstack((inputs[m_res==0, :2**self.n], \
                                   inputs[m_res==1, 2**self.n:])) # unnormlized post-state
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, axis=1)).unsqueeze(dim=1)
        post_state = 1./norms * post_state # normalize the state
        return post_state

    def backwardOutput_t(self, inputs, mseq=True):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        mseq: Boolean variable, True/False for sequential/parallel implementation
        '''
        output_full = self.qclayer(inputs) # outputs through quantum circuits before measurement
        # perform measurement
        if mseq == True:
            output_t = []
            for i in range(inputs.shape[0]):
                output_t.append(self.randomMeasure(output_full[i]))
            output_t = torch.vstack(output_t)
        else:
            output_t = self.randomMeasureParallel(output_full)
        return output_t
    
    def prepareInput_t(self, params_tot, t, Ndata):
        '''
        prepare the input samples for step t
        Args:
        params_tot: all circuit parameters till step t+1
        '''
        self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot)).cfloat()
        self.input_tplus1[:,:2**self.n] = torch.from_numpy(self.randomSampleGeneration(Ndata)).cfloat()
        params_tot = torch.from_numpy(params_tot).float()
        with torch.no_grad():
            for tt in range(self.T-1, t, -1):
                self.qclayer.q_weights[0] = params_tot[tt] # set quantum-circuit parameters
                self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, mseq=False)
        return self.input_tplus1


class QDDPM_cpu(nn.Module):
    def __init__(self, n, na, T, L):
        super().__init__()
        '''
        the QDDPM model: backward process only work on cpu
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit in each backward step
        '''
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), vectorized_argnums=0)

    def set_diffusionSet(self, states_diff):
        self.states_diff = torch.from_numpy(states_diff).cfloat()

    def HaarSampleGeneration(self, Ndata):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(22)
        states_T = unitary_group.rvs(dim=2**self.n, size=Ndata)[:,:,0]
        return torch.from_numpy(states_T).cfloat()

    def randomMeasure(self, input):
        '''
        Perform random meausurement on ancilla qubits in computational basis,
        return the output post-measuremenet state on data qubits.
        Currently only work on cpu
        '''
        q_idx = list(range(self.n_tot))
        c = tc.Circuit(self.n_tot, inputs=input)
        # the measurement result of ancillas
        zs, _ = c.measure_reference(*q_idx[:self.na])
        for i in range(self.na):
            c.post_select(i, keep=int(zs[i]))
            if int(zs[i]) == 1:
                c.x(i) # re-set every ancilla to be |0>
        post_state = c.state()[:2**self.n]
        normal_const = K.sqrt(K.real(post_state.conj() @ post_state))
        return post_state*(1./normal_const)
    
    def randomMeasureParallel(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Currently only work for one ancilla qubit.
        Args:
        inputs: states to be measured, first qubit is ancilla
        '''
        m0_probs = torch.sum(torch.abs(inputs[:, :2**self.n])**2, axis=1) # the probability of measure ancilla |0>
        m1_probs = torch.sum(torch.abs(inputs[:, 2**self.n:])**2, axis=1)
        m_probs = torch.vstack((m0_probs, m1_probs)).T
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze() # measurment results
        post_state = torch.vstack((inputs[m_res==0, :2**self.n], \
                                   inputs[m_res==1, 2**self.n:])) # unnormlized post-state
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, axis=1)).unsqueeze(dim=1)
        post_state = 1./norms * post_state # normalize the state
        return post_state

    def backwardOutput_t(self, inputs, params, mseq=True):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        mseq: Boolean variable, True/False for sequential/parallel implementation
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_vmap(inputs, params) 
        # perform measurement
        if mseq == True:
            output_t = []
            for i in range(inputs.shape[0]):
                output_t.append(self.randomMeasure(output_full[i]))
            output_t = torch.vstack(output_t)
        else:
            output_t = self.randomMeasureParallel(output_full)
        return output_t
    
    def prepareInput_t(self, params_tot, t, Ndata):
        '''
        prepare the input samples for step t
        Args:
        params_tot: all circuit parameters till step t+1
        '''
        self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot)).cfloat()
        self.input_tplus1[:,:2**self.n] = self.randomSampleGeneration(Ndata)
        params_tot = torch.from_numpy(params_tot).float()
        with torch.no_grad():
            for tt in range(self.T-1, t, -1):
                self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tot[tt], 
                                                                        mseq=False)
        return self.input_tplus1
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = torch.zeros((self.T+1, Ndata, 2**self.n_tot)).cfloat()
        states[-1, :, :2**self.n] = inputs_T
        params_tot = torch.from_numpy(params_tot).float()
        with torch.no_grad():
            for tt in range(self.T-1, -1, -1):
                states[tt, :, :2**self.n] = self.backwardOutput_t(states[tt+1], params_tot[tt], 
                                                                        mseq=False)
        return states


def naturalDistance(Set1, Set2):
    '''
        a natural measure on the distance between two sets of quantum states
        definition: 2*d - r1-r2
        d: mean of inter-distance between Set1 and Set2
        r1/r2: mean of intra-distance within Set1/Set2
    '''
    # a natural measure on the distance between two sets, according to trace distance
    r11 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set1))**2)
    r22 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set2.conj(), Set2))**2)
    r12 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set2))**2)
    return 2*r12 - r11 - r22


def diffusionDistance(Set1, Set2, band_width=0.05, q=None):
    '''
        diffusion distance to measure the distance between two sets of quantum states
        bandwidth: band width of the RBF kernel
        q: number of diffusion steps 
    '''
    # calculate distance matrix
    Set = torch.vstack((Set1, Set2))
    S = torch.abs(contract('mi, ni->mn', Set.conj(), Set))**2
    Kn = 1. - S

    Ndata1 = Set1.shape[0]
    if q is None:
        q = 2 * int(S.shape[0] ** 1.1)

    # compute the kernel matrix
    Kn /= 2.0 * band_width ** 2.0
    Kn = torch.exp(-Kn)

    Dinv = Kn.sum(axis=1)  # diagonal of inverse degree
    Dinv = 1.0 / Dinv
    P = (Dinv * Kn.T).T
    P = matrix_power(P, q)
    A = P * Dinv  # affinity matrix
    return torch.mean(A[:Ndata1, :Ndata1]) + torch.mean(A[Ndata1:, Ndata1:])\
            - 2*torch.mean(A[:Ndata1, Ndata1:])


def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - torch.abs(Set1.conj() @ Set2.T)**2.
    emt = torch.empty(0)
    Wass_dis = ot.emd2(emt, emt, M=D)
    return Wass_dis


def sinkhornDistance(Set1, Set2, reg=0.005, log=False):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    D = 1. - torch.abs(Set1.conj() @  Set2.T)**2.
    emt = torch.empty(0)
    if log == True:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, method='sinkhorn_log')
    else:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg)
    return sh_dis


def batchTraining_t(model, t, params_tot, Ndata, batchsize, epochs, dis_measure='nat', dis_params={}):
    '''
    the batch trianing for the backward PQC at step t
    Args:
    model: the QDDPM
    t: diffusion step
    params_tot: collection of PQC parameters for steps > t 
    Ndata: number of samples in training data set
    batchsize: number of samples in one batch
    epochs: number of iterations
    dis_measure: the distance measure to compare two distributions of quantum states
    dis_params: potential hyper-parameters for distance measure
    '''
    input_tplus1 = model.prepareInput_t(params_tot, t, Ndata) # prepare input

    states_diff = model.states_diff
    loss_hist = [] # record of training history
    ystd_hist = [] # record of std of bloch-y cooredinates

    batchnum = Ndata//batchsize # the number of batches

    sy = torch.from_numpy(np.array([[0,-1j], [1j, 0]])).cfloat()
    
    np.random.seed()
    params_t = torch.tensor(np.zeros(2*model.n_tot*model.L), dtype=torch.float32, requires_grad=True)
    # set optimizer and learning rate decay
    optimizer = torch.optim.SGD(params=(params_t, ), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=400, gamma=0.5)
    for step in range(epochs):
        b_indices = np.split(np.random.permutation(Ndata), batchnum)
        for b in range(batchnum):
            # optimize over each batch
            optimizer.zero_grad()
            td_indices = np.random.choice(states_diff.shape[1], size=Ndata, replace=False)
            true_data = states_diff[t, td_indices]

            output_t = model.backwardOutput_t(input_tplus1[b_indices[b]], params_t, mseq=False)
            if dis_measure == 'nat':
                # natural distance
                loss = naturalDistance(output_t, true_data)
            elif dis_measure == 'dd':
                # diffusion distance
                loss = diffusionDistance(output_t, true_data, band_width=dis_params['bandwidth'])
            elif dis_measure == 'wd':
                # Wassastein distance
                loss = WassDistance(output_t, true_data)
            elif dis_measure == 'shd':
                # sinkhorn distance
                loss = sinkhornDistance(output_t, true_data, reg=dis_params['reg'], log=dis_params['log'])
            
            loss.backward()
            optimizer.step()

        scheduler.step()

        loss_hist.append(loss.detach()) # record the current loss
        ys = torch.real(contract('mi,ij,mj->m', output_t.detach().conj(), sy, output_t.detach()))
        ystd_hist.append(torch.std(ys))
        
        if (step+1)%10 == 0 or not step:
            print(step+1, loss_hist[step], ystd_hist[step])
        
    return params_t.detach(), torch.stack(loss_hist).squeeze(), torch.stack(ystd_hist).squeeze()
