import os
import numpy as np
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group
import torch
from torch.optim.lr_scheduler import StepLR
from opt_einsum import contract
from QDDPM import DiffusionModel, QDDPM_cpu
from QDDPM import naturalDistance, WassDistance, sinkhornDistance
import time


def initialState(seed):
    '''
    two qubit initial state with |10> amplitude 0
    '''
    np.random.seed(seed)
    Psi0 = np.random.randn(4) + 1j*np.random.randn(4)
    Psi0[2] = 0
    return Psi0/np.linalg.norm(Psi0)


def corrNoiseEnsemble(psi0, delta, p, Ndata, seed):
    '''
    generate random states by applying exp(-i\epsilon XX) with probability p
    or exp(-i\epsilon ZZ) with probability 1-p on \psi_0
    \epsilon \in [-\delta, \delta]
    '''
    np.random.seed(seed)
    noise_types = np.random.choice(
        [1, 2], size=Ndata, replace=True, p=[p, 1-p])
    X = np.array([[0, 1.], [1., 0]])
    Z = np.array([[1., 0], [0, -1.]])
    XX = contract('ij,kl->ikjl', X, X).reshape((4, 4))
    ZZ = contract('ij,kl->ikjl', Z, Z).reshape((4, 4))
    angles = np.random.uniform(-delta, delta, Ndata).reshape((Ndata, 1, 1))
    xx_angles = angles[noise_types == 1]
    zz_angles = angles[noise_types == 2]
    Rxx = np.cos(xx_angles)*np.eye(4) - 1j*np.sin(xx_angles)*XX
    Rzz = np.cos(zz_angles)*np.eye(4) - 1j*np.sin(zz_angles)*ZZ
    Us = np.vstack((Rxx, Rzz))
    states = contract('mij, j->mi', Us, psi0)
    return states


def training_t(model, t, inputs_T, params_tot, Ndata, epochs, dis_measure='nat'):
    '''
    training for the backward PQC at step t using whole dataset
    Args:
    model: QDDPM model
    t: diffusion step
    params_tot: collection of PQC parameters for steps > t 
    Ndata: number of samples in training data set
    epochs: number of iterations
    dis_measure: the distance measure to compare two distributions of quantum states
    '''
    input_tplus1 = model.prepareInput_t(
        inputs_T, params_tot, t, Ndata)  # prepare input
    states_diff = model.states_diff
    loss_hist = []  # record of training history
    f_hist = []  # record of fidelity with |10>

    # initialize parameters
    np.random.seed()
    params_t = torch.tensor(np.random.randn(
        2*model.n_tot*model.L), dtype=torch.float32, requires_grad=True)

    # set optimizer and learning rate decay
    optimizer = torch.optim.SGD(params=(params_t, ), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

    for step in range(epochs):
        optimizer.zero_grad()
        indices = np.random.choice(
            states_diff.shape[1], size=Ndata, replace=False)
        true_data = states_diff[t, indices]
        output_t = model.backwardOutput_t(input_tplus1, params_t)
        if dis_measure == 'nat':
            # natural distance
            loss = naturalDistance(output_t, true_data)
        elif dis_measure == 'wd':
            # Wassastein distance
            loss = WassDistance(output_t, true_data)
        loss_hist.append(loss.detach())  # record the current loss
        f_hist.append(torch.mean(torch.abs(output_t.detach()[:, 2])**2))

        loss.backward()
        optimizer.step()
        scheduler.step()
    return params_t.detach(), torch.stack(loss_hist).squeeze(), torch.stack(f_hist).squeeze()


'''
# forward diffusion
n = 2
T = 20
N = 5000
diff_hs = np.hstack((np.linspace(0.05, 1., 12), np.linspace(1.2, 4., 8)))
print(diff_hs)
model_diff = DiffusionModel(n, T, N)

psi0 = initialState(12)
X = torch.from_numpy(corrNoiseEnsemble(psi0, 0.3, 0.8, N, seed=12)).cfloat()

Xout = np.zeros((T+1, N, 2**n), dtype=np.complex64)
Xout[0] = X

t1 = time.time()
for t in range(1, T+1):
    Xout[t] = model_diff.set_diffusionData_t(t, X, diff_hs[:t], seed=t).numpy()

np.save('data/QDDPM/noise/n2/corrNoiseDiff_n%dT%d_N%d.npy'%(n, T, N), Xout)
t2 = time.time()
print(t2-t1)
'''

# backward training
n, na = 2, 2
T = 20
L = 6
Ndata = 500
epochs = 2000
repeats = 5
method = 'nat'

diffModel = DiffusionModel(n, T, Ndata)
inputs_T = diffModel.HaarSampleGeneration(Ndata, seed=22)

model = QDDPM_cpu(n=n, na=na, T=T, L=L)
states_diff = np.load(
    'data/QDDPM/noise/n2/corrNoiseDiff_n%dT%d_N5000.npy' % (n, T))
model.set_diffusionSet(states_diff)

data_path = "data/QDDPM/noise/n%d/record_%s/" % (n, method)
if not os.path.exists(data_path):
    os.makedirs(data_path)

data_path = "data/QDDPM/circle/n%d/record_%s/" % (n, method)
if not os.path.exists(data_path):
    os.makedirs(data_path)

for t in range(19, 14, -1):
    params_tot = np.zeros((20, 2*(n+na)*L))
    for tt in range(t+1, T):
        params_tot[tt] = np.load('data/QDDPM/noise/n2/record_%s/QDDPMcircleYparams_n%dna%dT%dL%d_t%d_%s.npy'
                                 % (method, n, na, T, L, tt, method))

    t1 = time.time()
    params_all = np.zeros((repeats, 2*(n+na)*L))
    loss_all = np.zeros((repeats, epochs))
    f_all = np.zeros((repeats, epochs))

    for trial in range(repeats):
        params_all[trial], loss_all[trial], f_all[trial] = training_t(
            model, t, inputs_T, params_tot, Ndata, epochs, dis_measure=method)
        print(t, trial, loss_all[trial, -1])

    # find the best one in terms of minimal loss
    idx = np.argmin(loss_all[:, -1])
    # record the best result
    np.save('data/QDDPM/noise/n%d/record_%s/QDDPMcorrNoiseparams_n%dna%dT%dL%d_t%d_%s.npy'
            % (n, method, n, na, T, L, t, method), params_all[idx])
    np.save('data/QDDPM/circle/n%d/record_%s/QDDPMcorrNoiseloss_n%dna%dT%dL%d_t%d_%s.npy'
            % (n, method, n, na, T, L, t, method), loss_all[idx])
    np.save('data/QDDPM/circle/n%d/record_%s/QDDPMcorrNoisef_n%dna%dT%dL%d_t%d_%s.npy'
            % (n, method, n, na, T, L, t, method), f_all[idx])
    t2 = time.time()

    print('corr-noise, na=%d, t=%d, min loss=%s, F10=%s, time=%s' %
          (na, t+1, loss_all[idx, -1], f_all[idx, -1], t2-t1))
