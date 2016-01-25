import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import Gaussian
from cle.cle.data import Iterator
from cle.cle.utils import unpickle, tolist, OrderedDict
from cle.cle.utils.op import logsumexp

from sk.datasets.blizzard_h5 import Blizzard_h5_tbptt


data_path = '/raid/chungjun/data/blizzard_unseg/'
pkl_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/pkl/'

frame_size = 200
# How many examples you want to proceed at a time
batch_size = 188
debug = 0

pkl_name = 'm0_6_30000updates.pkl'

file_name = 'blizzard_unseg_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']

data = Blizzard_h5_tbptt(name='test',
                         path=data_path,
                         frame_size=frame_size,
                         file_name=file_name,
                         X_mean=X_mean,
                         X_std=X_std)

x = data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)

exp = unpickle(pkl_path + pkl_name)
nodes = exp.model.nodes
names = [node.name for node in nodes]


[main_lstm,
 x_1, x_2, x_3, x_4,
 theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig] = nodes


def inner_fn(x_t, s_tm1):

    theta_1_t = theta_1.fprop([s_tm1])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_mu_t = theta_mu.fprop([theta_4_t])
    theta_sig_t = theta_sig.fprop([theta_4_t])

    x_1_t = x_1.fprop([x_t])
    x_2_t = x_2.fprop([x_1_t])
    x_3_t = x_3.fprop([x_2_t])
    x_4_t = x_4.fprop([x_3_t])

    s_t = main_lstm.fprop([[x_4_t], [s_tm1]])

    return s_t, theta_mu_t, theta_sig_t

((s_t, theta_mu_t, theta_sig_t), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[main_lstm.get_init_state(batch_size),
                              None, None])

for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_theta_sig = theta_sig_t.reshape((theta_sig_t.shape[0]*theta_sig_t.shape[1], -1))

recon = Gaussian(reshaped_x, reshaped_theta_mu, reshaped_theta_sig)
recon = recon.reshape((theta_mu_t.shape[0], theta_mu_t.shape[1]))
recon_term = recon.sum(axis=0).mean()
recon_term.name = 'nll'

outputs = [recon_term]
monitor_fn = theano.function(inputs=[x],
                             outputs=outputs,
                             on_unused_input='ignore',
                             allow_input_downcast=True)

DataProvider = [Iterator(data, batch_size, start=2152704, end=2267008)]

data_record = []
for data in DataProvider:
    batch_record = []
    for batch in data:
        this_out = monitor_fn(*batch)
        batch_record.append(this_out)
    data_record.append(np.asarray(batch_record))
for record, data in zip(data_record, DataProvider):
    for i, ch in enumerate(outputs):
        this_mean = record[:, i].mean()
        if this_mean is np.nan:
            raise ValueError("NaN occured in output.")
        print("%s_%s: %f" % (data.name, ch.name, this_mean))
