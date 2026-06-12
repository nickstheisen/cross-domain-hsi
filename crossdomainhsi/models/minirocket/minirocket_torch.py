import torch.nn as nn
import torch.nn.functional as F
import logging
import torch
from .hdc_utils import *
from pprint import pprint

logger = logging.getLogger('log')

class Minirocket_Encoder(torch.nn.Module):
    def __init__(self,
                 dim=10000,
                 n_channels=1,
                 seq_len=None,
                 use_hdc=False,
                 seed=42,
                 config=None,
                 batch_size=512):
        super(Minirocket_Encoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.dim = dim
        self.seed = seed
        self.use_hdc = use_hdc
        self.batch_size = batch_size
        self.config = config
        self.n_channels = n_channels
        self.seq_length = seq_len
        self.encoder = self.minirocket()
        self.poses = None
        
    def forward(self, inputs):
        # create embeddings
        x_mr = self.encoder(inputs)

        # pad to dim (to have the same shape as the transformer)
        x_mr = torch.nn.functional.pad(x_mr, (0, self.dim - x_mr.shape[-1]), "constant", 0)
        return x_mr

    def transform(self, inputs):
        if inputs.shape[0] > self.batch_size:
            print("More than 1000 samples, splitting into batches of 512")
            sub_batch_size = self.batch_size
            x_tf = np.zeros((inputs.shape[0], self.dim), dtype=np.float32)
            for i in range(0, inputs.shape[0], sub_batch_size):
                print(f"Batch {i} to {i+sub_batch_size}")
                x = inputs[i:i+sub_batch_size]
                x_tf[i:i+sub_batch_size] = self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()
        else:
            x_tf = self.forward(torch.from_numpy(inputs.astype(np.float32)).to(self.device)).cpu().numpy()
        return x_tf

    def fit(self, inputs, labels=None):
        return self.encoder.fit(torch.from_numpy(inputs.astype(np.float32)).to(self.device))
    def minirocket(self):
        encoder = MiniRocketFeatures(c_in=self.n_channels,
                                     seq_len=self.seq_length,
                                     num_features=self.dim,
                                     random_state=self.seed,
                                     use_hdc=self.use_hdc,
                                     device=self.device).to(self.device)
        return encoder

class MiniRocketFeatures(nn.Module):
    """This is a Pytorch implementation of MiniRocket developed by Malcolm McLean and Ignacio Oguiza

    MiniRocket paper citation:
    @article{dempster_etal_2020,
      author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
      title   = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
      year    = {2020},
      journal = {arXiv:2012.08791}
    }
    Original paper: https://arxiv.org/abs/2012.08791
    Original code:  https://github.com/angus924/minirocket"""

    kernel_size, num_kernels, fitting = 9, 84, False

    def __init__(self, c_in, seq_len, num_features=10_000, max_dilations_per_kernel=32, random_state=None,
                 use_hdc=False, device='cpu'):
        super(MiniRocketFeatures, self).__init__()
        self.c_in, self.seq_len = c_in, seq_len
        self.num_features = num_features // self.num_kernels * self.num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.use_hdc = use_hdc
        self.device = device

        # Convolution
        indices = torch.combinations(torch.arange(self.kernel_size), 3).unsqueeze(1)
        kernels = (-torch.ones(self.num_kernels, 1, self.kernel_size)).scatter_(2, indices, 2)
        self.kernels = nn.Parameter(kernels.repeat(c_in, 1, 1), requires_grad=False)

        # Dilations & padding
        self._set_dilations(seq_len)

        # Channel combinations (multivariate)
        if c_in > 1:
            self._set_channel_combinations(c_in)
            # random channel IDs in in range {-1,1}
            self.channel_ids = torch.randint(0, 2, (c_in, num_features)).mul(2).sub(1).to(torch.float32)
            # random channel IDs in in range [-1,1]
            # self.channel_ids = torch.rand((c_in, num_features)).mul(2).sub(1).to(torch.float32)
        else:
            self.channel_ids = torch.ones((1, num_features)).to(torch.float32)

        # Bias
        for i in range(self.num_dilations):
            self.register_buffer(f'biases_{i}', torch.empty((self.num_kernels, self.num_features_per_dilation[i])))
        self.register_buffer('prefit', torch.BoolTensor([False]))

    def fit(self, X, chunksize=None):
        num_samples = X.shape[0]
        if chunksize is None:
            chunksize = min(num_samples, self.num_dilations * self.num_kernels)
        else:
            chunksize = min(num_samples, chunksize)
        np.random.seed(self.random_state)
        # idxs = np.random.choice(num_samples, chunksize, False)
        self.fitting = True
        if isinstance(X, np.ndarray):
            self(torch.from_numpy(X[:chunksize]).to(self.kernels.device))
        else:
            self(X[:chunksize].to(self.kernels.device))
        self.fitting = False

    def forward(self, x):
        _features = []
        self._index_counter = 0
        self.start_idx = 0
        self.counter = 0
        np.random.seed(self.random_state)
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):
            _padding1 = i % 2

            # Convolution
            C = F.conv1d(x, self.kernels, padding=padding, dilation=dilation, groups=self.c_in)
            if self.c_in > 1:  # multivariate
                C = C.reshape(x.shape[0], self.c_in, self.num_kernels, -1)
                channel_combination = getattr(self, f'channel_combinations_{i}')
                C = torch.mul(C, channel_combination)
                C = C.sum(1)

            # Bias
            if not self.prefit or self.fitting:
                num_features_this_dilation = self.num_features_per_dilation[i]
                bias_this_dilation = self._get_bias(C, num_features_this_dilation)
                setattr(self, f'biases_{i}', bias_this_dilation)
                if self.fitting:
                    if i < self.num_dilations - 1:
                        continue
                    else:
                        self.prefit = torch.BoolTensor([True])
                        return
                elif i == self.num_dilations - 1:
                    self.prefit = torch.BoolTensor([True])
            else:
                bias_this_dilation = getattr(self, f'biases_{i}')
            
            #if i < 3:
                #print(f'{i} : {bias_this_dilation[0,:10]}')
                #print(bias_this_dilation.shape)
            # Features
            _features.append(self._get_PPVs(C[:, _padding1::2], bias_this_dilation[_padding1::2]))
            _features.append(
                self._get_PPVs(C[:, 1 - _padding1::2, padding:-padding], bias_this_dilation[1 - _padding1::2]))
        return torch.cat(_features, dim=-1)

    def compute_poses(self, config=None):
        # create pose matrix for time encoding
        poses = create_pose_matrix(config['n_steps'], config['scale'], 
                                   config['HDC_dim'], seed=config['seed'])

        # load pose matrix to rocket transformer
        self.poses = torch.from_numpy(poses).to(self.device)


    def _get_PPVs(self, C, bias):
        """
        Function to compute the ppv based on the convolution output C and the bias plus poses P
        @param C: convolution output
        @param bias: bias
        @param P: poses
        @return: ppv
        """
        C = C.unsqueeze(-1)
        bias = bias.view(1, bias.shape[0], 1, bias.shape[1])
        if self.use_hdc:
            c = (C > bias)
            c = c.float().swapaxes(1,2).flatten(2)
            padding = self.poses.shape[0] - c.shape[1]
            pose = self.poses[padding//2:padding//2+c.shape[1], self._index_counter:self._index_counter + c.shape[2]]
            # bind pose to c
            ppv = ((c*2-1)*pose[None,:,:]).sum(1)
            # increase the counter for the next batch of c
            self._index_counter += c.shape[2]
        else:
            ppv = (C > bias).float().mean(2).flatten(1)
        return ppv

    def _set_dilations(self, input_length):
        num_features_per_kernel = self.num_features // self.num_kernels
        true_max_dilations_per_kernel = min(num_features_per_kernel, self.max_dilations_per_kernel)
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel
        max_exponent = np.log2((input_length - 1) / (9 - 1))
        dilations, num_features_per_dilation = \
            np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                      return_counts=True)
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)
        remainder = num_features_per_kernel - num_features_per_dilation.sum()
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)
        self.num_features_per_dilation = num_features_per_dilation
        self.num_dilations = len(dilations)
        self.dilations = dilations
        self.padding = []
        for i, dilation in enumerate(dilations):
            self.padding.append((((self.kernel_size - 1) * dilation) // 2))

    def _set_channel_combinations(self, num_channels):
        num_combinations = self.num_kernels * self.num_dilations
        max_num_channels = min(num_channels, 9)
        max_exponent_channels = np.log2(max_num_channels + 1)
        np.random.seed(self.random_state)
        num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(
            np.int32)
        channel_combinations = torch.zeros((1, num_channels, num_combinations, 1))
        for i in range(num_combinations):
            channel_combinations[:, np.random.choice(num_channels, num_channels_per_combination[i], False), i] = 1
        channel_combinations = torch.split(channel_combinations, self.num_kernels, 2)  # split by dilation
        for i, channel_combination in enumerate(channel_combinations):
            self.register_buffer(f'channel_combinations_{i}', channel_combination)  # per dilation

    def _get_quantiles(self, n, idx_start):
        return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(idx_start + 1, idx_start + n + 1)]).float()

    def _get_bias(self, C, num_features_this_dilation
                  ):
        # idxs = np.random.choice(C.shape[0], self.num_kernels)
        idxs = np.asarray([np.random.randint(C.shape[0]) for _ in range(self.num_kernels)])
        samples = C[idxs].diagonal().T
        # biases = torch.quantile(samples, self._get_quantiles(num_features_this_dilation, index_start).to(C.device), dim=1).T
        biases = torch.cat([torch.quantile(samples[i:i+1], self._get_quantiles(num_features_this_dilation,
                                                                               self.start_idx + num_features_this_dilation*i).to(C.device),
                                           dim=1).T for i in range(samples.shape[0])],dim=0)
        self.start_idx += num_features_this_dilation * samples.shape[0]
        return biases




