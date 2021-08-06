#定义判别器
import functools
import paddle
import paddle.nn as nn

class _SpectralNorm(nn.SpectralNorm):
    def __init__(self,
                 weight_shape,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(_SpectralNorm, self).__init__(weight_shape, dim, power_iters, eps,
                                            dtype)

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        _power_iters = self._power_iters if self.training else 0
        self._helper.append_op(type="spectral_norm",
                               inputs=inputs,
                               outputs={
                                   "Out": out,
                               },
                               attrs={
                                   "dim": self._dim,
                                   "power_iters": _power_iters,
                                   "eps": self._eps,
                               })

        return out


class Spectralnorm(paddle.nn.Layer):
    def __init__(self, layer, dim=None, power_iters=1, eps=1e-12, dtype='float32'):
        super(Spectralnorm, self).__init__()

        if dim is None: # conv: dim = 1, Linear: dim = 0
            if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv3D,
                    nn.Conv1DTranspose, nn.Conv2DTranspose, nn.Conv3DTranspose)):
                dim = 1
            else:
                dim = 0
        if layer.training == False: # don't do power iterations when infering
            power_iters = 0

        self.spectral_norm = _SpectralNorm(layer.weight.shape, dim, power_iters,
                                           eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape,
                                                 dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out
def build_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2D)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2D)
    elif norm_type == 'spectral':
        norm_layer = functools.partial(Spectralnorm)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
class NLayerDiscriminator(nn.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        spectral_norm = build_norm_layer('spectral')
        instance_norm = build_norm_layer('instance')

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2D(input_nc, ndf, kernel_size=kw,stride=2, padding=padw),True),
            nn.LeakyReLU(0.2,True)
        ]

        nf_mult = 1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2D(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=kw,stride=2,padding=padw,bias_attr=None),True),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers,8)
        sequence += [
            spectral_norm(nn.Conv2D(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=kw,stride=2,padding=padw,bias_attr=None),True),
            nn.LeakyReLU(0.2,True),
        ]

        sequence += [spectral_norm(nn.Conv2D(ndf*nf_mult,1,kernel_size=kw,stride=2,padding=padw,bias_attr=None),True)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
