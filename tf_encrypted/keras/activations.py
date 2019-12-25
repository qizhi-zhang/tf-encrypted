# pylint: disable=inconsistent-return-statements
"""Provide activation functions"""
from tf_encrypted import get_protocol


def relu(x):
  """Computes relu of x element-wise"""
  return get_protocol().relu(x)

def sigmoid(x):
  """Computes sigmoid of x element-wise"""
  return get_protocol().sigmoid(x)

def sigmoid_deriv(y, d_y):
  """Computes derive sigmoid of y"""
  return d_y * y * (get_protocol().negative(y) + 1)

def tanh(x):
  """Computes tanh of x element-wise"""
  return get_protocol().tanh(x)

def linear(x):
  return x

#---------------qizhi.zqz----------------------------------------
def relu_deriv(y, d_y):
    """Computes derive relu of x element-wise"""
    return d_y * get_protocol().is_negative(get_protocol().negative(y))

#--------------------------------qizhi.zqz----------------------


def get(identifier):
  """get the activation function"""
  if identifier is None:
    return linear
  if callable(identifier):
    return identifier
  if isinstance(identifier, str):
    activations = {"relu": relu,
                   "sigmoid": sigmoid,
                   "tanh": tanh,
                   "linear": linear}
    return activations[identifier]

def get_deriv(identifier):
  """get the activation derivative function"""
  if identifier is None:
    return linear
  if callable(identifier):
    raise NotImplementedError('During training, please use a string '
                              '(e.g "relu") to specify the activation '
                              'function instead of calling directly '
                              'the activation function.')
  if isinstance(identifier, str):
    #activations = {"sigmoid": sigmoid_deriv}                      # - by qizhi.zqz
    activations = {"sigmoid": sigmoid_deriv, "relu": relu_deriv}   # + by qizhi.zqz
    if identifier not in activations.keys():
      raise NotImplementedError('Activation function {} not yet implemented '
                                'during training'.format(identifier))
    return activations[identifier]

  raise ValueError('Could not interpret '
                   'activation function identifier:', identifier)
