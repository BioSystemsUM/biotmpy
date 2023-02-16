from keras.layers import Layer
from keras import initializers
import keras.backend as K


class AttentionLayer(Layer):
    """
    Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)
    - Yang et. al.
    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Theano backend
    """

    def __init__(self, attention_dim=100, return_coefficients=False, **kwargs):
        """
        :param attention_dim: dimension of the attention space
        :param return_coefficients: if True, the attention coefficients are returned
        :param kwargs: additional arguments for the Layer class (e.g. name). See Layer documentation for more details.
        """
        # Initializer
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')  # initializes values with uniform distribution
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        """
        Returns the config of the layer.

        :return: config of the layer
        """
        config = super().get_config().copy()
        config.update({
            'supports_masking': self.supports_masking,
            'return_coefficients': self.return_coefficients,
            'init': self.init,
            'attention_dim': self.attention_dim,
        })
        return config

    def build(self, input_shape):
        """
        Builds the layer.

        :param input_shape: shape of the input

        :return: None
        """
        # Builds all weights
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim,)), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights.extend([self.W, self.b, self.u])

        super(AttentionLayer, self).build(input_shape)

    @staticmethod
    def compute_mask(input, input_mask=None):
        return None

    def call(self, hit, mask=None):
        """
        Performs the actual computation of the layer.

        :param hit: input tensor
        :param mask: mask tensor

        :return: output tensor
        """
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W), self.b)
        uit = K.tanh(uit)

        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        :param input_shape: shape of the input

        :return: shape of the output
        """
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]
