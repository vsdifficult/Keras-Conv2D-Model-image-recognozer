# Keras-Conv2D-Model-image-recognozer
This code create model, which reconize objects on image
# About this layer (Keras documentation) 
2D convolution layer.

This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

Arguments

    filters: int, the dimension of the output space (the number of filters in the convolution).
    kernel_size: int or tuple/list of 2 integer, specifying the size of the convolution window.
    strides: int or tuple/list of 2 integer, specifying the stride length of the convolution. strides > 1 is incompatible with dilation_rate > 1.
    padding: string, either "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input.
    data_format: string, either "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch_size, height, width, channels) while "channels_first" corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    dilation_rate: int or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
    groups: A positive int specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters // groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.
    activation: Activation function. If None, no activation is applied.
    use_bias: bool, if True, bias will be added to the output.
    kernel_initializer: Initializer for the convolution kernel. If None, the default initializer ("glorot_uniform") will be used.
    bias_initializer: Initializer for the bias vector. If None, the default initializer ("zeros") will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the kernel after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an Optimizer.

Input shape

    If data_format="channels_last": A 4D tensor with shape: (batch_size, height, width, channels)
    If data_format="channels_first": A 4D tensor with shape: (batch_size, channels, height, width)

Output shape

    If data_format="channels_last": A 4D tensor with shape: (batch_size, new_height, new_width, filters)
    If data_format="channels_first": A 4D tensor with shape: (batch_size, filters, new_height, new_width)

Returns

A 4D tensor representing activation(conv2d(inputs, kernel) + bias).
