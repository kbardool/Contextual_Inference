from keras import losses

"""
As one may read here deserialize_keras_object is responsible for 
creation of a keras object from:

- configuration dictionary - if one is available,
- name identifier if a provided one was available.
  To understand second point image the following definition:

    model.add(Activation("sigmoid"))
    What you provide to Activation constructor is a string, not a keras object. 
    In order to make this work - deserialize_keras_object looks up defined names 
    and check if an object called sigmoid is defined and instantiates it.
"""    
    
    
    
    
    
def _weighted_masked_objective(fn):
    """Adds support for masking and sample-weighting to an objective function.

    It transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.

    # Arguments
        fn: The objective function to wrap,
            with signature `fn(y_true, y_pred)`.

    # Returns
        A function with signature `fn(y_true, y_pred, weights, mask)`.
    """
    if fn is None:
        return None

    def weighted(y_true, y_pred, weights, mask=None):
        """Wrapper function.

        # Arguments
            y_true: `y_true` argument of `fn`.
            y_pred: `y_pred` argument of `fn`.
            weights: Weights tensor.
            mask: Mask tensor.

        # Returns
            Scalar tensor.
        """
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in Theano
            mask = K.cast(mask, K.floatx())
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= K.mean(mask)

        # apply sample weighting
        if weights is not None:
            # reduce score_array to same ndim as weight array
            ndim = K.ndim(score_array)
            weight_ndim = K.ndim(weights)
            score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))
            score_array *= weights
            score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return K.mean(score_array)
    return weighted

    
def loss_functions(model, loss):
    # Prepare loss functions.
    print(model)
    print(loss)
    if isinstance(loss, dict):
        for name in loss:
            print(name)
            if name not in model.output_names:
                raise ValueError('Unknown entry in loss '
                                 'dictionary: "' + name + '". '
                                 'Only expected the following keys: ' +
                                 str(model.output_names))
        
        loss_functions = []
        for name in model.output_names:
            if name not in loss:
                warnings.warn('Output "' + name +
                              '" missing from loss dictionary. '
                              'We assume this was done on purpose, '
                              'and we will not be expecting '
                              'any data to be passed to "' + name +
                              '" during training.', stacklevel=2)
            loss_functions.append(losses.get(loss.get(name)))
            
    elif isinstance(loss, list):
        if len(loss) != len(model.outputs):
            print('Warning : When passing a list as loss, '
                             'it should have one entry per model outputs. '
                             'The model has ' + str(len(model.outputs)) +
                             ' outputs, but you passed loss=' +
                             str(loss))
        loss_functions = []
        
        for l in loss:
            i = losses.get(l)
            print('loss : ' , l, ' losses.get(l): ',i)
            loss_functions.append(i)
        # loss_functions = [losses.get(l) for l in loss]

    else:
        loss_function = losses.get(loss)
        loss_functions = [loss_function for _ in range(len(model.outputs))]
    model.loss_functions = loss_functions
    weighted_losses = [_weighted_masked_objective(fn) for fn in loss_functions]
    skip_target_indices = []
    skip_target_weighing_indices = []
    model._feed_outputs = []
    model._feed_output_names = []
    model._feed_output_shapes = []
    model._feed_loss_fns = []
    for i in range(len(weighted_losses)):
        if weighted_losses[i] is None:
            skip_target_indices.append(i)
            skip_target_weighing_indices.append(i)