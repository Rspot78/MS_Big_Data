exec(open("imports.py").read())



def create_batch(labels, X, y, images):
    """
    creates a batch of representative images for the 10 classes of the cifar10 dataset.

    parameters:
    - labels: array, binary labels of the cifar10 train set
    - X: array, features for the train set
    - y: array, one-hot encoded labels of the cifar10 train set
    - images: integer, number of images from each class to include in the batch

    return:
    - X_gradient: tensor, images from which gradient must be calculated
    - y_gradient: tensor, one-hot encoded labels from which gradient must be calculated
    """
    
    # create empty matrices of the size of the final batch
    X_gradient = empty([10*images, 32, 32, 3])
    y_gradient = empty([10*images, 10])
    # fill the matrices with images from each class
    for class_label in range(10):
        # find the positions of random images in the class
        pos = choice(where(labels==class_label)[0], images, replace=False)
        # fill the matrix with these images
        for image in range(images):
            X_gradient[images*class_label+image] = X[pos[image]]
            y_gradient[images*class_label+image] = y[pos[image]]
    # convert batch to tensor
    X_gradient = tf.convert_to_tensor(X_gradient)
    y_gradient = tf.convert_to_tensor(y_gradient)
    return X_gradient, y_gradient



def create_class_batch(labels, X, images):
    """
    creates a batch of representative images for each of the 10 classes of the cifar10 dataset, separately.

    parameters:
    - labels: array, binary labels of the cifar10 train set
    - X: array, features for the train set
    - images: integer, number of images from each class to include in the batch

    return:
    - X_class: list of tensors, each tensor containing images from one class
    """
    
    X_class = []
    # fill the matrices with images from each class, order being: 
    # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    for class_label in range(10):
        # create empty matrix of the size of the final batch
        X_one_class = empty([images, 32, 32, 3])
        # find the positions of random images in the class
        pos = choice(where(labels==class_label)[0], images, replace=False)
        # fill the matrix with these images
        for image in range(images):
            X_one_class[image] = X[pos[image]]
        # append to list
        X_class.append(X_one_class)
    return X_class



def get_gradient(X,y,layer,model):
    """
    obtain the gradient of the loss w.r.t to the weights for a given batch of data.

    parameters:
    - X: tensor, batch of images from which gradient must be calculated
    - y: tensor, one-hot encoded labels from which gradient must be calculated
    - layer: int, layer to be visualised
    - model: keras model, the CNN model for which the grandient must be calculated

    return:
    - gradient: numpy array, gradient of the loss with respect to the weights
    """
    
    loss_object = CategoricalCrossentropy()
    with GradientTape() as tape:
        prediction = model(X, False)
        loss = loss_object(y, prediction)
    gradient = tape.gradient(loss, model.trainable_variables)[layer]
    gradient = tfeval(gradient)
    return gradient



def get_activation(X, layer, model, conv):
    """
    obtain the activations for a given batch of data
    
    parameters:
    - X: list of tensors, each element being a batch of images from which activations must be calculated
    - layer: int, layer to be visualised
    - model: keras model, the CNN model for which the grandient must be calculated
    - conv: boolean, True if layer to be visualised is convolution, False if dense

    return:
    - activations: list of numpy arrays, activation for the 
    """
    
    activations = []
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [tffunction([inp], [out]) for out in outputs]    # evaluation functions    
    # loop over the 'all' class and 10 individual classes
    for class_label in range(10):
        # extract batch for a given class
        X_one_class = X[class_label]
        # obtain activations for this class
        layer_outs = [func([X_one_class]) for func in functors]
        # extract activations for the layer of interest
        activation = layer_outs[layer][0]
        # if layer is convolutional, average over batch
        if conv:
            activation = mean(activation,0)
        # append to list
        activations.append(activation)
    return activations
        
            
  
    
  