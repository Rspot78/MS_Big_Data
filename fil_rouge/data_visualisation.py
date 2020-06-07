exec(open("imports.py").read())



def load_cifar10():
    """
    loads the cifar10 dataset and displays general information.

    parameters:
    - none

    return:
    - train_images: array, images of the cifar10 train set
    - train_labels: array, binary labels of the cifar10 train set
    - test_images: array, images of the cifar10 train set
    - test_labels: array, binary labels of the cifar10 test set
    - X_train: array, features for the train set
    - y_train: array, one-hot encoded labels of the cifar10 train set
    - X_test: array, features for the test set
    - y_test: array, one-hot encoded labels of the cifar10 test set
    """
        
    # import cifar10 data, split into train and test
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # normalise
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # print general info
    print()
    print("Succesfully loaded the cifar10 dataset.")
    print("There are " + str(train_images.shape[0]) + " images in the train set and " + str(test_images.shape[0]) + " images in the test set.")
    print("Each image is " + str(train_images.shape[1]) + " by " + str(train_images.shape[2]) + " by " + str(train_images.shape[3]) + ".")
    print("There are " + str(size(unique(train_labels))) + " different classes in the dataset.")
    print()
    # format data for training
    X_train = train_images
    X_test = test_images
    y_train = to_categorical(train_labels, 10)
    y_test = to_categorical(test_labels, 10)
    return train_images, train_labels, test_images, test_labels, X_train, y_train, X_test, y_test



def show_cifar10(images, labels, examples):
    """
    displays a few examples of each class of the cifar10 dataset.

    parameters:
    - images: array, images of the cifar10 train set
    - labels: array, binary labels of the cifar10 train set
    - examples: integer, number of examples of each class to display

    return:
    - none
    """
    
    # collect class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # fill arrays with images of the corresponding class
    display_images = zeros([10*examples, 32, 32, 3])
    for class_label in range(10):
        # find the positions of random images from the class
        pos = choice(where(labels==class_label)[0], examples, replace=False)
        # fill the matrix
        for example in range(examples):
            display_images[examples*class_label+example] = images[pos[example]]
    # generate figure
    figure_images = figure(figsize=(10,4*examples))
    for image_class in range(5):
        for example in range(examples):
            subplot(2*examples, 5 , example*5 + image_class + 1)
            xticks([])
            yticks([])
            grid(False)
            imshow(display_images[examples*image_class+example])
            xlabel(class_names[image_class])
    for image_class in range(5):
        for example in range(examples):
            subplot(2*examples, 5 , examples*5 + example*5 + image_class + 1)
            xticks([])
            yticks([])
            grid(False)
            imshow(display_images[examples*5 + examples*image_class+example])
            xlabel(class_names[5+image_class])    
    show()
    pass



def images_loss_accuracy(min_epoch, max_epoch, window, loss, val_loss, accuracy, val_accuracy, show_figure, path):
    """
    generate images (one per epoch) of loss and accuracy for the train and test sets.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video 
    - window: integer, number of epochs displayed in each graph (window <= epochs)
    - loss: list, record of loss values on the train set
    - val_loss: list, record of loss values on the validation set
    - accuracy: list, record of accuracy values on the train set
    - val_accuracy: list, record of accuracy values on the validation set
    - show_figure: boolean, determines whether figures should be displayed in the notebook
    - path: string, path to project folder
    
    return:
    - none
    """ 
    
    # create array of epochs
    epoch_array = np.arange(min_epoch, max_epoch+1)
    # convert records from tensors to numpy arrays
    loss_array = asarray(loss[min_epoch-1: max_epoch])
    val_loss_array = asarray(val_loss[min_epoch-1: max_epoch])
    accuracy_array = asarray(accuracy[min_epoch-1: max_epoch])
    val_accuracy_array = asarray(val_accuracy[min_epoch-1: max_epoch])
    # values to produce formatted plots
    max_loss = max(amax(loss_array), amax(val_loss_array))
    # loop over epochs
    for index, epoch in enumerate(epoch_array):
        # generate figure
        figure_loss_accuracy, ax_loss_accuracy = subplots(figsize=(12,8), facecolor=(0.85, 0.85, 0.85))
        suptitle("Loss and accuracy for train and validation samples", fontsize=20, fontweight='bold', color='black', y=1.05);
        
        # subplot for train loss
        subplot_1 = subplot(2, 2, 1)
        # if first epoch, just draw a dot
        if index == 0:
            epochs = epoch_array[0]
            losses = loss_array[0]
            subplot_1.plot(epochs, losses, linewidth=3, c=(0.7, 0.1, 0.1), marker = 'o')
            xlim([min_epoch-1, min_epoch+window])
        elif index < window:
            epochs = epoch_array[:index+1]
            losses = loss_array[:index+1]
            subplot_1.plot(epochs, losses, linewidth=3, c=(0.7, 0.1, 0.1))
            xlim([min_epoch-1, min_epoch+window])
        else:
            epochs = epoch_array[index-window:index+1]
            losses = loss_array[index-window:index+1]
            subplot_1.plot(epochs, losses, linewidth=3, c=(0.7, 0.1, 0.1))
            xlim([epoch-window, epoch+1])
        ylim([- 0.1 * max_loss, 1.1 * max_loss])
        title('Loss on train sample', fontsize=16)
        xlabel('Epochs')
        ylabel('Loss')
        subplot_1.grid(True)
        
        # subplot for validation loss
        subplot_2 = subplot(2, 2, 2)
        # if first epoch, just draw a dot
        if index==0:
            epochs = epoch_array[0]
            val_losses = val_loss_array[0]
            subplot_2.plot(epochs, val_losses, linewidth=3, c=(0.1, 0.7, 0.1), marker = 'o')
            xlim([min_epoch-1, min_epoch+window])
        elif index < window:
            epochs = epoch_array[:index+1]
            val_losses = val_loss_array[:index+1]
            subplot_2.plot(epochs, val_losses, linewidth=3, c=(0.1, 0.7, 0.1))
            xlim([min_epoch-1, min_epoch+window])
        else:
            epochs = epoch_array[index-window:index+1]
            val_losses = val_loss_array[index-window:index+1]
            subplot_2.plot(epochs, val_losses, linewidth=3, c=(0.1, 0.7, 0.1))
            xlim([epoch-window, epoch+1])
        ylim([- 0.1 * max_loss, 1.1 * max_loss])
        title('Loss on validation sample', fontsize=16)
        xlabel('Epochs')
        ylabel('Loss')
        subplot_2.grid(True)
             
        # subplot for train accuracy
        subplot_3 = subplot(2, 2, 3)
        # if first epoch, just draw a dot
        if index==0:
            epochs = epoch_array[0]
            accuracies = accuracy_array[0]
            subplot_3.plot(epochs, accuracies, linewidth=3, c=(0.7, 0.1, 0.1), marker = 'o')
            xlim([min_epoch-1, min_epoch+window])
        elif index < window:
            epochs = epoch_array[:index+1]
            accuracies = accuracy_array[:index+1]
            subplot_3.plot(epochs, accuracies, linewidth=3, c=(0.7, 0.1, 0.1))
            xlim([min_epoch-1, min_epoch+window])
        else:
            epochs = epoch_array[index-window:index+1]
            accuracies = accuracy_array[index-window:index+1]
            subplot_3.plot(epochs, accuracies, linewidth=3, c=(0.7, 0.1, 0.1))
            xlim([epoch-window, epoch+1]) 
        ylim([- 0.1, 1.1])
        title('Accuracy on train sample', fontsize=16)
        xlabel('Epochs')
        ylabel('Accuracy')
        subplot_3.grid(True)
    
        # subplot for validation accuracy
        subplot_4 = subplot(2, 2, 4)
        # if first epoch, just draw a dot
        if index==0:
            epochs = epoch_array[0]
            val_accuracies = val_accuracy_array[0]
            subplot_4.plot(epochs, val_accuracies, linewidth=3, c=(0.1, 0.7, 0.1), marker = 'o')
            xlim([min_epoch-1, min_epoch+window])
        elif index < window:
            epochs = epoch_array[:index+1]
            val_accuracies = val_accuracy_array[:index+1]
            subplot_4.plot(epochs, val_accuracies, linewidth=3, c=(0.1, 0.7, 0.1))
            xlim([min_epoch-1, min_epoch+window])
        else:
            epochs = epoch_array[index-window:index+1]
            val_accuracies = val_accuracy_array[index-window:index+1]
            subplot_4.plot(epochs, val_accuracies, linewidth=3, c=(0.1, 0.7, 0.1))
            xlim([epoch-window, epoch+1])
        ylim([- 0.1 , 1.1])
        title('Accuracy on validation sample', fontsize=16)
        xlabel('Epochs')
        ylabel('Accuracy')
        subplot_4.grid(True)

        # adjust space between subplots
        figure_loss_accuracy.tight_layout(h_pad=1.4)
        # show figure if activated, else close it to avoid display
        if show_figure:
            show(figure_loss_accuracy)
        else:
            close(figure_loss_accuracy)
        # save as image
        string = "figure_loss_accuracy.savefig('" + path + "/medias/loss_and_accuracy_epoch_" + str(epoch) +\
        ".png', bbox_inches='tight', facecolor=(0.85, 0.85, 0.85))"
        eval(string);
    pass



def mount_video(min_epoch, max_epoch, path, image_name, framerate, delete):
    """
    generate video from a set of images.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - path: string, path to project folder
    - image_name: string, name component shared by all the images in the set
    - framerate: number of images in each second of videos
    - delete: boolean, decides whether the input images should be deleted after mounting the video

    return:
    - none
    """ 
    
    # assemble all images in a single list
    image_list = []
    for epoch in range(min_epoch, max_epoch+1):
        # filename = path + "/" + image_name + "_epoch_" + str(epoch) + ".png"
        filename = path + "/medias/" + image_name + "_epoch_" + str(epoch) + ".png"
        image = imread(filename)
        height, width, layers = image.shape
        size = (width,height)
        image_list.append(image)
    # then assemble all images in a single video
    output = VideoWriter(filename = path + "/medias/" + image_name + "_epochs_" + str(min_epoch) + "_" + str(max_epoch) + ".avi",
                         fourcc=VideoWriter_fourcc(*'DIVX'), fps=framerate, frameSize=size)
    for image in range(len(image_list)):
        output.write(image_list[image])
    output.release()
    # finally, delete images if required
    if delete:
        for epoch in range(min_epoch, max_epoch+1):
            filename = path + "/medias/" + image_name + "_epoch_" + str(epoch) + ".png"
            if exists(filename):
                remove(filename)
    pass


def mount_class_video(min_epoch, max_epoch, path, image_name, framerate, delete):
    """
    generate video from a set of images, one for each of the cifar 10 dataset.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - path: string, path to project folder
    - image_name: string, name component shared by all the images in the set
    - framerate: number of images in each second of videos
    - delete: boolean, decides whether the input images should be deleted after mounting the video

    return:
    - none
    """ 

    # collect class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']    
    # loop over classes
    for class_name in class_names:
        # assemble all images in a single list
        image_list = []
        # loop over epochs
        for epoch in range(min_epoch, max_epoch+1):
            # generate video name
            filename = path + "/medias/" + image_name + "_class_" + class_name + "_epoch_" + str(epoch) + ".png"
            image = imread(filename)
            height, width, layers = image.shape
            size = (width,height)
            image_list.append(image)
        # then assemble all images in a single video
        output = VideoWriter(filename = path + "/medias/" + image_name + "_class_" + class_name + "_epochs_" + str(min_epoch) + "_" + str(max_epoch) + ".avi",
                             fourcc=VideoWriter_fourcc(*'DIVX'), fps=framerate, frameSize=size)
        for image in range(len(image_list)):
            output.write(image_list[image])
        output.release()
        # finally, delete images if required
        if delete:
            for epoch in range(min_epoch, max_epoch+1):
                filename = path + "/medias/" + image_name + "_class_" + class_name + "_epoch_" + str(epoch) + ".png"
                if exists(filename):
                    remove(filename)
    pass



def reorganise_layer(weight, viz_conv, viz_from):
    """
    determines plot dimensions and rearrange dense layers into pseudo filters
    
    parameters:
    - weight: array, weights to be reshaped
    - viz_conv: boolean, True if layer is convolutional, False if dense
    - viz_from: integer, first filter/neuron to be visualised
    
    return:
    - none
    """
    
    # if layer is convolutional
    if viz_conv == True:
        # keep only weights that are plotted
        weight = weight[:,:,:,viz_from-1:]
        # get kernel and filter dimensions
        kernel_size, kernel_dim, n_filters = weight.shape[0], weight.shape[2], weight.shape[3]
        # calculate the max number of plots to display on the image (max 25 weight length, 15 weight width)
        length_plots, width_plots = min(n_filters, 25 // kernel_size), min(kernel_dim, 15 // kernel_size)
        # keep only weights that are plotted
        weight_plot = weight[:,:,:width_plots,:length_plots]
    # if layer is dense
    else:
        # trim weight to start viz at viz_from
        weight = weight[viz_from-1:,:]
        # get layer dimensions
        weight_length, weight_width = weight.shape[0], weight.shape[1]
        for i in range (5,0,-1):
            if (weight_length % i == 0) and (weight_width % i == 0):
                kernel_size = i
                break
        width_plots, length_plots = weight_width // kernel_size, weight_length // kernel_size
        # keep only weights that are plotted
        weight_plot = np.zeros((kernel_size,kernel_size,width_plots,length_plots))
        for length_index in range(length_plots):
            for width_index in range(width_plots):
                weight_plot[:,:,width_index,length_index] = weight[length_index*kernel_size:(length_index+1)*kernel_size, width_index*kernel_size:(width_index+1)*kernel_size]
    return weight_plot, kernel_size, length_plots, width_plots



def images_weights(min_epoch, max_epoch, epochs_backwards, viz_conv, viz_from, weights, show_figure, path):
    """
    generate 3d plots (one per epoch) of weights for the train set.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video 
    - epochs_backwards: integer, number of epochs to look backwards when computing the growth rate with current epoch weights
    - viz_conv: boolean, if 'True' a convolution layer is used as input, otherwise layer is dense
    - viz_from: integer: indicates at which filter visualisation starts (for convolution), or at which activation (dense layer)
    - weights: list, record of weight values on the train set
    - show_figure: boolean, determines whether figures should be displayed in the notebook
    - path: string, path to project folder
    
    return:
    - none
    """
        
    # function to set color of each bar for epochs before epochs_backwards
    def set_color_weight_default(weight):
        # defaut color is light red if weight is positive
        if weight > 0:
            color = (1, 0.2, 0.2)
        # default color is light blue otherwise
        else:
            color = (0.2, 0.2, 1)
        return color
    
    # function to set color of each bar for epochs after epochs_backwards
    def set_color_weight(weight, growth):
        # cap growth rate to 2%
        growth = min(2, growth)
        # color is red if weight is positive
        if weight > 0:
            color = (1, 1-growth/2, 1-growth/2)
        # color is blue otherwise
        else:
            color = (1-growth/2, 1-growth/2, 1)
        return color
    
    # get plot dimensions
    __, kernel_size, length_plots, width_plots = reorganise_layer(weights[0], viz_conv, viz_from)
    
    # set figure style as classic (for nice bar display)
    style.use('classic') 
        
    # create mesh over x and y values
    xpos = np.arange(kernel_size)
    ypos = np.arange(kernel_size)
    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
    xpos = xposM.flatten('F')
    ypos = yposM.flatten('F')
    zpos = zeros(kernel_size ** 2)
    # width of bars on x and y axis
    dx=0.6
    dy=0.6
    
    # then loop over epochs
    for index, epoch in enumerate(np.arange(min_epoch, max_epoch+1)):    
        # recover weight array
        weight = weights[epoch-1]
        # process weight to adapt to plot dimensions
        weight, __, __, __ = reorganise_layer(weight, viz_conv, viz_from)
        
        # if index >= epochs_backwards: looking backwards is required for plots, recover weights from previous epochs and process
        if epoch > epochs_backwards:
            weight_backwards = weights[epoch-epochs_backwards-1]
            weight_backwards, __, __, __ = reorganise_layer(weight_backwards, viz_conv, viz_from)

        # generate figure
        fig, ax = subplots(figsize=(6,8), facecolor=(1, 1, 1))
        # overall title
        string = "suptitle('Weights: epoch " + str(epoch) + "', fontsize=20, fontweight='bold', color='black', y=1.02)"
        eval(string);
        # remove ticks and ticks labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # initiate counter (for subplot positionning)
        counter = 0
        # loop over filters
        for flter in range(length_plots):
            # loop over the three dimensions of the convolution kernel
            for dimension in range(width_plots):
                counter += 1
                # recover the matrix of weights (regular and absolute standardised values)
                weight_matrix = weight[:,:,dimension,flter]
                matrix_mean, matrix_std = mean(weight_matrix), std(weight_matrix)
                weight_matrix_normalised = absolute((weight_matrix-matrix_mean)/matrix_std)
                # height of each bar and its original sign
                dz = weight_matrix_normalised.flatten('F')
                dz_sign = weight_matrix.flatten('F')
                # color of each bar: if no backward epoch available, all bars are light red/blue by default
                if index < epochs_backwards and epoch <= epochs_backwards:
                    bar_colors = [set_color_weight_default(z) for z in dz_sign]
                # color of each bar: if index >= epochs_backwards, bars are red or blue with intensity given by growth rate
                else:
                    # recover weights from previous epochs
                    weight_backwards_matrix = weight_backwards[:,:,dimension,flter]
                    # establish absolute growth rate
                    weight_growth = absolute(100*(weight_matrix - weight_backwards_matrix) /
                                             weight_backwards_matrix).flatten('F')
                    bar_colors = [set_color_weight(z,g) for z, g in zip(dz_sign, weight_growth)]
                # create subplot
                ax = fig.add_subplot(length_plots,width_plots,counter, projection = '3d')
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color = bar_colors)
                ax.set_zlim(0, 4)
                # remove ticks, axis and grids
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.axis('off')
                ax.grid(False)
                # set background color (dark grey here)
                ax.set_facecolor((0.3, 0.3, 0.3))
                # slightly change camera angle for better view
                ax.view_init(elev=50)
                
        # set space between subplots
        fig.tight_layout(h_pad=-0.8, w_pad=-3)
        # show figure if activated, else close it to avoid display       
        if show_figure:
            show(fig)
        else:
            close(fig)        
        # save as image
        string = "fig.savefig('" + path + "/medias/weights_epoch_" + str(epoch) + ".png', bbox_inches='tight')"
        eval(string);
    pass



def images_gradients(min_epoch, max_epoch, viz_conv, viz_from, gradients, show_figure, path):
    """
    generate 3d plots (one per epoch) of gradients for the train set.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - viz_conv: boolean, True if layer is conv, False if layer is dense
    - viz_from: filter (for conv) or neuron (for dense) from which visualisation starts
    - gradients: list, record of gradients values on the train set
    - show_figure: boolean, determines whether figures should be displayed in the notebook
    - path: string, path to project folder

    return:
    - none
    """

    # function to set color of each bar
    def set_color_gradient(gradient):
        # cap on gradient size at 1
        gradient = min(1, gradient)
        # color is green
        color = (1-gradient, 1, 1-gradient)
        return color

    # get plot dimensions
    __, kernel_size, length_plots, width_plots = reorganise_layer(gradients[0], viz_conv, viz_from)
      
    # set figure style as classic (for nice bar display)
    style.use('classic')

    # create mesh over x and y values
    xpos = np.arange(kernel_size)
    ypos = np.arange(kernel_size)
    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
    xpos = xposM.flatten('F')
    ypos = yposM.flatten('F')
    zpos = zeros(kernel_size ** 2)
    # width of bars on x and y axis
    dx=0.6
    dy=0.6

    # then loop over epochs
    for index, epoch in enumerate(np.arange(min_epoch, max_epoch+1)):    
        # recover gradients
        gradient = gradients[epoch-1]
        # process gradient to adapt to plot dimensions
        gradient, __, __, __ = reorganise_layer(gradient, viz_conv, viz_from)
        
        # generate figure
        fig, ax = subplots(figsize=(6,8), facecolor=(1, 1, 1))
        # overall title
        string = "suptitle('Gradients: epoch " + str(epoch) + "', fontsize=20, fontweight='bold', color='black', y=1.02)"
        eval(string);
        # remove ticks and ticks labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # initiate counter (for subplot positionning)
        counter = 0

        # loop over filters
        for flter in range(length_plots):
            # loop over the three dimensions of the convolution kernel
            for dimension in range(width_plots):
                counter += 1
                # recover the matrix of gradients and absolute values
                gradient_matrix = gradient[:,:,dimension,flter]                
                gradient_matrix_absolute = absolute(gradient_matrix)
                # height of each bar and its original sign
                dz = gradient_matrix_absolute.flatten('F')
                # color of each bar
                bar_colors = [set_color_gradient(z) for z in dz]
                # create subplot
                ax = fig.add_subplot(length_plots,width_plots,counter, projection = '3d')
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color = bar_colors)
                ax.set_zlim(0, 1.4)
                # remove ticks, axis and grids
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.axis('off')
                ax.grid(False)
                # set background color (dark grey here)
                ax.set_facecolor((0.3, 0.3, 0.3))
                # slightly change camera angle for better view
                ax.view_init(elev=50)

        # set space between subplots
        fig.tight_layout(h_pad=-0.8, w_pad=-3)
        # show figure if activated, else close it to avoid display       
        if show_figure:
            show(fig)
        else:
            close(fig)        
        # save as image
        string = "fig.savefig('" + path + "/medias/gradients_epoch_" + str(epoch) + ".png', bbox_inches='tight')"
        eval(string);
    pass



def images_activations(min_epoch, max_epoch, viz_conv, viz_from, activations, show_figure, path):
    """
    generate plots (one per epoch) of activations for the train set.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - viz_conv: boolean, True if layer is conv, False if layer is dense
    - viz_from: filter (for conv) or neuron (for dense) from which visualisation starts
    - activations: list, record of activations values on the train set
    - show_figure: boolean, determines whether figures should be displayed in the notebook
    - path: string, path to project folder
    
    return:
    - none
    """
    
    # collect class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # loop over classes
    for class_name in range(10):
        # loop over epochs
        for index, epoch in enumerate(np.arange(min_epoch, max_epoch+1)):            
            # if convolution
            if viz_conv:
                # total number of filters
                tot_filters = activations[0][0].shape[2]
                # recover activation
                activation = activations[epoch-1][class_name][:,:,viz_from-1:min(tot_filters, viz_from+14)]
                # get image size in pixels:
                im_size = activation.shape[0]
                # initiate full image (as a set of 5*3 small images, vectorised as 15*1 image for now)
                temp = zeros((15*im_size, im_size))
                # fill image with activations
                for image in range(activation.shape[2]):
                    temp[image*im_size:(image+1)*im_size,:] = activation[:,:,image]
                # reshape
                full_image = zeros((5*im_size,3*im_size))
                full_image[:,0:im_size] = temp[0:5*im_size,:]
                full_image[:,im_size:2*im_size] = temp[5*im_size:10*im_size,:]                
                full_image[:,2*im_size:3*im_size] = temp[10*im_size:15*im_size,:]
            # if dense
            else:
                # recover activation
                activation = activations[epoch-1][class_name][:,viz_from-1:]
                # average over observations
                activation = mean(activation,0)
                # find plot height and width
                height = int(ceil(sqrt(5 / 3 * activation.shape[0])))
                width = int(ceil(activation.shape[0] / height))
                # reorganise activation in array of dimension height, width
                full_image = zeros((height * width))
                full_image[:activation.shape[0]] = activation
                full_image = reshape(full_image, (height,width))     
            # create image
            fig = figure(figsize=(6,8), facecolor=(0.75, 0.75, 0.75))
            # overall title
            string = "suptitle('Activations: epoch " + str(epoch) + "', fontsize=14, fontweight='bold', color='black', y=.93)"
            eval(string);
            # insert image
            img = imshow(full_image, interpolation='nearest')
            img.set_cmap('nipy_spectral')
            axis('off')
        
            # show figure if activated, else close it to avoid display       
            if show_figure:
                show(fig)
            else:
                close(fig)        
            # save as image
            string = "fig.savefig('" + path + "/medias/activations_class_" + class_names[class_name] + "_epoch_" + str(epoch) +\
                ".png', bbox_inches='tight', facecolor=(0.75, 0.75, 0.75))"
            eval(string);
    pass



def images_correlations(min_epoch, max_epoch, viz_conv, viz_from, activations, show_figure, path):
    """
    generate images (one per epoch) of correlation heatmaps for activations on the train set.

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - viz_conv: boolean, True if layer is conv, False if layer is dense
    - viz_from: filter (for conv) or neuron (for dense) from which visualisation starts
    - activationss: list, record of activations values on the train set
    - show_figure: boolean, determines whether figures should be displayed in the notebook
    - path: string, path to project folder
    
    return:
    - none
    """
    
    # collect class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # loop over classes
    for class_name in range(10):
        # loop over epochs
        for index, epoch in enumerate(np.arange(min_epoch, max_epoch+1)):            
            # if convolution
            if viz_conv:
                # recover activation
                activation = activations[epoch-1][class_name][:,:,viz_from-1:]
                # get image size in pixels and number of filters:
                im_size, n_filters = activation.shape[0], activation.shape[2]
                # convert to 2d matrix
                temp = zeros((im_size**2,n_filters))
                for i in range(n_filters):
                    temp[:,i] = reshape(activation[:,:,i],(-1))
                # compute correlation matrix
                correlation_matrix = corrcoef(temp.T)
            # if dense
            else:
                # recover activation
                activation = activations[epoch-1][class_name][:,viz_from-1:]
                # compute correlation matrix
                correlation_matrix = corrcoef(activation.T)
            # create image
            # rcParams['savefig.facecolor'] = (.75,.75,.75)
            fig, ax = subplots(figsize=(6,8), facecolor=(.75,.75,.75))
            # overall title
            string = "suptitle('Correlations: epoch " + str(epoch) + "', fontsize=14, fontweight='bold', color='black', y=.93)"
            eval(string);
            # insert image
            img = ax.imshow(correlation_matrix, interpolation='nearest')
            img.set_cmap('YlOrRd')
            img.set_clim(-1, 1)
            fig.colorbar(img, ax=ax)
            axis('off')
            # show figure if activated, else close it to avoid display       
            if show_figure:
                show(fig)
            else:
                close(fig)        
            # save as image
            string = "fig.savefig('" + path + "/medias/correlations_class_" + class_names[class_name] + "_epoch_" + str(epoch) +\
                ".png', bbox_inches='tight', facecolor=(0.75, 0.75, 0.75))"
            eval(string);
    pass



def assemble_images(min_epoch, max_epoch, delete, path):
    """
    assembles two images, side by side (for weights and gradients).

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - delete: boolean, decides whether the input images should be deleted after mounting the video
    - path: string, path to project folder

    return:
    - none
    """     
    
    # create array of epochs
    epoch_array = np.arange(min_epoch, max_epoch+1)
    # then loop over epochs
    for index, epoch in enumerate(epoch_array): 
        string1 = path + "/medias/weights_epoch_" + str(epoch) + ".png"
        string2 = path + "/medias/gradients_epoch_" + str(epoch) + ".png"
        string3 = path + "/medias/weights_and_gradients_epoch_" + str(epoch) + ".png"
        images = [Image.open(x) for x in [string1, string2]]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        merged_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for image in images:
            merged_image.paste(image, (x_offset,0))
            x_offset += image.size[0]
        merged_image.save(string3)
        # delete individual pictures if required
        if delete:
            if exists(string1):
                remove(string1)
            if exists(string2):
                remove(string2)



def assemble_class_images(min_epoch, max_epoch, delete, path):
    """
    assembles two images side by side, for each class (for activations and correlations).

    parameters:
    - min_epoch: integer, epoch of the first image of the video
    - max_epoch: integer, epoch of the final image of the video
    - delete: boolean, decides whether the input images should be deleted after mounting the video
    - path: string, path to project folder
    
    return:
    - none
    """     
    
    # collect class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # create array of epochs
    epoch_array = np.arange(min_epoch, max_epoch+1)
    # loop over classes
    for class_name in class_names:
        # then loop over epochs
        for index, epoch in enumerate(epoch_array): 
            string1 = path + "/medias/activations_class_" + class_name + "_epoch_" + str(epoch) + ".png"
            string2 = path + "/medias/correlations_class_" + class_name + "_epoch_" + str(epoch) + ".png"
            string3 = path + "/medias/activations_and_correlations_class_" + class_name + "_epoch_" + str(epoch) + ".png"            
            images = [Image.open(x) for x in [string1, string2]]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            merged_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for image in images:
                merged_image.paste(image, (x_offset,0))
                x_offset += image.size[0]
            merged_image.save(string3)
            # delete individual pictures if required
            if delete:
                if exists(string1):
                    remove(string1)
                if exists(string2):
                    remove(string2)


                
def loss_accuracy_video(epoch, frequency, window, loss, val_loss, accuracy, val_accuracy, path, fps, show, delete):
    """
    creates loss and accuracy images, then mount them in a video.

    parameters:
    - epoch: integer, current epoch of the training
    - frequency: integer, frequency of epochs at which a video must be mounted
    - window: integer, number of epochs displayed in each graph (window <= frequency)
    - loss: list, record of loss values on the train set
    - val_loss: list, record of loss values on the validation set
    - accuracy: list, record of accuracy values on the train set
    - val_accuracy: list, record of accuracy values on the validation set
    - path: string, path to the images
    - fps: number of images in each second of videos
    - show: boolean, determines whether figures should be displayed in the notebook
    - delete: boolean, decides whether the input images should be deleted after mounting the video

    return:
    - none
    """
    
    # in the first place, check whether a video must be mounted
    if epoch % frequency == 0:
        # define first and last epoch in the video
        first_epoch, last_epoch = epoch - frequency + 1, epoch
        # create the loss and accuracy images
        images_loss_accuracy(min_epoch=first_epoch, max_epoch=last_epoch, window=window, loss=loss, val_loss=val_loss,
                             accuracy=accuracy, val_accuracy=val_accuracy, show_figure=show, path=path)
        # then mount a video from the images
        name = "loss_and_accuracy"
        mount_video(min_epoch=first_epoch, max_epoch=last_epoch, path=path, image_name=name, framerate=fps, delete=delete)
    pass



def weight_gradient_video(epoch, frequency, backwards, viz_conv, viz_from, weights, gradients, path, fps, show, delete):
    """
    creates weight and gradient images, then mount them in a video.

    parameters:
    - epoch: integer, current epoch of the training
    - frequency: integer, frequency of epochs at which a video must be mounted
    - backwards: integer, number of epochs to look backwards when computing growth rates (backwards <= frequency)
    - weights: list, record of weights values on the train set
    - gradients: list, record of gradients values on the validation set
    - path: string, path to the images
    - fps: number of images in each second of videos
    - show: boolean, determines whether figures should be displayed in the notebook
    - delete: boolean, decides whether the input images should be deleted after mounting the video

    return:
    - none
    """
    
    # in the first place, check whether a video must be mounted
    if epoch % frequency == 0:
        # define first and last epoch in the video
        first_epoch, last_epoch = epoch - frequency + 1, epoch
        # create the weight and gradient images, then merge them
        images_weights(min_epoch=first_epoch, max_epoch=last_epoch, epochs_backwards=backwards, 
                       viz_conv=viz_conv, viz_from=viz_from, weights=weights, show_figure=show, path=path)
        images_gradients(min_epoch=first_epoch, max_epoch=last_epoch, viz_conv=viz_conv, 
                         viz_from=viz_from, gradients=gradients, show_figure=show, path=path)
        assemble_images(min_epoch=first_epoch, max_epoch=last_epoch, delete=delete, path=path)
        # then mount a video from the images
        name = "weights_and_gradients"
        mount_video(min_epoch=first_epoch, max_epoch=last_epoch, path=path, image_name=name, framerate=fps, delete=delete)
    pass



def activation_correlation_video(epoch, frequency, viz_conv, viz_from, activations, path, fps, show, delete):
    """
    creates activation and correlation images, then mount them in a video (for each class).

    parameters:
    - epoch: integer, current epoch of the training
    - frequency: integer, frequency of epochs at which a video must be mounted
    - viz_conv: boolean, True if layer is conv, False if layer is dense
    - viz_from: filter (for conv) or neuron (for dense) from which visualisation starts
    - activations: list, record of activation values on the train set
    - path: string, path to the images
    - fps: number of images in each second of videos
    - show: boolean, determines whether figures should be displayed in the notebook
    - delete: boolean, decides whether the input images should be deleted after mounting the video

    return:
    - none
    """
    
    # in the first place, check whether a video must be mounted
    if epoch % frequency == 0:
        # define first and last epoch in the video
        first_epoch, last_epoch = epoch - frequency + 1, epoch
        # create the activation and correlation images, then merge them
        
        images_activations(min_epoch=first_epoch, max_epoch=last_epoch, viz_conv=viz_conv,
                           viz_from=viz_from, activations=activations, show_figure=show, path=path)
        images_correlations(min_epoch=first_epoch, max_epoch=last_epoch, viz_conv=viz_conv,
                           viz_from=viz_from, activations=activations, show_figure=show, path=path)
        assemble_class_images(min_epoch=first_epoch, max_epoch=last_epoch, delete=delete, path=path)
        # then mount a video from the images
        name = "activations_and_correlations"
        mount_class_video(min_epoch=first_epoch, max_epoch=last_epoch, path=path, image_name=name, framerate=fps, delete=delete)
    pass    





