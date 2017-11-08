#!/usr/bin/env python
from headers import *
from state_class import *

class hierarchical():

    def __init__(self):

        self.num_epochs = 10
        self.save_every = 100
        self.num_images = 276
        self.current_parsing_index = 0
        self.parse_tree = [parse_tree_node()]
        self.paintwidth = -1
        self.minimum_width = -1
        self.images = []
        self.original_images = []
        self.true_labels = []
        self.image_size = -1
        self.intermittent_lambda = 0.
        self.suffix = []

#####################################################################################################
#####################################################################################################
    
    # REMEMBER, FIRST BUILDING THE NETWORK FOR RESIZED 256x256 image input.
    def build(self, sess, model_file=None):
        self.sess = sess

        # if vgg16_npy_path is None:
        path = sys.modules[self.__class__.__module__].__file__
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(path, "vgg16.npy")
        vgg16_npy_path = path
        # logging.info("Load npy file from '%s'.", vgg16_npy_path)

        # if not os.path.isfile(vgg16_npy_path):
        #   logging.error(("File '%s' not found. Download it from "
        #                  "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
        #                  "models/vgg16.npy"), vgg16_npy_path)
        #   sys.exit(1)
        debug = False
        random_init_fc8 = False
        VGG_MEAN = [ 175.5183833 ,  176.6830765 ,  192.35719172]
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4
        print("npy file loaded")

        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        train = True
        num_classes = 2
        # with tf.device('/device:GPU:0'):
        with tf.device('/gpu:0'):
            # self.input_image = tf.placeholder(tf.float32,shape=(256,256,3))
            self.input = tf.placeholder(tf.float32,shape=(1,256,256,3))
            # self.expanded_input = tf.expand_dims(self.input,0)
            # print("HEYYYYYY")
            # print(self.expanded_input)
            # red, green, blue = tf.split(self.expanded_input, 3, 3)
            red, green, blue = tf.split(self.input, 3, 3)
            self.bgr = tf.concat([blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2]], axis=3)
            self.conv1_1 = self._conv_layer(self.bgr, "conv1_1")
            self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

            self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

            self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)
        with tf.device('/gpu:1'):
            self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

            self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")

            self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)
        with tf.device('/gpu:2'):

            self.fc6 = self._fc_layer(self.pool5, "fc6")

            if train:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5)

            self.fc7 = self._fc_layer(self.fc6, "fc7")
            if train:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5)

            # For 256x256 images (for which we resize to), FC7 always takes shape 8x8x4096. 
            # There is no need to do the global spatial pooling to obtain constant shape. 
            # We can get away by just reshaping. 
            # When we switch to NOT resizing the original input, for an FCN,
            # we can do global spatial pooling (averaging) to feed into the rule / primitive streams. 

            # self.fc_input_shape = 8*8*4096    
            # self.policy_branch_fcinput = tf.reshape(self.fc7,[-1,self.fc_input_shape])

        with tf.device('/gpu:0'):
            # if random_init_fc8:
            #   self.score_fr = self._score_layer(self.fc7, "score_fr", num_classes)
            # else:

            # From 4096 to 128. 
            # self.feature_dimension = 64           
            self.num_filters = 64

            ###########################################################################################
            # Putting a fully-convolutional layer here. 
            self.W_fullyconv = tf.Variable(tf.truncated_normal([1,1,4096,64],stddev=0.1),name='W_fullyconv')
            self.b_fullyconv = tf.Variable(tf.constant(0.1,shape=[64]),name='b_fullyconv')

            self.finalconv_output = tf.nn.relu(tf.nn.conv2d(self.fc7,self.W_fullyconv,strides=[1,1,1,1],padding='SAME')+self.b_fullyconv,name='finalconv_output')

            ###########################################################################################
            self.score_fr = self._fc_layer(self.fc7, "score_fr", num_classes=num_classes, relu=False)

            self.fc_input_shape = 8*8*self.num_filters
            self.policy_branch_fcinput = tf.reshape(self.finalconv_output,[-1,self.fc_input_shape])

            self.pred = tf.argmax(self.score_fr, dimension=3)
            self.upscore2 = self._upscore_layer(self.score_fr, shape=tf.shape(self.pool4), num_classes=num_classes, debug=debug, name='upscore2', ksize=4, stride=2)

        with tf.device('/gpu:1'):
            self.score_pool4 = self._score_layer(self.pool4, "score_pool4", num_classes=num_classes)
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)
            self.upscore32 = self._upscore_layer(self.fuse_pool4, shape=tf.shape(self.bgr), num_classes=num_classes, debug=debug, name='upscore32', ksize=32, stride=16)
            self.pred_up = tf.argmax(self.upscore32, dimension=3)

            #################################################################################################
        self.previous_goal = npy.zeros(2)
        self.current_start = npy.zeros(2)

        # Defining the training optimizer. 
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train = self.optimizer.minimize(self.total_loss,name='Adam_Optimizer')

        # Writing graph and other summaries in tensorflow.
        self.writer = tf.summary.FileWriter('training',self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        #################################
        if model_file:
            # DEFINING CUSTOM LOADER:
            print("RESTORING MODEL FROM:", model_file)
            reader = tf.train.NewCheckpointReader(model_file)
            saved_shapes = reader.get_variable_to_shape_map()
            var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    curr_var = name2var[saved_var_name]
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)
            saver = tf.train.Saver(max_to_keep=None,var_list=restore_vars)
            saver.restore(self.sess, model_file)
        #################################

######################################################################################################
######################################################################################################

######################################################################################################
# UTILITY CODE FOR BUILDING AND LOADING VGG WEIGHTS.
######################################################################################################

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _fc_layer(self, bottom, name, num_classes=None, relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape, num_classes, name, debug, ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            # logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape, trainable=False)

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape, trainable=False)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        return tf.get_variable(name="biases", initializer=init, shape=shape, trainable=False)

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape, trainable=False)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer, trainable=False)

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init, shape=shape, trainable=False)

######################################################################################################
######################################################################################################

    def save_model(self, model_index, iteration_number=-1):
        if not(os.path.isdir("saved_models")):
            os.mkdir("saved_models")

        self.saver = tf.train.Saver(max_to_keep=None)           

        if not(iteration_number==-1):
            save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}_iter{1}.ckpt'.format(model_index,iteration_number))
        else:
            save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}.ckpt'.format(model_index))


    def define_plots(self):
        image_index = 0
        
        if self.plot:

            self.fig, self.ax = plt.subplots(1,5,sharey=True)
            self.fig.show()
            
            self.sc0 = self.ax[0].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
            self.sc0.set_clim([-1,1])
            self.ax[0].set_title("Parse Tree")
            self.ax[0].set_adjustable('box-forced')

            self.sc1 = self.ax[1].imshow(self.original_images[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
            # self.sc1.set_clim([-1,1])
            self.ax[1].set_title("Actual Image")
            self.ax[1].set_adjustable('box-forced')

            self.sc2 = self.ax[2].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
            self.sc2.set_clim([-1,1])
            self.ax[2].set_title("True Labels")
            self.ax[2].set_adjustable('box-forced')

            self.sc3 = self.ax[3].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
            self.sc3.set_clim([-1,1])
            self.ax[3].set_title("Predicted labels")
            self.ax[3].set_adjustable('box-forced')         

            self.sc4 = self.ax[4].imshow(self.true_labels[image_index],aspect='equal',cmap='jet',extent=[0,self.image_size,0,self.image_size],origin='lower')
            self.sc4.set_clim([-1,1])
            self.ax[4].set_title("Segmented Painted Image")
            self.ax[4].set_adjustable('box-forced')         

            self.fig.canvas.draw()
            plt.pause(0.1)  

    def meta_training(self,train=True):
        
        image_index = 0
        self.painted_image = -npy.ones((self.image_size,self.image_size))
        self.predicted_labels = npy.zeros((self.num_images,self.image_size, self.image_size))
        self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))
        # self.minimum_width = self.paintwidth
        
        if self.plot:
            print(self.plot)
            self.define_plots()

        self.to_train = train
        # For all epochs
        if not(train):
            self.num_epochs=1

        # For all epochs
        for e in range(self.num_epochs):

            image_list = npy.array(range(self.num_images))
            npy.random.shuffle(image_list)            

            for jx in range(self.num_images):

                # Image index to process.
                i = image_list[jx]
                
                self.vertical_grad = self.gradients[i,0]
                self.horizontal_grad = self.gradients[i,1]

                self.initialize_tree()
                self.construct_parse_tree(i)
                self.propagate_rewards()                
                print("#___________________________________________________________________________")
                print("Epoch:",e,"Training Image:",jx,"TOTAL REWARD:",self.parse_tree[0].reward)

                if train:
                    self.backprop(i)
                self.start_list = []
                self.goal_list = []
                
                if ((i%self.save_every)==0):
                    self.save_model(e,jx/self.save_every)

            if train:
                npy.save("parsed_{0}.npy".format(e),self.predicted_labels)
                npy.save("painted_images_{0}.npy".format(e),self.painted_images)
                self.save_model(e)
            else: 
                npy.save("validation_{0}.npy".format(self.suffix),self.predicted_labels)
                npy.save("validation_painted_{0}.npy".format(self.suffix),self.painted_images)
                
            self.predicted_labels = npy.zeros((self.num_images,self.image_size,self.image_size))
            self.painted_images = -npy.ones((self.num_images, self.image_size,self.image_size))

    def preprocess_images_labels(self):

    #   noise = 0.2*npy.random.rand(self.num_images,self.image_size,self.image_size)
    #   self.images[npy.where(self.images==2)]=-1
    #   self.true_labels[npy.where(self.true_labels==2)]=-1
    #   self.images += noise  

        # INSTEAD OF ADDING NOISE to the images, now we are going to normalize the images to -1 to 1 (float values).
        # Convert labels to -1 and 1 too.
        # self.true_labels = self.true_labels.astype(float)
        # self.true_labels /= self.true_labels.max()
        # self.true_labels -= 0.5
        # self.true_labels *= 2

        self.images = self.images.astype(float)
        self.grads = npy.zeros((self.num_images,2,self.image_size))
        self.grads[:,:,:-1] = copy.deepcopy(self.gradients)
        self.gradients = copy.deepcopy(self.grads)
        # self.image_means = self.images.mean(axis=(0,1,2))
        # self.images -= self.image_means
        # self.images_maxes = self.images.max(axis=(0,1,2))
        # self.images /= self.images_maxes


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def parse_arguments():

    parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
    parser.add_argument('--images',dest='images',type=str)
    parser.add_argument('--labels',dest='labels',type=str)
    parser.add_argument('--size',dest='size',type=int)
    parser.add_argument('--paintwidth',dest='paintwidth',type=int)
    parser.add_argument('--minwidth',dest='minwidth',type=int)
    parser.add_argument('--lambda',dest='inter_lambda',type=float)
    parser.add_argument('--model',dest='model',type=str)
    parser.add_argument('--suffix',dest='suffix',type=str)
    parser.add_argument('--gpu',dest='gpu')
    parser.add_argument('--plot',dest='plot',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--gradient',dest='gradients',type=str)

    return parser.parse_args()

def main(args):

    args = parse_arguments()
    print(args)

    # # Create a TensorFlow session with limits on GPU usage.
    # gpu_ops = tf.GPUOptions(visible_device_list=args.gpu)
    gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
    config = tf.ConfigProto(gpu_options=gpu_ops, allow_soft_placement = True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.90
    sess = tf.Session(config=config)

    hierarchical_model = hierarchical()

    hierarchical_model.images = npy.load(args.images)
    hierarchical_model.original_images = npy.load(args.images)
    hierarchical_model.true_labels = npy.load(args.labels)
    hierarchical_model.gradients = npy.load(args.gradients)

    hierarchical_model.image_size = args.size 
    hierarchical_model.preprocess_images_labels()

    hierarchical_model.paintwidth = args.paintwidth
    hierarchical_model.minimum_width = args.minwidth
    hierarchical_model.intermittent_lambda = args.inter_lambda

    hierarchical_model.plot = args.plot
    hierarchical_model.to_train = args.train
    
    # if hierarchical_model.to_train:
    hierarchical_model.suffix = args.suffix

    if args.model:
        # hierarchical_model.initialize_tensorflow_model(sess,args.model)
        hierarchical_model.build(sess,args.model)
    else:
        # hierarchical_model.initialize_tensorflow_model(sess)
        hierarchical_model.build(sess)

    hierarchical_model.meta_training(train=args.train)

if __name__ == '__main__':
    main(sys.argv)