# Author: Yihao Fang, M.Eng., Ph.D. candidate, Department of Computing and Software, McMaster University
# Created Date: Nov 26, 2017
# Updated Date: Dec 14, 2017

import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn.decomposition import PCA
from pandas import DataFrame
from tensorflow.python.ops import variable_scope as vs

'''
#pearson = [0.602936873,	0,	0.56541025,	0.564380951,	0.256654264,	0.744555295,	0.521587062,	0.1,	0.461217069,	0.654667636,	1,	0.476111645,	0.62756085]

#kendall = [0.391246564,	0,	0.401648375,	0.376735656,	0.108272683,	0.573004783,	0.35100123,	0.1,	0.376647234,	0.434394886,	1,	0.339624692,	0.410482467]

#spearman = [0.391464515,	0,	0.401858032,	0.375793157,	0.108293134,	0.573062975,	0.350945774,	0.1,	0.3768338,	0.434562409,	1,	0.339640727,	0.41037937]

pearson = [0.251355434,	1,	0.234039644,	0.233564698,	0.108428644,	0.31670197,	0.213818451,	0.180713121,	0.185962117,	0.275225395,	0.434571004,	0.192834874,	0.262717597]
kendall = [0.23963697,	1,	0.245286274,	0.231755984,	0.114048326,	0.338351261,	0.21777942,	0.118541284,	0.231707962,	0.263071157,	0.570255655,	0.211600734,	0.250084138]
spearman = [0.239641668,	1,	0.24528239,	0.23113658,	0.114039817,	0.338197959,	0.217651521,	0.118540629,	0.231701353,	0.263031561,	0.569903281,	0.211516097,	0.24990705]
'''
tf.set_random_seed(0)

batch_size = 64
num_features = 13
num_main_hid_units = 256
num_branch_hid_units = 256
num_epoches = 2000
num_main_hid_layers = 2
nn = 'gbnn'
corr_index = 0


# pca analysis for test dataset by eigen vectors, which are generated from training dataset
def pca_by_eigenvectors(components, X_test):
    """
    PCA analysis for test dataset
    === parameters ===
    'components':pca.components_ (eigenvectors)
                  from training dataset, shape of (n_components, n_features)
    'X_test': input of test dataset, shape of (batch, n_features)
    === return ===
    X_test after pca, shape of (batch, n_components)
    """
    if components.shape[1] != len(X_test[0]):
        return False
    return np.dot(components, X_test.transpose()).transpose()


def load_txt(file_names):
    examples = []
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                l_num_arr = [float(n) for n in line.split()]
                examples.append(l_num_arr)

    return examples


def create_dataset(examples, pca_components=None, mean=None, std=None, corr=None):
    '''
    non_merge_examples = []
    merge_examples = []
    for example in examples:
        l = [float(n) for n in example.split()]
        if l[34] == 0:
            non_merge_examples.append(example)
        else:
            merge_examples.append(example)

    non_merge_examples = non_merge_examples[int(len(non_merge_examples)*start): int(len(non_merge_examples)*end)]
    merge_examples = merge_examples[int(len(merge_examples)*start): int(len(merge_examples)*end)]

    examples = []
    for i in range(len(non_merge_examples)):
        examples.append(non_merge_examples[i])
        examples.append(merge_examples[i])
    #examples.extend(non_merge_examples)
    #examples.extend(merge_examples)
    '''
    x = []
    y = []
    for l in examples:
        # 3: left distance, 6: self velocity
        x.append(np.concatenate((l[3:4], l[5:8], l[11:14], l[15:18], l[27:30], l[31:34])))
        y.append(l[34])

    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    x_scaled = (x - mean) / std

    if pca_components is not None:
        x = pca_by_eigenvectors(pca_components, x_scaled)
    else:
        pca = PCA(n_components=0.95)
        x = pca.fit_transform(x_scaled)
        pca_components = pca.components_

    # examples = np.concatenate((x, np.reshape(y, (-1, 1))), axis = 1)
    x_n_y = []
    for i, r in enumerate(x):
        x_n_y.append(np.concatenate((r, [y[i]])))

    if corr is None:
        scaled_df14 = DataFrame(x_n_y)
        # looking for correlations
        corr_matrix_pearson = scaled_df14.corr(method='pearson')
        corr_matrix_kendall = scaled_df14.corr(method='kendall')
        corr_matrix_spearman = scaled_df14.corr(method='spearman')
        '''
        corr_matrix_pearson_sorted = corr_matrix_pearson[13].sort_values(ascending=False)
        corr_matrix_kendall_sorted = corr_matrix_kendall[13].sort_values(ascending=False)
        corr_matrix_spearman_sorted = corr_matrix_spearman[13].sort_values(ascending=False)
        print("pearson: ", corr_matrix_pearson_sorted)
        print("kendall: ", corr_matrix_kendall_sorted)
        print("spearman: ", corr_matrix_spearman_sorted)
        '''
        corr_pearson = corr_matrix_pearson[13].as_matrix()[:13]
        corr_kendall = corr_matrix_kendall[13].as_matrix()[:13]
        corr_spearman = corr_matrix_spearman[13].as_matrix()[:13]
        print("pearson: ", corr_pearson)
        print("kendall: ", corr_kendall)
        print("spearman: ", corr_spearman)
        corr_pearson_norm = np.divide(np.absolute(corr_pearson), np.absolute(corr_pearson).max()) * 0.9 + 0.1
        corr_kendall_norm = np.divide(np.absolute(corr_kendall), np.absolute(corr_kendall).max()) * 0.9 + 0.1
        corr_spearman_norm = np.divide(np.absolute(corr_spearman), np.absolute(corr_spearman).max()) * 0.9 + 0.1
        print("pearson norm: ", corr_pearson_norm)
        print("kendall norm: ", corr_kendall_norm)
        print("spearman norm: ", corr_spearman_norm)

        corr = (corr_pearson_norm, corr_kendall_norm, corr_spearman_norm)

    # shuffle(x_n_y)
    return x_n_y, pca_components, mean, std, corr


def create_batches(examples):
    x_batches = []
    y_batches = []

    for i, l in enumerate(examples):
        if i % batch_size == 0:
            if i > 0:
                x_batches.append(x)
                y_batches.append(y)
            x = np.ndarray(shape=[batch_size, num_features], dtype=np.float32)
            y = np.ndarray(shape=[batch_size, 2], dtype=np.float32)
            # l = [float(n) for n in line.split()]

        x[i % batch_size] = l[:13]

        if l[13] == 0:
            y[i % batch_size][0] = 1
            y[i % batch_size][1] = 0
        else:
            y[i % batch_size][0] = 0
            y[i % batch_size][1] = 1

    return x_batches, y_batches


def cut(train_n_valid_examples, start, end):
    first_half = train_n_valid_examples[:int(len(train_n_valid_examples) * start // batch_size * batch_size)]
    second_half = train_n_valid_examples[int(len(train_n_valid_examples) * end // batch_size * batch_size):]
    if len(first_half) == 0:
        train_examples = second_half
    elif len(second_half) == 0:
        train_examples = first_half
    else:
        train_examples = np.concatenate((first_half, second_half), axis=0)

    valid_examples = train_n_valid_examples[int(len(train_n_valid_examples) * start // batch_size * batch_size):int(
        len(train_n_valid_examples) * end // batch_size * batch_size)]
    return train_examples, valid_examples


def do_round(train_examples, valid_examples):
    train_examples, pca_components, mean, std, corr = create_dataset(train_examples, pca_components=None, mean=None,
                                                                     std=None, corr=None)
    valid_examples, _, _, _, _ = create_dataset(valid_examples, pca_components, mean, std, corr)

    train_batches = create_batches(train_examples)
    valid_batches = create_batches(valid_examples)
    return [[train_batches, valid_batches], corr]


def load_batches(file_names):
    examples = load_txt(file_names)
    shuffle(examples)
    test_examples = examples[int(len(examples) * 0.8 // batch_size * batch_size):]
    train_n_valid_examples = examples[:int(len(examples) * 0.8 // batch_size * batch_size)]
    # train_n_valid_examples = examples
    rounds = []
    for i in range(10):
        start = 0.1 * i
        end = 0.1 * (i + 1)
        train_examples, valid_examples = cut(train_n_valid_examples, start, end)

        rounds.append(do_round(train_examples, valid_examples))

    train_examples, test_examples = cut(examples, 0.8, 1)
    rounds.append(do_round(train_examples, test_examples))

    return rounds


def validate(test_batches, x, y, b, b_v):
    m_x_vals = []
    nm_x_vals = []
    m_y_vals = []
    nm_y_vals = []
    for i, x_b in enumerate(test_batches[0]):
        y_b = test_batches[1][i]
        for j, r in enumerate(y_b):
            if r[0] == 1:
                nm_x_vals.append(x_b[j])
                nm_y_vals.append(r)
            else:
                m_x_vals.append(x_b[j])
                m_y_vals.append(r)

    m_y_ps = sess.run(y, {x: m_x_vals, b: b_v})
    truths_b = [0] * len(m_x_vals)
    for j, x_val in enumerate(m_x_vals):
        truths_b[j] = 0
        if m_y_ps[j][0] > m_y_ps[j][1]:
            if m_y_vals[j][0] == 1:
                truths_b[j] = 1
        else:
            if m_y_vals[j][1] == 1:
                truths_b[j] = 1
    merge_accuracy = np.average(truths_b)

    nm_y_ps = sess.run(y, {x: nm_x_vals, b: b_v})
    truths_b = [0] * len(nm_x_vals)
    for j, x_val in enumerate(nm_x_vals):
        truths_b[j] = 0
        if nm_y_ps[j][0] > nm_y_ps[j][1]:
            if nm_y_vals[j][0] == 1:
                truths_b[j] = 1
        else:
            if nm_y_vals[j][1] == 1:
                truths_b[j] = 1
    non_merge_accuracy = np.average(truths_b)

    return merge_accuracy, non_merge_accuracy


def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
      Note that the Variable is initialized with a truncated normal distribution.
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
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
                          dtype=dtype)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def calculate_cnn_layer_output_size(size, k_size, s_size):
    output_size = (size - k_size + 2 * (k_size // 2)) // s_size + 1
    return output_size


# -------------------------alexnet start-------------------

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def create_alex_layers(mid, input_tensor, input_height, input_width, b):
    parameters = []
    with vs.variable_scope(mid):
        images = input_tensor
        images_height = input_height
        images_width = input_width
        # Model parameters
        images = tf.reshape(images, [-1, images_height, images_width, 1])

    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 11, 1, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # lrn1
    # TODO(shlens, jiayq): Add a GPU version of local response normalization.

    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 1, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 5, 96, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 1, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 256, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 384, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 1, 1],
                           padding='VALID',
                           name='pool5')

    print_activations(pool5)

    tensor = tf.nn.lrn(pool5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # with tf.device("/cpu:0"):
    #    norm_1 = tf.Print(norm_1, [tf.shape(norm_1)], message="norm_1 shape:")

    tensor_height = 1
    tensor_width = 7

    num_output_channels = 256

    vec_len = tensor_height * tensor_width * num_output_channels
    a_mb = tf.reshape(tensor, [-1, vec_len])

    # Model input and output
    W_out = tf.get_variable(str(mid) + '_W_out', shape=[1792, 2])
    b_out = tf.get_variable(str(mid) + '_b_out', shape=[2])

    y = tf.matmul(a_mb, W_out) + b_out + b
    return y


# -------------------end alexnet--------------------------------


# ------------------start cnn

def create_cnn_layers(mid, input_tensor, input_height, input_width, num_layers, kernels, strides, channels):
    with vs.variable_scope(mid):
        tensor = input_tensor
        tensor_height = input_height
        tensor_width = input_width
        num_input_channels = 1

        # Model parameters
        tensor = tf.reshape(tensor, [-1, tensor_height, tensor_width, 1])
        for li in range(num_layers):
            with tf.variable_scope('conv_%d' % (li + 1)):
                num_output_channels = channels[li]

                kernel = _variable_with_weight_decay('weights',
                                                     shape=[kernels[li][0], kernels[li][1], num_input_channels,
                                                            num_output_channels],
                                                     stddev=5e-2,
                                                     wd=0.0)
                conv = tf.nn.conv2d(input=tensor, filter=kernel, strides=[1, 1, 1, 1], padding='SAME',
                                    data_format='NHWC')
                biases = tf.get_variable('biases', [num_output_channels], initializer=tf.constant_initializer(0.1),
                                         dtype=tf.float32)
                pre_activation = tf.nn.bias_add(conv, biases)
                conv = tf.nn.relu(pre_activation)
                pool = tf.nn.max_pool(conv, ksize=[1, kernels[li][0], kernels[li][1], 1],
                                      strides=[1, strides[li][0], strides[li][1], 1],
                                      padding='SAME', data_format='NHWC')
                tensor = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                # with tf.device("/cpu:0"):
                #    norm_1 = tf.Print(norm_1, [tf.shape(norm_1)], message="norm_1 shape:")

                tensor_height = calculate_cnn_layer_output_size(tensor_height, kernels[li][0], strides[li][0])
                tensor_width = calculate_cnn_layer_output_size(tensor_width, kernels[li][1], strides[li][1])

                num_input_channels = num_output_channels

    return tensor, tensor_height, tensor_width, num_output_channels


def create_gbnn_model(mid, x, b, corr):
    # Model parameters

    tensor, tensor_height, tensor_width, num_output_channels = create_cnn_layers(mid, x, 1, 13, 2, [[1, 3], [1, 3]],
                                                                                 [[1, 2], [1, 2]], [16, 32])

    vec_len = tensor_height * tensor_width * num_output_channels
    a_mb = tf.reshape(tensor, [-1, vec_len])

    # ---------------------------branch begin---------------------
    W_gb = tf.get_variable(str(mid) + '_W_bp', shape=[num_features, num_branch_hid_units])
    b_gb = tf.get_variable(str(mid) + '_b_bp', shape=[num_branch_hid_units])
    # mask = tf.get_variable('m', initializer=tf.constant([1,-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=tf.float32))
    mask = tf.constant(corr, dtype=tf.float32)
    # ---------------------------branch end-----------------------
    # Model input and output
    W_out = tf.get_variable(str(mid) + '_W_out', shape=[192, 2])
    b_out = tf.get_variable(str(mid) + '_b_out', shape=[2])

    # --------------------------- branch begin---------------------
    # a_bp = tf.nn.relu(tf.matmul(x * mask, W_bp))
    a_gb = tf.nn.selu(tf.matmul(x * mask, W_gb) + b_gb)
    # --------------------------- branch end-----------------------

    y = tf.matmul(tf.concat([a_mb, a_gb], 1), W_out) + b_out + b
    return y


def create_cnn_model(mid, x, b):
    # Model parameters

    tensor, tensor_height, tensor_width, num_output_channels = create_cnn_layers(mid, x, 1, 13, 2, [[1, 5], [1, 5]],
                                                                                 [[1, 2], [1, 2]], [16, 32])

    vec_len = tensor_height * tensor_width * num_output_channels
    a_mb = tf.reshape(tensor, [-1, vec_len])

    # Model input and output
    W_out = tf.get_variable(str(mid) + '_W_out', shape=[128, 2])
    b_out = tf.get_variable(str(mid) + '_b_out', shape=[2])

    y = tf.matmul(a_mb, W_out) + b_out + b
    return y


def create_mlp_model(mid, x, b):
    # Model parameters
    W_inp = tf.get_variable(str(mid) + '_W_inp',
                            shape=[num_features, num_main_hid_units])  # initializer=tf.constant([23, 42])
    b_inp = tf.get_variable(str(mid) + '_b_inp', shape=[num_main_hid_units])
    W_hids = []
    b_hids = []
    for i in range(num_main_hid_layers):
        W_hids.append(tf.get_variable(str(mid) + '_W_hid_' + str(i), shape=[num_main_hid_units, num_main_hid_units]))
        b_hids.append(tf.get_variable(str(mid) + '_b_hid_' + str(i), shape=[num_main_hid_units]))

    W_out = tf.get_variable(str(mid) + '_W_out', shape=[num_main_hid_units, 2])
    b_out = tf.get_variable(str(mid) + '_b_out', shape=[2])

    # W_bp = tf.get_variable(str(mid)+'_W_bp', shape=[num_features,num_main_hid_units])
    # mask = tf.get_variable('m', initializer=tf.constant([1,-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=tf.float32))
    # mask = tf.constant(pearson, dtype=tf.float32)
    # Model input and output

    a_hid = tf.nn.relu(tf.matmul(x, W_inp) + b_inp)

    for i in range(num_main_hid_layers):
        a_hid = tf.nn.relu(tf.matmul(a_hid, W_hids[i]) + b_hids[i])

    # a_bp = tf.nn.relu(tf.matmul(x * mask, W_bp))

    y = tf.matmul(a_hid, W_out) + b_out + b
    return y


def run():
    for r, [batches, corr] in enumerate(rounds):
        # saver = tf.train.import_meta_graph('model.ckpt.meta')
        # saver.restore(sess,tf.train.latest_checkpoint('./'))

        x = tf.placeholder(tf.float32, shape=[None, num_features])
        # b = tf.placeholder(tf.float32, shape=[num_main_hid_units])
        b = tf.placeholder(tf.float32, shape=[2])

        mid = nn + '_' + str(corr_index) + '_' + str(num_main_hid_units) + '_' + str(num_epoches) + '_' + str(r)
        if nn == 'gbnn':
            y = create_gbnn_model(mid, x, b, corr[corr_index])
        elif nn == 'cnn':
            y = create_cnn_model(mid, x, b)
        elif nn == 'mlp':
            y = create_mlp_model(mid, x, b)

        elif nn == 'alex':
            y = create_alex_layers(mid, x, 1, 13, b)

        y_ = tf.placeholder(tf.float32, shape=[batch_size, 2])

        with open('accuracy_' + mid + '.log', 'a') as log_file:
            log_file.write(mid + '\n')

        # loss
        loss = tf.losses.softmax_cross_entropy(y_, y)
        # loss = tf.reduce_sum(tf.square(y - y_)) # sum of the squares
        # optimizer

        # optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)  # reset values to wrong

        # b_v = np.ones(num_main_hid_units).astype(np.float32)
        b_v = np.ones(2).astype(np.float32)
        losses = []

        import time

        train_time = 0

        for k in range(1, num_epoches + 1):
            for i, x_b in enumerate(batches[0][0]):
                y_b = batches[0][1][i]
                x_b = np.reshape(x_b, (batch_size, num_features))
                y_b = np.reshape(y_b, (batch_size, 2))

                # training loop
                start_train_time = time.time()
                sess.run(train, {x: x_b, y_: y_b, b: b_v})
                end_train_time = time.time()
                train_time = train_time + end_train_time - start_train_time
            if k == 1 or k % 10 == 0:
                l = sess.run(loss, {x: x_b, y_: y_b, b: b_v})
                losses.append(l)

            if k % 50 == 0:
                ma, nma = validate(batches[1], x, y, b, b_v)
                with open('accuracy_' + mid + '.log', 'a') as log_file:
                    log_file.write(
                        str(k) + "th epoch merge accuracy:" + str(ma) + ", non-merge accuracy:" + str(nma) + "\n")

        if k % 50 != 0:
            ma, nma = validate(batches[1], x, y, b, b_v)
            with open('accuracy_' + mid + '.log', 'a') as log_file:
                log_file.write(
                    str(k) + "th epoch merge accuracy:" + str(ma) + ", non-merge accuracy:" + str(nma) + "\n")

        # saver = tf.train.Saver()
        # save_path = saver.save(sess, "gbnn_model.ckpt")

        with open('roc_' + mid + '.csv', 'w') as roc_f:
            roc_f.write('merge,non-merge\n')
            for i in range(400):
                # b_v = np.random.normal(1, 1, num_main_hid_units)
                b_v = [1.0, -199 + i]
                ma, nma = validate(batches[1], x, y, b, b_v)
                roc_f.write(str(ma) + ',' + str(nma) + '\n')

        # log losses
        with open('loss_' + mid + '.log', 'w') as logf:
            for l in losses:
                logf.write(str(l) + '\n')

        print(mid + " training time: ", train_time)
    # evaluate training accuracy
    # curr_W, curr_b, curr_loss = sess.run([W1, b1, loss], {x: x_b, y_: y_b})
    # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

    # x_batches, y_batches = load_txt(['train_Data0515.txt'])

    '''
    truths = []
    for i, x_b in enumerate(x_batches2):
        y_b = y_batches2[i]
        x_b = np.reshape(x_b,(batch_size,num_features))
        y_b = np.reshape(y_b,(batch_size,2))
        y_p = sess.run(y, {x: x_b})

        truths_b = [0]*batch_size
        for j in range(batch_size):
            truths_b [j] = 0
            if y_p[j][0] > y_p[j][1]:
                if y_b[j][0] == 1:
                    truths_b [j] = 1
            else:
                if y_b[j][1] == 1:
                    truths_b [j] = 1
        truths.extend(truths_b)

    accuracy = np.average(truths)
    print("accuracy:", accuracy)
    '''


rounds = load_batches(['train_Data0400.txt', 'train_Data0500.txt', 'train_Data0515.txt'])

with tf.Session() as sess:
    for n in ['alex']:
        nn = n
        for nhus in [64]:
            num_main_hid_units = nhus
            num_branch_hid_units = nhus
            for nes in [500]:
                num_epoches = nes
                for ci in [0]:
                    corr_index = ci
                    run()


