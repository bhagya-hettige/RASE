import tensorflow as tf
import scipy.sparse as sp
import numpy as np

seed = 42


def sparse_feeder(M):
    M = sp.coo_matrix(M, dtype=np.float32)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


def sparse_gather(indices, values, selected_indices, axis=0):
    mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
    to_select = tf.where(mask)[:, 1]
    return tf.gather(indices, to_select, axis=0), tf.gather(values, to_select, axis=0)


class RASE:
    def __init__(self, args):
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.X = tf.SparseTensor(*sparse_feeder(args.X))
        self.N, self.D = args.X.shape
        self.L = args.embedding_dim
        self.n_hidden = [512]

        self.alpha = args.alpha
        self.p = args.p

        self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_k = tf.compat.v1.placeholder(name='u_k', dtype=tf.int32, shape=[None])
        self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])

        self.__create_model()

        if not args.is_all:
            self.val_edges = args.val_edges
            self.val_ground_truth = args.val_ground_truth
            if args.structural_distance == 'W2':
                self.neg_val_energy = -self.energy_w2(self.val_edges[:, 0], self.val_edges[:, 1])  # W2 distance
            elif args.structural_distance == 'KL':
                self.neg_val_energy = -self.energy_kl(self.val_edges[:, 0], self.val_edges[:, 1])  # KL distance
            self.val_set = True
        else:
            self.val_set = False

        # attribute loss
        X_hat = tf.gather(self.X_hat, self.u_k)  # decoded node attribute vectors
        slice_indices, slice_values = sparse_gather(self.X.indices, self.X.values, tf.cast(self.u_k, tf.int64))
        X = tf.gather(tf.sparse.to_dense(tf.SparseTensor(slice_indices, slice_values,
                                                         tf.cast(tf.shape(self.X), tf.int64)), validate_indices=False),
                      self.u_k)  # original node attribute vectors
        L_a = tf.reduce_mean(tf.square(tf.subtract(X, X_hat)))  # Euclidean distance

        # attribute regularization
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005)
        attr_reg = tf.reduce_mean(tf.contrib.layers.apply_regularization(l1_regularizer, self.trained_variables))

        # structural loss
        if args.structural_distance == 'W2':
            self.energy = -self.energy_w2(self.u_i, self.u_j)  # W2 distance
        elif args.structural_distance == 'KL':
            self.energy = -self.energy_kl(self.u_i, self.u_j)  # KL distance
        L_s = -tf.reduce_mean(tf.math.log_sigmoid(self.label * self.energy))  # Sigmoid loss with negative sampling

        # structural regularization
        str_reg = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.sigma - tf.square(self.embedding) -
                                                      tf.exp(self.sigma), axis=1))

        # overall loss
        self.loss = L_s + 1e-5 * str_reg + self.alpha * (L_a + 1e-5 * attr_reg)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def __create_model(self):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden
        trained_variables = []

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())
            trained_variables.extend([W])
            trained_variables.extend([b])

            if i == 1:
                X = tf.cond(tf.greater(self.p, 0), lambda: self.binomial_noise_layer(self.X), lambda: self.X)
                encoded = tf.sparse_tensor_dense_matmul(X, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init())
        self.embedding = tf.matmul(encoded, W_mu) + b_mu

        W_sigma = tf.get_variable(name='W_sigma', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_sigma = tf.get_variable(name='b_sigma', shape=[self.L], dtype=tf.float32, initializer=w_init())
        log_sigma = tf.matmul(encoded, W_sigma) + b_sigma
        self.sigma = tf.nn.elu(log_sigma) + 1 + 1e-14

        sizes.reverse()

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W_dec{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b_dec{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())
            trained_variables.extend([W])
            trained_variables.extend([b])

            if i == 1:
                decoded = tf.matmul(encoded, W) + b
            else:
                decoded = tf.matmul(decoded, W) + b

            if i == len(sizes) - 1:
                self.X_hat = tf.nn.elu(decoded)
            else:
                decoded = tf.nn.relu(decoded)

        self.trained_variables = trained_variables

    def binomial_noise_layer(self, input_layer):
        noise = tf.keras.backend.random_binomial(tf.shape(input_layer), self.p, seed=seed)
        return input_layer.__mul__(noise)

    def energy_kl(self, u_i, u_j):
        mu_i = tf.gather(self.embedding, u_i)
        sigma_i = tf.gather(self.sigma, u_i)
        mu_j = tf.gather(self.embedding, u_j)
        sigma_j = tf.gather(self.sigma, u_j)

        sigma_ratio = sigma_j / sigma_i
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_i - mu_j) / sigma_i, 1)

        ij_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        sigma_ratio = sigma_i / sigma_j
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(mu_j - mu_i) / sigma_j, 1)

        ji_kl = 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

        kl_distance = 0.5 * (ij_kl + ji_kl)

        return kl_distance

    def energy_w2(self, u_i, u_j):
        mu_i = tf.gather(self.embedding, u_i)
        sigma_i = tf.gather(self.sigma, u_i)
        mu_j = tf.gather(self.embedding, u_j)
        sigma_j = tf.gather(self.sigma, u_j)

        delta = mu_i - mu_j
        d1 = tf.reduce_sum(delta * delta, axis=1)
        x0 = sigma_i - sigma_j
        d2 = tf.reduce_sum(x0 * x0, axis=1)
        wd = d1 + d2

        return wd
