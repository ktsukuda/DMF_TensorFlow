import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DMF:

    def __init__(self, n_user, n_item, rating_matrix, lr, config):
        self._parse_args(n_user, n_item, rating_matrix, lr, config)
        self._build_inputs()
        self._build_parameters()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, n_user, n_item, rating_matrix, lr, config):
        self.n_user = n_user
        self.n_item = n_item
        self.max_rating = 5
        self.user_ratings = tf.convert_to_tensor(rating_matrix)
        self.item_ratings = tf.transpose(self.user_ratings)
        self.lr = lr
        self.user_layers = list(map(int, config['MODEL']['user_layers'].split()))
        self.item_layers = list(map(int, config['MODEL']['item_layers'].split()))

    def _build_inputs(self):
        with tf.name_scope('inputs'):
            self.user = tf.placeholder(tf.int32)
            self.item = tf.placeholder(tf.int32)
            self.rating = tf.placeholder(tf.float32)

    def _build_parameters(self):
        def initialized_parameters(name, shape):
            return tf.get_variable(
                name,
                shape=shape,
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(mean=0, stddev=0.01))

        with tf.name_scope('parameters'):
            with tf.name_scope('user_layers'):
                self.user_Ws = []
                self.user_biases = []
                self.user_Ws.append(initialized_parameters('user_W1', [self.n_item, self.user_layers[0]]))
                for i, output_dim in enumerate(self.user_layers[1:]):
                    self.user_Ws.append(initialized_parameters('user_W{}'.format(i+2), [self.user_layers[i], output_dim]))
                    self.user_biases.append(initialized_parameters('user_bias{}'.format(i+2), [output_dim]))

            with tf.name_scope('item_layers'):
                self.item_Ws = []
                self.item_biases = []
                self.item_Ws.append(initialized_parameters('item_W1', [self.n_user, self.item_layers[0]]))
                for i, output_dim in enumerate(self.item_layers[1:]):
                    self.item_Ws.append(initialized_parameters('item_W{}'.format(i+2), [self.item_layers[i], output_dim]))
                    self.item_biases.append(initialized_parameters('item_bias{}'.format(i+2), [output_dim]))

    def _build_model(self):
        with tf.name_scope('model'):
            user_rating = tf.nn.embedding_lookup(self.user_ratings, self.user)
            item_rating = tf.nn.embedding_lookup(self.item_ratings, self.item)

            user_out = tf.matmul(user_rating, self.user_Ws[0])
            for user_W, user_bias in zip(self.user_Ws[1:], self.user_biases):
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, user_W), user_bias))

            item_out = tf.matmul(item_rating, self.item_Ws[0])
            for item_W, item_bias in zip(self.item_Ws[1:], self.item_biases):
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, item_W), item_bias))

            normalized_user_out = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
            normalized_item_out = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
            cossim = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (normalized_user_out * normalized_item_out)
            self.pred = tf.maximum(1e-6, cossim)

    def _build_loss(self):
        with tf.name_scope('loss'):
            normalized_rating = self.rating / self.max_rating
            losses = normalized_rating * tf.log(self.pred) + (1 - normalized_rating) * tf.log(1 - self.pred)
            self.loss = -tf.reduce_sum(losses)

    def _build_train(self):
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def predict(self, sess, feed_dict):
        return self.pred.eval(feed_dict=feed_dict, session=sess)
