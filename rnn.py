import tensorflow as tf
from progress.bar import Bar
import numpy as np
import random
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from brackseqgen import seq_len

from constants import MAX_SEQ_LEN, BRACKETS_BATCH, SAVED_MODEL_PATH


class RNN:
    def __init__(self,
                 session: tf.Session,
                 use_lstm: bool = False,
                 lr: float = .1,
                 hidden_n: int = 100,
                 brackets_batch: int = BRACKETS_BATCH,
                 checkpoint_path = SAVED_MODEL_PATH):
        self.sess = session
        self.use_lstm = use_lstm
        self.lr = lr
        self.hidden_n = hidden_n
        self.brackets_batch = brackets_batch
        assert MAX_SEQ_LEN % self.brackets_batch == 0
        self._create_model()
        self.saver = tf.train.Saver()
        self.checkpoint_path = checkpoint_path
        try:
            self.saver.restore(self.sess, checkpoint_path)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            tf.global_variables_initializer().run()

    def _add_recurrent_layers(self):
        self.x = tf.placeholder(shape=[None, MAX_SEQ_LEN], dtype=tf.float32, name='x')
        self.y = tf.placeholder(shape=[None, 3], dtype=tf.float32, name='y')
        self.mb_size = tf.shape(self.x)[0]
        self.max_len = tf.shape(self.x)[1]
        print(self.max_len)
        tf.assert_equal(tf.mod(self.max_len, self.brackets_batch), tf.constant(0),
                        message="length of the padding must be divisible by size of brackets batch")
        time_steps = MAX_SEQ_LEN // self.brackets_batch
        self.x_batched = tf.reshape(self.x, shape=[self.mb_size, time_steps, self.brackets_batch])
        zeros_dims = tf.stack([self.mb_size, self.hidden_n])
        w_init = tf.initializers.random_normal(stddev=np.sqrt(2 / (self.brackets_batch + self.hidden_n)))
        if self.use_lstm:
            state = tf.fill(zeros_dims, 0.), tf.fill(zeros_dims, 0.)
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_n, initializer=w_init)
        else:
            state = tf.fill(zeros_dims, 0.)
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_n)

        for time_step in range(time_steps):
            cur_x = self.x_batched[:, time_step, :]
            self.rnn_cell_output, state = self.rnn_cell(inputs=cur_x, state=state)

    def _add_dense_layers(self):
        self.output_layer = tf.layers.dense(units=3,
                                            inputs=self.rnn_cell_output,
                                            activation=None,
                                            use_bias=True)

    def _add_training_objectives(self):
        # self.loss = tf.losses.absolute_difference(self.y, self.output_layer)
        self.max_open_loss = tf.losses.mean_squared_error(self.y[:, 0], self.output_layer[:, 0])
        self.max_cons_loss = tf.losses.mean_squared_error(self.y[:, 1], self.output_layer[:, 1])
        self.max_dist_loss = tf.losses.mean_squared_error(self.y[:, 2], self.output_layer[:, 2])
        self.loss = tf.losses.mean_squared_error(self.y, self.output_layer)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_model(self):
        self._add_recurrent_layers()
        self._add_dense_layers()
        self._add_training_objectives()

    def plot_val_losses(self, val_losses, max_open_losses, max_cons_losses, max_dist_losses):
        print("Saving plot of learning processs")
        plt.figure(figsize=(7, 4))
        plt.title("Val loss")
        plt.ylim(ymax=300)
        plt.plot(val_losses, label='Overall MSE')
        plt.plot(max_open_losses, label='Max open MSE')
        plt.plot(max_cons_losses, label='Max cons MSE')
        plt.plot(max_dist_losses, label='Max dist MSE')
        plt.xlabel("epoch")
        plt.legend()
        base_path = self.checkpoint_path[4:-5]
        plt.savefig('plots/' + base_path + '_val_losses.png')

    def plot_losses_by_length(self, losses, max_open_losses, max_cons_losses, max_dist_losses):
        print("Saving plot of losses by length")
        plt.figure(figsize=(7, 4))
        plt.title("Losses by length of the sequence")
        lengths = list(range(0, MAX_SEQ_LEN + 1, 2))
        plt.plot(lengths, losses[::2], label='Overall MSE')
        plt.plot(lengths, max_open_losses[::2], label='Max open MSE')
        plt.plot(lengths, max_cons_losses[::2], label='Max cons MSE')
        plt.plot(lengths, max_dist_losses[::2], label='Max dist MSE')
        plt.xlabel("sequence length")
        plt.legend()
        base_path = self.checkpoint_path[4:-5]
        plt.savefig('plots/' + base_path + '_losses_by_length.png')

    def plot_errors_by_ground_truth(self, max_open_errs, max_cons_errs, max_dist_errs, mode: str,):
        print("Saving lot of {0} errors by ground truth".format(mode))
        plt.figure(figsize=(7, 4))
        plt.title("Errors by ground truth")
        plt.plot(max_open_errs, label="Max open ME ({0})".format(mode))
        plt.plot(max_cons_errs, label="Max cons ME ({0})".format(mode))
        odd = list(range(1, MAX_SEQ_LEN, 2))
        plt.plot(odd, max_dist_errs[odd], label="Max dist ME ({0}".format(mode))
        plt.xlabel("Ground truth")
        plt.ylabel("mean |prediction - ground_truth|")
        plt.legend()
        base_path = self.checkpoint_path[4:-5]
        plt.savefig('plots/' + base_path + '_{0}_errors_by_ground_truth.png'.format(mode))

    def fit(self, x, y, num_epochs=20, steps_per_epoch=200, batch_size=1000, validate=False,
            x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        losses = []
        val_losses = []
        val_max_open_losses = []
        val_max_cons_losses = []
        val_max_dist_losses = []
        for epoch_no in range(num_epochs):
            print("Epoch {epoch_no}/{num_epochs}".format(epoch_no=epoch_no, num_epochs=num_epochs))
            bar = Bar('', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)ds')
            for step_no in range(steps_per_epoch):
                eff_batch_size = min(batch_size, x.shape[0])
                batch_idx = random.choices(population=list(range(x.shape[0])), k=eff_batch_size)
                batch_x = x[batch_idx]
                batch_y = y[batch_idx]
                loss, train = self.sess.run([self.loss, self.train_op],
                                            feed_dict={self.x: batch_x, self.y: batch_y})
                losses.append(loss)
                bar.message = 'loss: {loss:.8f}'.format(loss=np.mean(losses[-steps_per_epoch:]))
                bar.next()
            bar.finish()
            if validate:
                val_loss, val_max_open_loss, val_max_cons_loss, val_max_dist_loss =\
                    self.evaluate(x_val, y_val, batch_size=batch_size)
                val_losses.append(val_loss)
                val_max_open_losses.append(val_max_open_loss)
                val_max_cons_losses.append(val_max_cons_loss)
                val_max_dist_losses.append(val_max_dist_loss)
                print("Validation loss: {val_loss:.8f}".format(val_loss=val_loss))

        # Plot
        self.plot_val_losses(val_losses, val_max_open_losses, val_max_cons_losses, val_max_dist_losses)

    def predict(self, x, batch_size=99):
        preds = []
        steps = (len(x) // batch_size) + (1 if len(x) % batch_size > 0 else 0)
        for step_no in range(steps):
            cur_x = x[step_no * batch_size: (step_no + 1) * batch_size]
            cur_preds = self.sess.run([self.output_layer], feed_dict={self.x: cur_x})[0]
            preds.append(cur_preds)
        preds = np.concatenate(preds, axis=0)
        return preds

    def evaluate(self, x, y, batch_size=99) -> Tuple[float, float, float, float]:
        """
        :param x:
        :param y:
        :param batch_size:
        :return: (MSE loss, MSE loss for max_open statistic,
        MSE loss for max_cons statistic, MSE loss for max_dist statistic)
        """
        loss = 0.
        max_open_loss = 0.
        max_cons_loss = 0.
        max_dist_loss = 0.
        steps = (len(x) // batch_size) + (1 if len(x) % batch_size > 0 else 0)
        for step_no in range(steps):
            cur_x = x[step_no * batch_size: (step_no + 1) * batch_size]
            cur_y = y[step_no * batch_size: (step_no + 1) * batch_size]
            cur_loss, cur_max_open_loss, cur_max_cons_loss, cur_max_dist_loss =\
                self.sess.run([self.loss, self.max_open_loss, self.max_cons_loss, self.max_dist_loss],
                              feed_dict={self.x: cur_x, self.y: cur_y})
            loss += cur_loss * len(cur_x)
            max_open_loss += cur_max_open_loss * len(cur_x)
            max_cons_loss += cur_max_cons_loss * len(cur_x)
            max_dist_loss += cur_max_dist_loss * len(cur_x)
        loss /= len(x)
        max_open_loss /= len(x)
        max_cons_loss /= len(x)
        max_dist_loss /= len(x)
        return loss, max_open_loss, max_cons_loss, max_dist_loss

    def measure_length_loss_correlation(self, x, y):
        print("Measuring correlation between length and loss")
        losses = [[] for _ in range(MAX_SEQ_LEN+1)]
        max_open_losses = [[] for _ in range(MAX_SEQ_LEN+1)]
        max_cons_losses = [[] for _ in range(MAX_SEQ_LEN+1)]
        max_dist_losses = [[] for _ in range(MAX_SEQ_LEN+1)]
        for i in range(len(x)):
            inp = [x[i]]
            ans = [y[i]]
            length = seq_len(inp)
            loss, max_open_loss, max_cons_loss, max_dist_loss = self.evaluate(inp, ans)
            losses[length].append(loss)
            max_open_losses[length].append(max_open_loss)
            max_cons_losses[length].append(max_cons_loss)
            max_dist_losses[length].append(max_dist_loss)

        # Calculate mean of found losses
        losses = list(map(np.mean, losses))
        max_open_losses = list(map(np.mean, max_open_losses))
        max_cons_losses = list(map(np.mean, max_cons_losses))
        max_dist_losses = list(map(np.mean, max_dist_losses))
        self.plot_losses_by_length(losses, max_open_losses, max_cons_losses, max_dist_losses)

    def measure_ground_truth_prediction_correlation(self, x, y):
        print("Measuring ground truth prediction correlation")
        max_open_errs = [[] for _ in range(MAX_SEQ_LEN+1)]
        max_cons_errs = [[] for _ in range(MAX_SEQ_LEN+1)]
        max_dist_errs = [[] for _ in range(MAX_SEQ_LEN+1)]
        preds = self.predict(x)
        for i in range(len(x)):
            max_open_ground, max_cons_ground, max_dist_ground = y[i]
            max_open_pred, max_cons_pred, max_dist_pred = preds[i]
            max_open_errs[max_open_ground].append(abs(max_open_ground - max_open_pred))
            max_cons_errs[max_cons_ground].append(abs(max_cons_ground - max_cons_pred))
            max_dist_errs[max_dist_ground].append(abs(max_dist_ground - max_dist_pred))

        # Absolute errors
        max_open_errs_abs = np.array(list(map(np.mean, max_open_errs)))
        max_cons_errs_abs = np.array(list(map(np.mean, max_cons_errs)))
        max_dist_errs_abs = np.array(list(map(np.mean, max_dist_errs)))

        def abs_to_rel(errs):
            ret = np.copy(errs)
            for i in range(len(ret)):
                ret[i] /= np.float32(i)
            return ret
        max_open_errs_rel = abs_to_rel(max_open_errs_abs)
        max_cons_errs_rel = abs_to_rel(max_cons_errs_abs)
        max_dist_errs_rel = abs_to_rel(max_dist_errs_abs)
        self.plot_errors_by_ground_truth(max_open_errs_abs, max_cons_errs_abs, max_dist_errs_abs, mode='absolute')
        self.plot_errors_by_ground_truth(max_open_errs_rel, max_cons_errs_rel, max_dist_errs_rel, mode='relative')


    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)
