import tensorflow as tf
import random
import argparse
import os

import rnn
from brackseqgen import *
from constants import VAL_SPLIT, BRACKETS_BATCH


def prepare_data(val_split: float = VAL_SPLIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Preparing data")
    if os.path.isfile('x_train.npy') and os.path.isfile('y_train.npy')\
       and os.path.isfile('x_val.npy') and os.path.isfile('y_val.npy'):
        print("Loading from files")
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        x_val = np.load('x_val.npy')
        y_val = np.load('y_val.npy')
    else:
        print("First time preparing")
        x_train = []
        y_train = []

        # Make sure there is one sample for each combination of parameters up to length 100
        gen = FixedParamsGen()
        x, y = gen.all_sequences(max_length=100)
        x_train.append(x)
        y_train.append(y)

        # Generate random samples of various lengths
        gen = RandomBrackGen()
        lim = 200000
        for length in range(2, MAX_SEQ_LEN + 1, 2):
            for t in range(2 * lim // MAX_SEQ_LEN):
                x, y = gen.generate_sequence_of_length(length=length)
                x_train.append(x)
                y_train.append(y)

        # Generate open close sequences of all lengths
        gen = OpenCloseBrackSeqGen()
        for length in range(2, MAX_SEQ_LEN + 1, 2):
            x, y = gen.generate_sequence_of_length(length=length)
            x_train.append(x)
            y_train.append(y)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Extract validation set
        indices = list(range(len(x_train)))
        random.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]

        val_size = int(len(x_train) * val_split)
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]
        x_train = x_train[:-val_size]
        y_train = y_train[:-val_size]

        print("Saving data")
        np.save('x_train', x_train)
        np.save('y_train', y_train)
        np.save('x_val', x_val)
        np.save('y_val', y_val)

    return x_train, y_train, x_val, y_val


def interact(net: rnn.RNN) -> None:
    seq_str = input().strip()
    seq = str_to_arr(seq_str)
    pred = net.predict(seq)
    print("Predictions: ", pred, ", correct answers: ", compute_stats(seq))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train or play with RNN.')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-n', default=128, type=int, help='hidden neurons count')
    parser.add_argument('-e', default=1000, type=int, help='number of epochs')
    parser.add_argument('-b', default=BRACKETS_BATCH, type=int, help='brackets batch size')
    parser.add_argument('--mode', choices=['lstm', 'rnn'], type=str, default='lstm')
    parser.add_argument('--interact', action='store_true')
    args = parser.parse_args()
    hidden_n = args.n
    num_epochs = args.e
    lr = args.lr
    brackets_batch = args.b
    use_lstm = (args.mode == 'lstm')
    interactive = args.interact

    with tf.Session() as sess:
        checkpoint_path = 'tmp/' + ('lstm' if use_lstm else 'rnn') + str(hidden_n) + '_' + str(brackets_batch) + '.ckpt'
        net = rnn.RNN(session=sess, use_lstm=use_lstm, lr=lr, brackets_batch=brackets_batch, hidden_n=hidden_n,
                      checkpoint_path=checkpoint_path)
        if interactive:
            while True:
                interact(net)
        else:
            x_train, y_train, x_val, y_val = prepare_data()
            net.fit(x_train, y_train, num_epochs=num_epochs, validate=True, x_val=x_val, y_val=y_val)
            net.save()
            net.measure_ground_truth_prediction_correlation(x_val, y_val)
            net.measure_length_loss_correlation(x_val, y_val)
