__author__ = 'taras-sereda'

import numpy as np
import numpy.random
import pickle
import gzip
import math
import utils
import PIL.Image as Image
import os
import copy

class RBM():
    def __init__(self,
                 input=None,
                 n_visible=None,
                 n_hidden=None,
                 a=None,
                 b=None,
                 W=None,
                 lr=None,
                 b_size=None
                 ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.b_size = b_size

        if W is None:
            self.W = np.random.random((n_visible, n_hidden))

        if a is None:
            self.a = np.random.random(n_visible)

        if b is None:
            self.b = np.random.random(b)

        self.input = input

    def free_energy(self, visible):

        wx_b = np.dot(visible, self.W) + self.b
        vbias_term = np.dot(visible, self.a)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)))

        return - vbias_term - hidden_term

    def sigmoid(self, activation):
        return 1 / (1 + np.exp(-activation))

    def forward(self, visible):
        wx_b = np.dot(visible, self.W) + self.b
        return self.sigmoid(wx_b)

    def backward(self, hidden):
        wh_a = np.dot(hidden, self.W.T) + self.a
        return self.sigmoid(wh_a)

    def sample_h_given_v(self, v1_sample):
        h1_mean = self.forward(v1_sample)
        h1_sample = np.random.binomial(n=1, p=h1_mean, size=h1_mean.shape)
        return h1_mean, h1_sample

    def sample_v_given_h(self, h1_sample):
        v1_mean = self.backward(h1_sample)
        v1_sample = np.random.binomial(n=1, p=v1_mean, size=v1_mean.shape)
        return v1_mean, v1_sample

    def compute_updates(self):

        batches = int(math.ceil(self.input.shape[0]*1.0/self.b_size))
        for i in range(batches):
            v1_sample = self.input[i*self.b_size: (i+1)*self.b_size]
            # positive phase
            h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
            # negative phase
            v2_mean, v2_sample = self.sample_v_given_h(h1_sample)
            h2_mean = self.forward(v2_sample)

            p_pos = v1_sample[:, :, np.newaxis] * h1_mean[:, np.newaxis, :]
            p_neg = v2_sample[:, :, np.newaxis] * h2_mean[:, np.newaxis, :]

            self.W += self.lr * (p_pos.mean(0) - p_neg.mean(0))
            self.b += self.lr * (h1_mean.mean(0) - h2_mean.mean(0))
            self.a += self.lr * (v1_sample.mean(0) - v2_sample.mean(0))

        return self.W, self.b, self.a

    def reconstruct_visible(self, v1_sample):
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        _, v2_sample = self.sample_v_given_h(h1_sample)
        return v2_sample

def load_data():
    mnist = 'data/mnist.pkl.gz'

    f = gzip.open(mnist, 'rb')
    train_set, test_set, val_set = pickle.load(f)

    return train_set, test_set, val_set

def main():
    if not os.path.exists('rbm_vis'):
        os.makedirs('rbm_vis')

    batch_size = 20
    train_set, test_set, val_set = load_data()

    test_x, test_y = test_set
    test_idx = np.where(test_y == 2)[0]
    test_sub_set = test_x[test_idx]

    train_x, train_y = train_set
    train_idx = np.where(train_y == 2)[0]
    train_sub_set = train_x[train_idx]
    rbm = RBM(
          n_visible= 28 * 28,
          n_hidden=100,
          input=train_sub_set,
          lr=0.5,
          b_size=batch_size
          )
    epoches = 20
    for i in range(epoches):

        w, b, a = rbm.compute_updates()
        img = Image.fromarray(utils.tile_raster_images(w.T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)))
        img.save('rbm_vis/filter_eapoch_{}.png'.format(i))

        # test phase
        random_idx = numpy.random.choice(range(len(test_sub_set)))
        random_smpl = test_sub_set[random_idx]
        corr_random_smpl = copy.deepcopy(random_smpl)
        corr_random_smpl[:392] = 0

        # reshape
        test_itm = test_sub_set[random_idx].reshape((1,-1))
        visible_p = corr_random_smpl.reshape((1,-1))
        v2_sample = rbm.reconstruct_visible([corr_random_smpl])
        before_after = np.concatenate((test_itm, visible_p, v2_sample), axis=0)
        reconstruct = Image.fromarray(utils.tile_raster_images(before_after,
                img_shape=(28, 28),
                tile_shape=(1, 3),
                tile_spacing=(1, 1)))
        reconstruct.save('rbm_vis/reconst_eapoch_{}.png'.format(i))
        print('finished eapoch {}'.format(i))


if __name__ == "__main__":
    main()






