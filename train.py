import tensorflow as tf
import argparse
from model import RASE
from utils import DataUtils, score_link_prediction, score_node_classification
import pickle
import time
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default='cora_ml')
    parser.add_argument('--suf', default='')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--p', type=float, default=0.98)  # set to 0 if no attribute noise
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--is_all', default=False)  # train with all edges; no validation or test set
    parser.add_argument('--structural_distance', default='W2', help='W2 or KL')
    args = parser.parse_args()
    args.is_all = True if args.is_all == 'True' else False
    train(args)


def train(args):
    graph_file = 'data/%s/%s.npz' % (args.name, args.name)
    graph_file = graph_file.replace('.npz', '_train.npz') if not args.is_all else graph_file
    data_loader = DataUtils(graph_file, args.is_all)

    initial_tolerance = 200
    early_stopping_score_max = -1.0
    tolerance = initial_tolerance

    args.X = data_loader.X if args.suf != 'oh' else sp.identity(data_loader.X.shape[0])
    if not args.is_all:
        args.val_edges = data_loader.val_edges
        args.val_ground_truth = data_loader.val_ground_truth

    model = RASE(args)

    def save_embeddings(sess):
        mu, sigma = sess.run([model.embedding, model.sigma])
        pickle.dump({'mu': data_loader.embedding_mapping(mu), 'sigma': data_loader.embedding_mapping(sigma)},
                    open('emb/rase%s_embedding.pkl' % ('_all' if args.is_all else ''), 'wb'))

    with tf.Session() as sess:
        print('-------------------------- RASE Alpha=%0.2f p=%0.2f --------------------------' % (args.alpha, args.p))
        if model.val_set:
            print('batches\tloss\tval_auc\tval_ap\tsampling time\ttraining_time\tdatetime')
        else:
            # print('batches\tloss\tf1_micro\tf1_macro\tsampling time\ttraining_time\tdatetime')
            print('batches\tloss\tsampling time\ttraining_time\tdatetime')

        tf.global_variables_initializer().run()
        sampling_time, training_time = 0, 0

        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, u_k, label = data_loader.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.u_k: u_k, model.label: label}
            t2 = time.time()
            sampling_time += t2 - t1

            loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)

            training_time += time.time() - t2

            if model.val_set:
                # link prediction
                val_energy = sess.run(model.neg_val_energy)
                val_auc, val_ap = score_link_prediction(data_loader.val_ground_truth, val_energy)
                early_stopping_score = val_auc + val_ap

                if b % 50 == 0:
                    print('%d\t%f\t%f\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, val_auc, val_ap, sampling_time, training_time,
                                                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    sampling_time, training_time = 0, 0
            else:
                # node classification
                # emb = sess.run(model.embedding)
                # f1_micro, f1_macro = score_node_classification(emb, data_loader.labels, 0.3, 5)
                # early_stopping_score = f1_micro + f1_macro
                early_stopping_score = -loss
                if b % 50 == 0:
                    # print('%d\t%f\t%f\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, f1_micro, f1_macro, sampling_time, training_time,
                    #                                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                    sampling_time, training_time = 0, 0

            if early_stopping_score > early_stopping_score_max:
                early_stopping_score_max = early_stopping_score
                tolerance = initial_tolerance
                save_embeddings(sess)
            else:
                tolerance -= 1

            if tolerance == 0:
                break

        if tolerance > 0:
            print('The model has not been converged.. Exit due to number of batches..')


if __name__ == '__main__':
    main()
