# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from codecs import open as codecs_open
import time
from datetime import datetime

from ml.nlp.utils import VocabLoader, DataLoader, load_embedding, save, load, mkdir_p
from ml.nlp.cnn import Model, CnnDetectorContext

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ENCODING = 'utf-8'
__config = dict()

__context = None
__model = None


def load_default_prediction_model(train_dir=None, data_dir=None):
    global __config
    global __context
    global __model
    if __model is not None:
        return __model
    if train_dir is None:
        train_dir = os.path.join(THIS_DIR, 'models', '1543501620')
    if data_dir is None:
        data_dir = os.path.join(THIS_DIR, 'data', 'ted500')
    config = load(os.path.join(train_dir, 'flags.cPickle'))
    config['train_dir'] = train_dir
    config['data_dir'] = data_dir
    if 'num_classes' not in config:
        config['num_classes'] = 65
    if 'sent_len' not in config:
        config['sent_len'] = 257
    __config = config
    __context = CnnDetectorContext(config)

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            train_dir = config['train_dir']
            ckpt = tf.train.get_checkpoint_state(train_dir)
            # Since this might be on a different FS mmodify the path
            # Todo: check for this
            model_filename = Path(ckpt.model_checkpoint_path).name
            model_file = os.path.join(train_dir, model_filename)
            config['model_checkpoint_path'] = model_file
            config['model_filename'] = model_filename
            if ckpt and ckpt.model_checkpoint_path:
                __model = m = Model(config, is_train=False)
                __model.set_context(__context)
            else:
                raise IOError("Loading checkpoint file failed!")
    return __model


def predict_batch(x, raw_text=True):
    from timeit import default_timer as timer
    load_default_prediction_model()
    vocab = __context.vocab
    class_names = vocab.class_names
    train_dir = __config['train_dir']
    ckpt = tf.train.get_checkpoint_state(train_dir)
    # Since this might be on a different FS mmodify the path
    # Todo: check for this
    model_filename = Path(ckpt.model_checkpoint_path).name
    model_file = os.path.join(train_dir, model_filename)
    __config['model_checkpoint_path'] = model_file
    __config['model_filename'] = model_filename
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            with tf.Session() as sess:
                if ckpt and ckpt.model_checkpoint_path:
                    m = Model(__config, is_train=False)
                    saver = tf.train.Saver(tf.global_variables(), allow_empty=True)
                    saver.restore(sess, model_file)
                    for i, xi in enumerate(x):
                        try:
                            if raw_text:
                                tid = vocab.text2id(xi)
                                if tid is None:
                                    yield ['inv', []]
                                xi = tid
                                x_input = np.array([xi])
                            else:
                                x_input = xi
                            # Predict
                            scores = sess.run(m.scores, feed_dict={m.inputs: x_input})

                            if raw_text:
                                scores = [float(str(i)) for i in scores[0]]
                                y_pred = class_names[int(np.argmax(scores))]
                                #scores = dict(zip(class_names, scores))
                            else:
                                y_pred = np.argmax(scores, axis=1)
                            yield [y_pred, scores]
                        except Exception as e:
                            yield ['inv', []]
                else:
                    raise IOError("Loading checkpoint file failed!")


def predict(x, raw_text=True):
    """ Build evaluation graph and run. """
    load_default_prediction_model()
    g = predict_batch([x], raw_text=raw_text)
    first = next(g)
    return first


def load_language_codes():
    ret = {}
    path = os.path.join(THIS_DIR, 'language_codes.tsv')
    with codecs_open(path, 'r', encoding=ENCODING) as f:
        for line in f.readlines():
            if not line.startswith('#') and len(line.strip()) > 0:
                c = line.strip().split('\t')
                if len(c) > 1:
                    ret[c[0]] = c[1]
    return ret


def _summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])


def get_config(train_dir="models", data_dir="data", batch_size=100, num_epoch=50, use_pretrain=False, vocab_size=4090,
               init_lr=0.01, log_step=10, summary_step=200, checkpoint_step=200,
               tolerance_step=500, lr_decay=0.95, emb_size=300, num_kernel=100, min_window=3, max_window=5,
               sent_len=257, l2_reg=1e-5, optimizer='adam', dropout=0.5):
    config = dict()
    config['train_dir'] = train_dir
    config['data_dir'] = data_dir
    config['batch_size'] = batch_size
    config['num_epoch'] = num_epoch
    config['use_pretrain'] = use_pretrain
    config['vocab_size'] = vocab_size
    config['init_lr'] = init_lr
    config['summary_step'] = summary_step
    config['checkpoint_step'] = checkpoint_step
    config['log_step'] = log_step
    config['tolerance_step'] = tolerance_step
    config['lr_decay'] = lr_decay
    config['emb_size'] = emb_size
    config['num_kernel'] = num_kernel
    config['min_window'] = min_window
    config['max_window'] = max_window
    config['sent_len'] = sent_len
    config['l2_reg'] = l2_reg
    config['optimizer'] = optimizer
    config['dropout'] = dropout
    return config


def train(train_dir="models", data_dir="data", batch_size=100, num_epoch=50, use_pretrain=False, vocab_size=4090,
          init_lr=0.01, log_step=10, summary_step=200, checkpoint_step=200,
          tolerance_step=500, lr_decay=0.95, emb_size=300, num_kernel=100, min_window=3, max_window=5,
          sent_len=257, l2_reg=1e-5, optimizer='adam', dropout=0.5):
    # train_dir
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(train_dir, timestamp))
    # abs_data_dir = os.path.abspath(os.path.join(data_dir, timestamp))
    # save flags
    if not os.path.exists(out_dir):
        mkdir_p(out_dir)
    config = get_config(train_dir, data_dir, batch_size, num_epoch, use_pretrain, vocab_size,
                        init_lr, log_step, summary_step, checkpoint_step, tolerance_step, lr_decay, emb_size,
                        num_kernel, min_window,
                        max_window, sent_len, l2_reg, optimizer, dropout)

    save(config, os.path.join(out_dir, 'flags.cPickle'))
    print("Parameters:")
    for k in config:
        print('%20s %r' % (k, config[k]))
    device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
    with tf.device(device):

        # load data
        print("Preparing train data ...")
        train_loader = DataLoader(data_dir, 'train.cPickle', batch_size=batch_size)
        print("Preparing test data ...")
        dev_loader = DataLoader(data_dir, 'test.cPickle', batch_size=batch_size)
        max_steps = train_loader.num_batch * num_epoch
        config['num_classes'] = train_loader.num_classes
        config['sent_len'] = train_loader.sent_len

        with tf.Graph().as_default():
            with tf.variable_scope('cnn', reuse=None):
                m = Model(config, is_train=True)
            with tf.variable_scope('cnn', reuse=True):
                mtest = Model(config, is_train=False)

            # checkpoint
            saver = tf.train.Saver(tf.global_variables())
            save_path = os.path.join(out_dir, 'model.ckpt')
            summary_op = tf.summary.merge_all()

            # session
            sess = tf.Session()

            # summary writer
            proj_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embedding = proj_config.embeddings.add()
            embedding.tensor_name = m.W_emb.name
            embedding.metadata_path = os.path.join(data_dir, 'metadata.tsv')
            summary_dir = os.path.join(out_dir, "summaries")
            summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, proj_config)

            sess.run(tf.global_variables_initializer())

            # assign pretrained embeddings
            if use_pretrain:
                print("Use pretrained embeddings to initialize model ...")
                emb_file = os.path.join(data_dir, 'emb.txt')
                vocab_file = os.path.join(data_dir, 'vocab.txt')
                pretrained_embedding = load_embedding(emb_file, vocab_file, vocab_size)
                m.assign_embedding(sess, pretrained_embedding)

            # initialize parameters
            current_lr = init_lr
            lowest_loss_value = float("inf")
            decay_step_counter = 0
            global_step = 0

            # evaluate on dev set
            def dev_step(mtest, sess, data_loader):
                dev_loss = 0.0
                dev_accuracy = 0.0
                for _ in range(data_loader.num_batch):
                    x_batch_dev, y_batch_dev = data_loader.next_batch()
                    dev_loss_value, dev_true_count = sess.run([mtest.total_loss, mtest.true_count_op],
                                                              feed_dict={mtest.inputs: x_batch_dev,
                                                                         mtest.labels: y_batch_dev})
                    dev_loss += dev_loss_value
                    dev_accuracy += dev_true_count
                dev_loss /= data_loader.num_batch
                dev_accuracy /= float(data_loader.num_batch * batch_size)
                data_loader.reset_pointer()
                return dev_loss, dev_accuracy

            # train loop
            print('\nStart training, %d batches needed, with %d examples per batch.' % (
                train_loader.num_batch, batch_size))
            for epoch in range(num_epoch):
                train_loss = []
                train_accuracy = []
                train_loader.reset_pointer()
                for _ in range(train_loader.num_batch):
                    m.assign_lr(sess, current_lr)
                    global_step += 1

                    start_time = time.time()
                    x_batch, y_batch = train_loader.next_batch()

                    feed = {m.inputs: x_batch, m.labels: y_batch}
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss_value, true_count = sess.run([m.train_op, m.total_loss, m.true_count_op],
                                                         feed_dict=feed, options=run_options, run_metadata=run_metadata)
                    proc_duration = time.time() - start_time
                    train_loss.append(loss_value)
                    train_accuracy.append(true_count)

                    assert not np.isnan(loss_value), "Model loss is NaN."

                    if global_step % log_step == 0:
                        examples_per_sec = batch_size / proc_duration
                        accuracy = float(true_count) / batch_size
                        format_str = '%s: step %d/%d (epoch %d/%d), acc = %.2f, loss = %.2f ' + \
                                     '(%.1f examples/sec; %.3f sec/batch), lr: %.6f'
                        print(format_str % (datetime.now(), global_step, max_steps, epoch + 1, num_epoch,
                                            accuracy, loss_value, examples_per_sec, proc_duration, current_lr))

                    # write summary
                    if global_step % summary_step == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_run_metadata(run_metadata, 'step%04d' % global_step)
                        summary_writer.add_summary(summary_str, global_step)

                        # summary loss/accuracy
                        train_loss_mean = sum(train_loss) / float(len(train_loss))
                        train_accuracy_mean = sum(train_accuracy) / float(len(train_accuracy) * batch_size)
                        summary_writer.add_summary(_summary('train/loss', train_loss_mean), global_step=global_step)
                        summary_writer.add_summary(_summary('train/accuracy', train_accuracy_mean),
                                                   global_step=global_step)

                        test_loss, test_accuracy = dev_step(mtest, sess, dev_loader)
                        summary_writer.add_summary(_summary('dev/loss', test_loss), global_step=global_step)
                        summary_writer.add_summary(_summary('dev/accuracy', test_accuracy), global_step=global_step)

                        print("\nStep %d: train_loss = %.6f, train_accuracy = %.3f" % (
                            global_step, train_loss_mean, train_accuracy_mean))
                        print(
                            "Step %d:  test_loss = %.6f,  test_accuracy = %.3f\n" % (
                                global_step, test_loss, test_accuracy))

                    # decay learning rate if necessary
                    if loss_value < lowest_loss_value:
                        lowest_loss_value = loss_value
                        decay_step_counter = 0
                    else:
                        decay_step_counter += 1
                    if decay_step_counter >= tolerance_step:
                        current_lr *= lr_decay
                        print('%s: step %d/%d (epoch %d/%d), Learning rate decays to %.5f' % \
                              (datetime.now(), global_step, max_steps, epoch + 1, num_epoch, current_lr))
                        decay_step_counter = 0

                    # stop learning if learning rate is too low
                    if current_lr < 1e-5:
                        break

                    # save checkpoint
                    if global_step % checkpoint_step == 0:
                        saver.save(sess, save_path, global_step=global_step)
            saver.save(sess, save_path, global_step=global_step)


# model_config = utils.load(os.path.join(THIS_DIR, 'models', 'ted500', 'flags.cPickle'))
# language_codes = load_language_codes()

#
#
# def main(argv=None):
#     FLAGS = tf.app.flags.FLAGS
#     this_dir = os.path.abspath(os.path.dirname(__file__))
#     tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data', 'ted500'), 'Directory of the data')
#     tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'model', 'ted500'), 'Where to read model')
#     FLAGS._parse_flags()
#
#     text = u"日本語のテスト。"
#     config = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
#     config['data_dir'] = FLAGS.data_dir
#     config['train_dir'] = FLAGS.train_dir
#
#     # predict
#     result = predict(text, config, raw_text=True)
#     language_codes = util.load_language_codes()
#     print 'prediction = %s' % language_codes[result['prediction']]
#     print 'scores = %s' % str({language_codes[k]: v for k, v in result['scores'].iteritems()})


if __name__ == '__main__':
    tf.app.run()
