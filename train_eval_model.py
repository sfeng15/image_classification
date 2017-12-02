import tensorflow as tf
import numpy as np
import os
import pandas as pd
import datetime
from utils import io_tools


def create_submission(predictions, test_id):
    result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                'c4', 'c5', 'c6', 'c7',
                                                'c8', 'c9'])
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result.to_csv(sub_file, index=False)


def train_model(model,learning_rate, batch_size,
                training_epoch, keep_prob):
    data_txt_file='data/driver_imgs_list.csv'
    image_data_path='data/train'
    rows = 224
    cols = 224

    saver = tf.train.Saver()
    # load saved model if any
    #if os.path.isfile('models/model.ckpt'):
    #    load_path = saver.restore(model.session, 'models/model.ckpt')

    files = io_tools.load_train(
            data_txt_file, image_data_path)

    for epoch in range(training_epoch):

        # read 32 images at a time
        for i in range(0, len(files), 32):
            batch_x = []
            batch_y = []

            cur_batch_files = files[i: i+32]
            if len(cur_batch_files) != 32:
                continue
            for line in cur_batch_files:
                line = line.strip()
                path = image_data_path + '/' + str(line.split(',')[1]) + '/' + str(line.split(',')[2])
                batch_x.append(io_tools.read_image(path, rows, cols))
                batch_y.append(int(line.split(',')[1][1]))

            batch_x = np.array(batch_x, dtype=np.float32)
            batch_y = np.array(batch_y, dtype=np.float32)

            batch_y = tf.keras.utils.to_categorical(batch_y, 10)

            mean_pixel = [103.939, 116.779, 123.68]

            for c in range(3):
                batch_x[:, :, :, c] = batch_x[:, :, :, c] - mean_pixel[c]
 

            _, loss, accuracy_train, ts = model.session.run(
                [model.update_op_tensor, model.loss_tensor, model.accuracy_tensor, model.outputs_tensor],
                feed_dict={model.x_placeholder: batch_x, model.y_placeholder: batch_y,
                           model.learning_rate_placeholder: learning_rate, model.keep_prob_placeholder: keep_prob, model.phase_train: True}
            )

            print(ts.shape)
            print("drodout %f, epoch %d: training accuracy = %f, loss = %f" %
                  (keep_prob, epoch, accuracy_train, loss))

    # save model
    save_path = saver.save(model.session, 'models/model.ckpt')
    print("Model saved in file: %s" % save_path)

    return model


def eval_model(model, x_test, x_test_id, batch_size=32):
    N = x_test.shape[0]
    batch_epoch_num = N // batch_size
    yfull_test = []

    for k in range(batch_epoch_num):
        start = k * batch_size
        batch_test = x_test[start:start + batch_size]
        prediction = model.session.run(model.outputs_tensor, feed_dict={
            model.x_placeholder: batch_test, model.keep_prob_placeholder: 1})
        yfull_test.append(prediction)

    if N % batch_size != 0:
        batch_test = x_test[batch_epoch_num * batch_size:]
        prediction = model.session.run(model.outputs_tensor, feed_dict={
            model.x_placeholder: batch_test, model.keep_prob_placeholder: 1, model.phase_train: False})
        yfull_test.append(prediction)

    create_submission(yfull_test, x_test_id)
