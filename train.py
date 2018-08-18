import os
import datetime

import tensorflow as tf
import numpy as np

from model import Model


def load_or_create_model(sess, model, saver, model_dir):

    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters...')
        model.restore(sess, saver, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print('Created new model parameters...')
        sess.run(tf.global_variables_initializer())


def main():
    break_flag = 0
    model = Model()
    inputs, decode_topic, decode_sen = model.build()

    saver = tf.train.Saver()
    sess = tf.Session()

    load_or_create_model(sess, model, saver, model.model_path)
    step = model.global_step.eval(session=sess)

    new_state = sess.run(inputs['initial_state'])

    for epoch in range(model.max_epoch):
        with open(model.logs_file, 'a') as f:
            f.write("current epoch:{}\n".format(str(epoch)))

        batches = model.data_utils.prepare_batch()
        for encode, decode_topic_label, decode_sen_input, decode_sen_label, encode_length, decode_sen_length in batches:

            if len(encode) != model.batch_size:
                continue

            feed = {inputs['encode']: encode,
                    inputs['decode_topic_label']: decode_topic_label,
                    inputs['decode_sen_input']: decode_sen_input,
                    inputs['decode_sen_label']: decode_sen_label,
                    inputs['encode_length']: encode_length,
                    inputs['decode_sen_length']: decode_sen_length,
                    inputs['initial_state']: new_state
                    }

            _, decode_topic_loss, decode_topic_accuracy, _, decode_sen_loss = sess.run(
                [decode_topic['decode_topic_optimizer'],
                 decode_topic['decode_topic_loss'],
                 decode_topic['decode_topic_accuracy'],
                 decode_sen['decode_sen_optimizer'],
                 decode_sen['decode_sen_loss'],
                 ],
                feed_dict=feed)

            decode_perplexity = np.exp(decode_sen_loss)

            # Write into log file
            with open(model.logs_file, 'a') as f:
                f.write("{}\tepoch:{}\tstep:{}\tdecode_topic_loss:{}\tdecode_topic_accuracy:{}\tdecode_sen_loss:{}\tdecode_perplexity:{}\n".format(
                    datetime.datetime.now().strftime('%c'),
                    str(epoch),
                    str(step),
                    str(decode_topic_loss),
                    str(decode_topic_accuracy),
                    str(decode_sen_loss),
                    str(decode_perplexity))
                )

            # Print result
            if step % model.print_step == 0:
                print("{}\tepoch:{}\tstep:{}\tpre_loss:{}\tpre_accuracy:{}\tpost_loss:{}\tperplexity:{}\n".format(
                    datetime.datetime.now().strftime('%c'),
                    str(epoch),
                    str(step),
                    str(decode_topic_loss),
                    str(decode_topic_accuracy),
                    str(decode_sen_loss),
                    str(decode_perplexity))
                )

            if decode_topic_loss < model.end_loss and decode_sen_loss < model.end_loss:
                break_flag = 1
                break

            step += 1
            model.increment_global_step_op.eval(session=sess)

        if epoch % model.save_epoch == 0:
            model_path = os.path.join(model.model_path, model.save_model_name)
            saver.save(sess, model_path, global_step=step)

        if break_flag == 1:
            break

    model_path = os.path.join(model.model_path, model.save_model_name)
    saver.save(sess, model_path, global_step=step)
    sess.close()


if __name__ == '__main__':
    main()
