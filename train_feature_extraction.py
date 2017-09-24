import pickle
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

number_of_classes = 43
BATCH_SIZE = 128
N_EPOCHS = 3

# TODO: Load traffic signs data.

with open('./train.p', 'rb') as f:
    data = pickle.load(f)
# note: data is a dictionary with keys: dict_keys(['coords', 'features', 'sizes', 'labels'])

# TODO: Split data into training and validation sets.
x_train, x_test, y_train, y_test = train_test_split(data['features'], data['labels'])

# TODO: Define placeholders and resize operation.
original_features_placeholder = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized_features_placeholder = \
    tf.image.resize_images(original_features_placeholder, (227, 227))

labels_placeholder = tf.placeholder(tf.int64, None)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(features=resized_features_placeholder, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], number_of_classes)
fc8W = tf.Variable(tf.truncated_normal(shape=shape))
fc8b = tf.Variable(tf.zeros(shape=number_of_classes))
fc8_logits = tf.nn.xw_plus_b(x=fc7, weights=fc8W, biases=fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc8_logits, labels=labels_placeholder)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=.001)
training_operation = optimizer.minimize(loss_operation)

preds = tf.arg_max(fc8_logits, 1)

accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels_placeholder), tf.float32))

init = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.

def evaluate(X_data, y_data, the_session):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, loss = the_session.run([accuracy_op, loss_operation],
                                             feed_dict={
                                                 original_features_placeholder: batch_x,
                                                 labels_placeholder: batch_y
                                             }
                                         )
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples

with tf.Session() as sess:
    sess.run(init)

    train_length = x_train.shape[0]

    for i in range(N_EPOCHS):
        x_train, x_test = shuffle(x_train, y_train)
        le_time = time.time()
        for offset in range(0, train_length, BATCH_SIZE):
            end = offset + BATCH_SIZE
            print(y_train[offset:end].shape)
            sess.run(fetches=training_operation,
                     feed_dict={
                         original_features_placeholder: x_train[offset:end],
                         labels_placeholder: y_train[offset:end]
                     })
        validation_accuracy, validation_loss = evaluate(X_data=x_test, y_data=y_train, the_session=sess)
        print("Epoch", i + 1)
        print("Time: %.3f seconds" % (time.time() - le_time))
        print("Validation Loss =", validation_accuracy)
        print("Validation Accuracy =", validation_loss)
        print("")