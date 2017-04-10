import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(666)

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,10])


#W1 = tf.Variable(tf.random_normal([784,256]))
#W2 = tf.Variable(tf.random_normal([256,256]))
#W3 = tf.Variable(tf.random_normal([256,10]))

W1 = tf.get_variable("W1", shape=[784,256], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))

# Construct model

#L1 = tf.nn.relu(tf.add(tf.matmul( X,W1),B1))
#L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B1))
#hypothesis = tf.add(tf.matmul(L2,W3),B3)

dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul( X,W1),B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3),B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4),B4))
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4,W5),B5)



#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

activation = tf.nn.softmax(hypothesis)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#print("Accyraccy:", accuracy.eval({X:mnist.test.images, Y:mnist.test.labels,dropout_rate:1}))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #sess.run(optimizer, feed_dict={X:batch_xs, Y:batch_ys})
            sess.run(optimizer, feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:0.7})
            avg_cost += sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:0.7})/total_batch
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))














#dropout (keep_prob) rate 0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
