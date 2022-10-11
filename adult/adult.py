import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
import numpy as np
import scipy
import scipy.optimize
import pdb
import tensorflow as tf
import argparse
import statistics
import sys



def data_preprocessing_v2(whole_data,train, test, max_len, max_words=50000):
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,filters='@')
	tokenizer.fit_on_texts(whole_data)
	train_idx = tokenizer.texts_to_sequences(train)
	test_idx = tokenizer.texts_to_sequences(test)
	train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
	test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
	return train_padded, test_padded, max_words + 2

def _save_weight(sess):
	global smallest_weight
	tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	smallest_weight = sess.run(tf_vars)

def _load_weights(sess):
	tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	ops = []
	for i_tf in range(len(tf_vars)):
		ops.append(tf.assign(tf_vars[i_tf], smallest_weight[i_tf]))
	sess.run(ops)

def classifier(s,dec_rate):
	print(s)
	tf.random.set_random_seed(s)
	np.random.seed(s)
	overall_numbers = {}
	results = []
	test_acc= []

	train_all_text =[]
	train_all_label=[]

	test_all_text =[]
	test_all_label=[]

	dev_all_text =[]
	dev_all_label=[]

	index_male_test = []
	index_female_test = []

	test_counter =0

	fairness_results = []

	train_inputfile = open("train_text_adult_features","r")
	train_lablefile = open("train_text_adult_label_features","r")

	test_inputfile = open("test_text_adult_features","r")
	test_lablefile = open("test_text_adult_label_features","r")

	dev_inputfile = open("dev_text_adult_features","r")
	dev_lablefile = open("dev_text_adult_label_features","r")

	for i in train_inputfile.readlines():
		train_all_text.append(i.strip("\n"))

	for i in train_lablefile.readlines():
		train_all_label.append(int(i.strip("\n")))

	for i in test_inputfile.readlines():
		test_all_text.append(i.strip("\n"))
		splits = i.split(" ")
		if(splits[9]=="Male"):
			index_male_test.append(test_counter)
		elif(splits[9]=="Female"):
			index_female_test.append(test_counter)
		test_counter +=1

	for i in test_lablefile.readlines():
		test_all_label.append(int(i.strip("\n")))

	for i in dev_inputfile.readlines():
		test_all_text.append(i.strip("\n"))
		splits = i.split(" ")
		if(splits[9]=="Male"):
			index_male_test.append(test_counter)
		elif(splits[9]=="Female"):
			index_female_test.append(test_counter)
		test_counter +=1

	for i in dev_lablefile.readlines():
		test_all_label.append(int(i.strip("\n")))

	train_text = pd.Series(train_all_text)
	test_text = pd.Series(test_all_text)
	whle_text_data = train_all_text+test_all_text
	whole_data = pd.Series(whle_text_data)

	X_train, X_test, vocab_size = data_preprocessing_v2(whole_data,train_text,test_text,max_len=14)


	train_label =train_all_label
	test_label =test_all_label

	Y_train = np.max(train_label) + 1
	Y_train =np.eye(Y_train)[train_label]


	Y_test = np.max(test_label) + 1
	Y_test =np.eye(Y_test)[test_label]



	DIM_INPUT = X_train.shape[1]
	DIM_HIDDEN = 256
	DIM_OUTPUT = 2

	X_placeholder = tf.placeholder(tf.int32, [None, 14])
	Y_placeholder = tf.placeholder(tf.float32, [None, DIM_OUTPUT])
	keep_prob = tf.placeholder(tf.float32)
	mask = tf.placeholder(tf.float32,[None,14])

	embeddings_var = tf.Variable(tf.random_uniform([vocab_size, 128], -1.0, 1.0,seed=s),trainable=True)
	batch_embedded = tf.nn.embedding_lookup(embeddings_var, X_placeholder)

	
	attn_W = tf.Variable(tf.random_normal([128], stddev=0.1,seed=s))
	#attn_W = v_placeholder
	attn_H = batch_embedded
	attn_M = tf.tanh(attn_H)

	alpha = tf.math.multiply( tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(attn_M, [-1, 128]),
                                                        tf.reshape(attn_W, [-1, 1])),
                                              (-1, X_train.shape[1])) )  ,mask  )# batch_size x seq_len

	attn_r = tf.matmul(tf.transpose(attn_H, [0, 2, 1]),
                      tf.reshape(alpha, [-1, X_train.shape[1], 1]))

	attn_r = tf.squeeze(attn_r)
	h_star = tf.tanh(attn_r)

	y_hat_init =tf.layers.dense(tf.reshape(h_star,[-1,128]),50, activation=tf.nn.relu)
	y_hat =tf.layers.dense(y_hat_init,2, activation=None)

	loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=Y_placeholder))

	prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
	diff = tf.to_float(prediction) - Y_placeholder[:, 1]
	accuracy = 1 - tf.math.reduce_mean(tf.math.abs(diff))

	loss_to_minimize = loss
	tvars = tf.trainable_variables()

	gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

	grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

	global_step = tf.Variable(0, name="global_step", trainable=False)

	lr = tf.Variable(1e-3, name='lr', trainable=False)
	lr_decay_op = lr.assign(lr * 0.95)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)

	train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step,
                                                       name='train_step')

	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		wait = 0
		smallest_loss_total_dev = float('inf')
		smallest_weight = None
		patience_lr_decay = 5
		patience_wait = 100

		for epoch in range(100):
			train_loss,train_op_train,train_embeddings_var= sess.run(
				[loss,train_op,embeddings_var],
					feed_dict={
						X_placeholder: X_train,
						Y_placeholder: Y_train,
						keep_prob: 0.5,
						mask: np.ones([X_train.shape[0],X_train.shape[1]])
					}
			)
			print("train loss is")
			print(train_loss)
		
		_save_weight(sess)


		_load_weights(sess)
		loss_test,acc_test,test_pred,test_alpha = sess.run(
				[loss,accuracy,prediction,alpha],
					feed_dict={
						X_placeholder: X_test,
						Y_placeholder: Y_test,
						keep_prob: 1.0,
						mask: np.ones([X_test.shape[0],X_test.shape[1]])
					}
		)

		loss_train,acc_train,train_alpha = sess.run(
				[loss,accuracy,alpha],
					feed_dict={
						X_placeholder: X_train,
						Y_placeholder: Y_train,
						keep_prob: 1.0,
						mask: np.ones([X_train.shape[0],X_train.shape[1]])
					}
			)
		print("train accuracy is")
		print(acc_train)

		print("test accuracy is")
		print(acc_test)
		test_female_one_prediction = np.where(test_pred[index_female_test] == 1)[0].astype(np.int32)
		test_male_one_prediction = np.where(test_pred[index_male_test] == 1)[0].astype(np.int32)
		p_one_g_female = test_female_one_prediction.shape[0]/len(index_female_test)
		p_one_g_male = test_male_one_prediction.shape[0]/len(index_male_test)
		print(abs(p_one_g_female-p_one_g_male))
		original_fairness_results = abs(p_one_g_female-p_one_g_male)
		original_alpha = test_alpha
		for i in range(14):
			temp_mask = np.ones([X_test.shape[0],X_test.shape[1]])
			temp_mask[:,i]=0
			tt_temp = np.multiply(temp_mask,original_alpha)
			norm_alpha = np.linalg.norm(tt_temp,ord=1,axis=1)
			norm_alpha = np.reciprocal(norm_alpha)
			norm_alpha = np.repeat(np.expand_dims(norm_alpha,axis=1),X_test.shape[1],axis=1)
			normalized_temp_mask = np.multiply(temp_mask,norm_alpha)

			loss_test,acc_test,test_pred,test_alpha = sess.run(
					[loss,accuracy,prediction,alpha],
						feed_dict={
							X_placeholder: X_test,
							Y_placeholder: Y_test,
							keep_prob: 1.0,
							mask: normalized_temp_mask
						}
			)
			test_female_one_prediction = np.where(test_pred[index_female_test] == 1)[0].astype(np.int32)
			test_male_one_prediction = np.where(test_pred[index_male_test] == 1)[0].astype(np.int32)
			p_one_g_female = test_female_one_prediction.shape[0]/len(index_female_test)
			p_one_g_male = test_male_one_prediction.shape[0]/len(index_male_test)
			print("*********************************")
			print(acc_test)
			print(abs(p_one_g_female-p_one_g_male))
			fairness_results.append(abs(p_one_g_female-p_one_g_male))

		temp_mask = np.ones([X_test.shape[0],X_test.shape[1]])
		for i in range(14):
			if fairness_results[i] <= original_fairness_results:
				temp_mask[:,i]=dec_rate
		

		tt_temp = np.multiply(temp_mask,original_alpha)
		norm_alpha = np.linalg.norm(tt_temp,ord=1,axis=1)
		norm_alpha = np.reciprocal(norm_alpha)
		norm_alpha = np.repeat(np.expand_dims(norm_alpha,axis=1),X_test.shape[1],axis=1)
		normalized_temp_mask = np.multiply(temp_mask,norm_alpha)
		loss_test,acc_test,test_pred,test_alpha = sess.run(
					[loss,accuracy,prediction,alpha],
						feed_dict={
							X_placeholder: X_test,
							Y_placeholder: Y_test,
							keep_prob: 1.0,
							mask: normalized_temp_mask
						}
			)
		test_female_one_prediction = np.where(test_pred[index_female_test] == 1)[0].astype(np.int32)
		test_male_one_prediction = np.where(test_pred[index_male_test] == 1)[0].astype(np.int32)
		p_one_g_female = test_female_one_prediction.shape[0]/len(index_female_test)
		p_one_g_male = test_male_one_prediction.shape[0]/len(index_male_test)
		print("*********************************")
		print(acc_test)
		print(abs(p_one_g_female-p_one_g_male))





if __name__=="__main__":
	s = sys.argv[1]
	dec_rate = sys.argv[2]
	print(s)
	classifier(int(s),float(dec_rate))

	
