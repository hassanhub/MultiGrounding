import tensorflow as tf
import os
import sys
import cv2
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
from utils import *
import lmdb
import random
import pickle

#plans: 2. put validators in a separate class or file
#plans: 7. take care of build, save, and load stuff of the entire model (including elmo)
#plans: 8. proper name scopes and attention function (for inference)
#plans: 9. proper requirements.txt file
#plans: 10. proper readme.md with examples
#plans: 12. prepare examples for demonstrating how to use each block/function/class
#plans: 13. proper documentation/comments in the files
#plans: 14. add debugging ability, summary of loss and level index
#plans: 15. add validation option
#plans: 16. make sure initializer doesn't override pnas and elmo
#plans: 17. clean all deprecated tf ops
#plans: 18. clean all checkpoints and their namings
#plans: 19. add option of training from 'jpg'+'txt'
#plans: 20. add option of fine-tuning

class GroundNet():
	def __init__(self,
				 gnet_config=None,
				 kernel_initializer=None,
				 reg_scale=0.0005,
				 gamma_1 = 5.0,
				 gamma_2 = 10.0,
				 debugging = False,
				 **kwargs):
	
		if not gnet_config:
			gnet_config = {'image_model': 'pnasnet_large',
			                'text_model': 'ELMo',
			                'conv_type': '3x3',
			                'image_conv_filters': [1024],
			                'word_fc_units': [1024,1024],
			                'sentence_fc_units': [1024,1024],
			                'image_size': [299,299],
			                'visual_layers': ['Cell_5', 'Cell_7', 'Cell_9', 'Cell_11']}

		self.gnet_config = gnet_config
		self.var_init = False
		self.saver = None
		self.regularizer = tf.contrib.layers.l2_regularizer(reg_scale)
		self.gamma_1 = gamma_1
		self.gamma_2 = gamma_2
		self.debugging = debugging
		self.resize_method = tf.image.ResizeMethod.BILINEAR
		self.end_points = {}
		self.image_size = self.gnet_config['image_size']
		self.elmo = kwargs.get('elmo')

		#with tf.variable_scope('GroundNet'):
		#	self._build_gnet()

		#if self.debugging:
		#	self.train_writer = tf.summary.FileWriter('./logs/', self.sess.graph)
		#	self.merged = tf.summary.merge_all()

		#self.train_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - set(self.vis_model.model_weights))

	def __call__(self,image_input,text_input,training):
		with tf.variable_scope('GroundNet', reuse=tf.AUTO_REUSE):
			self._build_gnet(image_input,text_input,training)
		return self.end_points
	
	def _build_gnet(self,image_input,text_input,training):
		#building visual model
		print('Building Visual Model...')
		if 'vgg' in self.gnet_config['image_model']:
		  processing_mode = 'vgg_preprocessing'
		else:
		  processing_mode = 'inception_preprocessing'

		with tf.variable_scope('image_block'):
			pre_processed_img = pre_process(image_input, processing_mode)
			self.vis_model = pre_trained_load(model_name=self.gnet_config['image_model'], 
											  image_shape=(None,self.image_size[0],self.image_size[1],3),
											  input_tensor=pre_processed_img)

		def _ELMo_aggregate(wrd,lstm1,lstm2):
			W = tf.get_variable('Aggr_W',
								shape=(3, ),
								initializer=tf.zeros_initializer,
								regularizer=self.regularizer,
								trainable=training)

			# normalize the weights
			normed_weights = tf.split(tf.nn.softmax(W + 1.0 / 3), 3)
			# split LM layers
			layers = [wrd,lstm1,lstm2]
	
			# compute the weighted, normalized LM activations
			pieces = []
			for w, l in zip(normed_weights, layers):
				pieces.append(w * l)
			elmo_w_embd = tf.add_n(pieces)
			return elmo_w_embd

		#building text model
		print('Building Text Model...')
		with tf.variable_scope('text_block'):
			#loading pre-trained ELMo
			if not self.elmo:
				self.elmo = hub.Module("./models/text_models/ELMo_Module/", trainable=False, name='ELMo')
			#getting ELMo embeddings
			elmo_embds = self.elmo(text_input, signature="default", as_dict=True)
			#taking index of last word in each sentence
			idx = elmo_embds['sequence_len']-1

			with tf.variable_scope('text_post_process'):
				batch_idx = tf.stack([tf.range(0,tf.size(idx),1),idx],axis=1)

				if self.gnet_config['text_model']=='ELMo':
				  lstm1_embd = elmo_embds['lstm_outputs1'] #?xTXD
				  lstm2_embd = elmo_embds['lstm_outputs2'] #?xTXD
				  wrd_embd = elmo_embds['word_emb'] #?xTxD/2
				  e_w = tf.identity(_ELMo_aggregate(tf.tile(wrd_embd,[1,1,2]),lstm1_embd,lstm2_embd), name='elmo_word_embd') #?xTXD
				  # Concatenate first of backward with last of forward to get sentence embeddings
				  dim = lstm1_embd.get_shape().as_list()[-1]
				  sen_embd_1 = tf.concat([lstm1_embd[:,0,int(dim/2):],
										  tf.gather_nd(lstm1_embd[:,:,:int(dim/2)],batch_idx)], axis=-1) #[batch,dim]
				  sen_embd_2 = tf.concat([lstm2_embd[:,0,int(dim/2):],
										  tf.gather_nd(lstm2_embd[:,:,:int(dim/2)],batch_idx)], axis=-1) #[batch,dim]
				  sen_embd = tf.concat([tf.expand_dims(sen_embd_1,axis=2),
											 tf.expand_dims(sen_embd_2,axis=2)], axis=2, name='elmo_sen_embd') #[batch,dim,2]
				  sen_embd = tf.layers.dense(sen_embd,units=1, use_bias=False) #?xDx1
				  e_s = tf.squeeze(sen_embd,axis=2)
				else:
				  w_embd = tf.identity(elmo_embds['word_emb'], name='elmo_word_embd') #?xTxD/2
				  lstm_embd = self._build_bilstm(w_embd,elmo_embds['sequence_len']) #?xTxD
				  #taking index of last word in each sentence
				  idx = elmo_embds['sequence_len']-1
				  batch_idx = tf.stack([tf.range(0,tf.size(idx),1),idx],axis=1)
				  # Concatenate first of backward with last of forward to get sentence embeddings
				  dim = lstm_embd.get_shape().as_list()[-1]
				  e_s = tf.concat([lstm_embd[:,0,int(dim/2):],
										  tf.gather_nd(lstm_embd[:,:,:int(dim/2)],batch_idx)], axis=-1) #[batch,dim]
				  w_embd_tiled = tf.tile(w_embd,[1,1,2])
				  w_embd = tf.concat([tf.expand_dims(w_embd_tiled,axis=3),tf.expand_dims(lstm_embd,axis=3)],axis=3)
				  e_w = tf.layers.dense(w_embd, units=1)[:,:,:,0]


		print('Common space mapping and attention...')
		with tf.variable_scope('mapping'):
			with tf.variable_scope('image_mapping'):
				v = self._image_mapping(model=self.vis_model, n_conv_filters=self.gnet_config['image_conv_filters'])

			with tf.variable_scope('text_mapping'):
				e_w, e_s = self._text_mapping(e_w, e_s)

		with tf.variable_scope('attention'):
			self.end_points.update(self._build_attention(e_w,v,e_s))
		
		self.end_points['stacked_v'] = v
		self.end_points['sen_embedding'] = e_s
		self.end_points['wrd_embedding'] = e_w
		print('Model built successfully.')

	def _build_bilstm(self,w_embd,seq_length):
		with tf.variable_scope('BiLSTM'):
			# Forward direction cell
			lstm_fw_cell = tf.contrib.rnn.LSTMCell(512, forget_bias=1.0)
			# Backward direction cell
			lstm_bw_cell = tf.contrib.rnn.LSTMCell(512, forget_bias=1.0)
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, w_embd, sequence_length = seq_length,
													  dtype=tf.float32)
		output = tf.concat(outputs,axis=2,name='BiLSTM_out')
		return output

	def _build_attention(self,e_w,v,e_s):
		## Input:
		##			1. Word embeddings: e_w ?xTxD
		##			2. Sentence embeddings: e_s ?xD
		##			3. Image features at different levels: v ?xNx4xD
		## Output: 
		##			1. Heatmap for each word and sentence at different levels
		##			2. Scores for word and sentnce at different levels
		##			3. Chosen levels for each word and sentnce
		##			4. Heatmap for chosen level for each word and sentence

		attn_outs = {}
		###word-level###
		with tf.variable_scope('word_level'):
			with tf.variable_scope('heatmap'):
				#heatmap pool
				h = tf.nn.relu(tf.einsum('bij,bklj->bikl',e_w,v)) #pair-wise ev^T: ?xTxNx4

			with tf.variable_scope('attend'):
				#attention
				a = tf.einsum('bijk,bjkl->bilk',h,v) #?xTxDx4 attnded visual reps for each of T words
				a = tf.nn.l2_normalize(a,axis=2)

			with tf.variable_scope('score'):
				#pair-wise score calculation
				R_ik = tf.einsum('bilk,bil->bik',a,e_w) #cosine for T (words,img_reps) for all pairs
				end_point = 'level_score_word'
				R_ik = tf.identity(R_ik,name=end_point)
				attn_outs[end_point] = R_ik
				end_point = 'score_word'
				R_i = tf.reduce_max(R_ik,axis=-1,name=end_point) #?xT
				attn_outs[end_point] = R_i
				#R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_1*R_i),axis=1),1/gamma_1)) #? corrs
			
			with tf.variable_scope('level_selection'):
				#level selection
				end_point = 'level_index_word'
				idx_i = tf.argmax(R_ik,axis=-1,name=end_point) #?xT index of the featuremap which maximizes R_i
				attn_outs[end_point] = idx_i
				ii,jj = tf.meshgrid(tf.range(tf.shape(idx_i)[0]),tf.range(tf.shape(idx_i)[1]),indexing='ij')
				ii = tf.cast(ii,tf.int64)
				jj = tf.cast(jj,tf.int64)
				batch_idx_i = tf.stack([tf.reshape(ii,(-1,)),
									  tf.reshape(jj,(-1,)),
									  tf.reshape(idx_i,(-1,))],axis=1) #?Tx3 indices of argmax
			
			with tf.variable_scope('heatmap_selection'):
				#heatmap selection
				h_max = tf.gather_nd(tf.transpose(h,[0,1,3,2]),batch_idx_i) #?TxN retrieving max heatmaps
				N0=int(np.sqrt(h.get_shape().as_list()[2]))
				end_point = 'heatmap_word'
				heatmap_wd = tf.reshape(h_max,[tf.shape(h)[0],tf.shape(h)[1],N0,N0],name=end_point)
				attn_outs[end_point] = heatmap_wd
				end_point = 'level_heatmap_word'
				heatmap_wd_l = tf.reshape(h,[tf.shape(h)[0],tf.shape(h)[1],N0,N0,tf.shape(h)[3]],name=end_point)
				attn_outs[end_point] = heatmap_wd_l

		###sentence-level###
		with tf.variable_scope('sentence_level'):
			with tf.variable_scope('heatmap'):
				#heatmap pool
				h_s = tf.nn.relu(tf.einsum('bj,blkj->blk',e_s,v)) #pair-wise e_bar*v^T: ?xNx4

			with tf.variable_scope('attend'):
				#attention
				a_s = tf.einsum('bjk,bjki->bik',h_s,v) #?xDx4 attnded visual reps for sen.
				a_s = tf.nn.l2_normalize(a_s,axis=1)

			with tf.variable_scope('score'):
				#pair-wise score
				R_sk = tf.einsum('bik,bi->bk',a_s,e_s) #cosine for (sen,img_reps)
				end_point = 'level_score_sentence'
				R_sk = tf.identity(R_sk,name=end_point)
				attn_outs[end_point] = R_sk
				end_point = 'score_sentence'
				R_s = tf.reduce_max(R_sk,axis=-1,name=end_point) #?
				attn_outs[end_point] = R_s

			with tf.variable_scope('level_selection'):
				#level selection
				end_point = 'level_index_sentence'
				idx_k = tf.argmax(R_sk,axis=-1,name=end_point) #? index of the featuremap which maximizes R_i
				attn_outs[end_point] = idx_k
				ii_k = tf.cast(tf.range(tf.shape(idx_k)[0]),dtype='int64')
				batch_idx_k = tf.stack([ii_k,idx_k],axis=1)
			
			with tf.variable_scope('heatmap_selection'):
				#heatmap selection
				h_s_max = tf.gather_nd(tf.transpose(h_s,[0,2,1]),batch_idx_k) #?xN retrieving max heatmaps
				N0_g=int(np.sqrt(h_s.get_shape().as_list()[1]))
				end_point = 'heatmap_sentence'
				heatmap_sd = tf.reshape(h_s_max,[-1,N0_g,N0_g],name='heatmap_sentence')
				attn_outs[end_point] = heatmap_sd
				end_point = 'level_heatmap_sentence'
				heatmap_sd_l = tf.reshape(h_s,[tf.shape(h_s)[0],N0_g,N0_g,tf.shape(h_s)[2]],name=end_point)
				attn_outs[end_point] = heatmap_sd_l

		return attn_outs

	def _add_conv(self,
				feat_map,
				n_filters,
				name,
				regularizer):

		assert self.gnet_config['conv_type'] in ['1x1', '3x3']
		if self.gnet_config['conv_type']=='3x3':
		  kernel_size = [3,3]
		elif self.gnet_config['conv_type']=='1x1':
		  kernel_size = [1,1]

		with tf.variable_scope(name+'_postConv'):
			for filters in n_filters:
				feat_map = tf.layers.conv2d(inputs=feat_map,
											filters=filters,
											kernel_size=kernel_size,
											padding='SAME',
											kernel_regularizer=regularizer)
				feat_map = tf.nn.leaky_relu(feat_map,alpha=.25)
		return feat_map

	def _image_mapping(self,
					   model,
					   n_conv_filters=[1024]):

		#getting biggest featuremap
		size_max = [0,0]
		for layer in self.gnet_config['visual_layers']:
			size = model[layer].get_shape().as_list()[1:3]
			if size[0]*size[1]>size_max[0]*size_max[1]:
				size_max = size
		v_list = []

		#building up common space mapping
		for n,layer in enumerate(self.gnet_config['visual_layers']):
			v = tf.identity(model[layer],name='v'+str(n+1))
			size = v.get_shape().as_list()[1:3]
			if size!=size_max:
				v = tf.image.resize_images(v, size_max, method=self.resize_method)
			v = self._add_conv(v,n_filters=n_conv_filters,name='v'+str(n+1),regularizer=self.regularizer)
			v_list.append(v)

		v_all = tf.stack(v_list, axis=3)
		v_all = tf.reshape(v_all,[-1,v_all.shape[1]*v_all.shape[2],v_all.shape[3],v_all.shape[4]])
		v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
		return v_all

	def _text_mapping(self,
					  e_w, 
					  e_s):

		with tf.variable_scope('word_level'):
			for units in self.gnet_config['word_fc_units']:
				e_w = tf.layers.dense(e_w,units=units)
				e_w = tf.nn.leaky_relu(e_w,alpha=.25)
				e_w = tf.nn.l2_normalize(e_w, axis=-1, name='wrd_embedding')

		with tf.variable_scope('sentence_level'):
			for units in self.gnet_config['sentence_fc_units']:
				e_s = tf.layers.dense(e_s,units=units)
				e_s = tf.nn.leaky_relu(e_s,alpha=.25)
				e_s = tf.nn.l2_normalize(e_s, axis=-1, name='sen_embedding')

		return e_w, e_s

	def matching_loss(self,e_w,v,e_s):
		#e: ?xTxD, v: ?xNx4xD, e_bar: ?xD
		with tf.variable_scope('attention_loss'):
			###word-level###
			#heatmap
			h = tf.nn.relu(tf.einsum('bij,cklj->bcikl',e_w,v)) #pair-wise ev^T: ?x?xTxNx4
			#attention
			a = tf.einsum('bcijl,cjlk->bcikl',h,v) #?x?xTxDx4 attnded visual reps for each of T words for all pairs
			a = tf.nn.l2_normalize(a,axis=3)
			#pair-wise score
			R_ik = tf.einsum('bcilk,bil->bcik',a,e_w) #cosine for T (words,img_reps) for all pairs
			R_i = tf.reduce_max(R_ik,axis=-1) #?x?xT
			R = tf.log(tf.pow(tf.reduce_sum(tf.exp(self.gamma_1*R_i),axis=2),1/self.gamma_1)) #?x? cap-img pairs
			#posterior probabilities
			P_DQ = tf.diag_part(tf.nn.softmax(self.gamma_2*R,axis=0)) #P(cap match img)
			P_QD = tf.diag_part(tf.nn.softmax(self.gamma_2*R,axis=1)) #p(img match cap)
			#losses
			L1_w = -tf.reduce_mean(tf.log(P_DQ))
			L2_w = -tf.reduce_mean(tf.log(P_QD))

			###sentence-level###
			#heatmap
			h_s = tf.nn.relu(tf.einsum('bj,cklj->bckl',e_s,v)) #pair-wise e_bar*v^T: ?x?xNx4
			#attention
			a_s = tf.einsum('bcjk,cjkl->bclk',h_s,v) #?x?xDx4 attnded visual reps for sen. for all pairs
			a_s = tf.nn.l2_normalize(a_s,axis=2)
			#pair-wise score
			R_sk = tf.einsum('bclk,bl->bck',a_s,e_s) #cosine for (sen,img_reps) for all pairs
			R_s = tf.reduce_max(R_sk,axis=-1) #?x?
			#posterior probabilities
			P_DQ_s = tf.diag_part(tf.nn.softmax(self.gamma_2*R_s,axis=0)) #P(cap match img)
			P_QD_s = tf.diag_part(tf.nn.softmax(self.gamma_2*R_s,axis=1)) #P(img match cap)
			#losses
			L1_s = -tf.reduce_mean(tf.log(P_DQ_s))
			L2_s = -tf.reduce_mean(tf.log(P_QD_s))
			#overall loss
			loss = L1_w + L2_w + L1_s + L2_s

		return loss