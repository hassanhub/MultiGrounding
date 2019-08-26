import tensorflow as tf
import numpy as np
from dataflow import MultiProcessRunner, BatchData
import sys, os
from groundnet import GroundNet
from gen import ImageTextBatchGen
import tensorflow_hub as hub
import yaml

class gnet_trainer():
	def __init__(self,
				 data_path=None,
				 gpu='0',
				 batch_size=32,
				 lr=0.0001,
				 gnet_config='./configs/pnas_elmo_3x3.yml',
				 **kwargs):

		print('Initializing trainer...')
		self.gnet_config = yaml.load(open(gnet_config, 'r'))
		self.gpu = gpu
		self.batch_size = batch_size
		self.lr_0 = lr
		self.data_path = data_path
		self.img_size = self.gnet_config['image_size']
		self.log_step = kwargs.get('log_step', 5000)
		# Define session configuration
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.config = tf.ConfigProto(gpu_options=gpu_options,
									 log_device_placement=True,
									 allow_soft_placement=True)
		self.sess = tf.InteractiveSession(config = self.config)

		print('Initializing generator...')
		# Define batch generator
		self.df = ImageTextBatchGen(direct_batch=False,  
									data_path=self.data_path, 
									shuffle=True,  
									img_size=self.img_size[:2])

		self.iter_per_epoch = int(len(self.df)/self.batch_size)

		#create dataflow for faster batchgen
		self.gen = BatchData(self.df, self.batch_size)
		self.gen = MultiProcessRunner(self.gen, num_prefetch=512, num_proc=8)
		#self.gen.reset_state()

		def tuple_gen():
			for d in self.gen:
				yield tuple(d)

		self.dataset = tf.data.Dataset.from_generator(tuple_gen,output_types=(tf.float32, tf.string),output_shapes=([None,self.img_size[0],self.img_size[1],3],[None]))
		self.dataset = self.dataset.repeat(1000)
		self.input_iterator = self.dataset.make_initializable_iterator()
		self.sess.run(self.input_iterator.initializer)
		self.image_input, self.text_input = self.input_iterator.get_next()

		self.gnet = GroundNet(gnet_config=self.gnet_config)
		self.end_points = self.gnet(image_input=self.image_input,
									text_input=self.text_input,
									training=True)

		self._define_loss()

		vis_model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GroundNet/image_block')
		train_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - set(vis_model_vars))
		train_Op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=train_vars, name='train_op')

		with tf.control_dependencies([train_Op]):
			self.loss_Op = tf.identity(self.loss)

		self.saver_vis = tf.train.Saver(var_list=vis_model_vars)
		self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GroundNet'))

		print('Initializing variables...')
		#self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
		global_vars = tf.global_variables()
		self.sess.run(tf.tables_initializer())
		uninitialized   = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
		uninitialized_vars = [v for (v, f) in zip(global_vars, uninitialized) if not f]
		self.sess.run(tf.variables_initializer(uninitialized_vars))
		print('Restoring visual model...')
		self.saver_vis.restore(self.sess, self.gnet.vis_model.ckpt_path)
		print('All initializations done.')
  
	def _define_loss(self):
		loss_ = self.gnet.matching_loss(self.end_points['wrd_embedding'],
										self.end_points['stacked_v'], 
										self.end_points['sen_embedding']) + tf.losses.get_regularization_loss()
		self.loss = tf.identity(loss_, name='loss')

	def train(self,epochs,ckpt_path='./models/groundnet',save=True):
		print('Training...')
		for e in range(epochs):
			loss = 0
			print('\n======Epoch: {}/{}'.format(e+1,epochs))
			for cnt in range(self.iter_per_epoch):
				loss_ = self.sess.run(self.loss_Op)
				loss += loss_
				AvgLoss = loss/(cnt+1)
				if cnt%self.log_step==0:
					log_str = 'loss_log_{}_{}_{}.txt'.format(self.gnet_config['image_model'],
															 self.gnet_config['conv_type'],
															 self.gnet_config['text_model'])
					open(log_str, 'a').write('train, e: {}, iter: {}, loss: {:0.5f}\n'.format(e,cnt,AvgLoss))
				prnt_AvgLoss = '{}/{} === loss: {:0.4f}\r'.format(cnt+1, self.iter_per_epoch, AvgLoss)
				sys.stdout.write(prnt_AvgLoss)
				sys.stdout.flush()

			if save:
				step = e+1
				prnt_loc = self.saver.save(self.sess, ckpt_path, global_step=step)
				print('\n\nSaving model to ' + prnt_loc + '\n')

class gnet_trainer_distributed():
	def __init__(self,
				 data_path=None,
				 gpu='0',
				 batch_size=32,
				 lr=0.0001,
				 decay_steps=1000,
				 decay_rate=0.95,
				 grdnt_rng=5.,
				 grdnt_clip=False,
				 gnet_config='./configs/pnas_elmo_3x3.yml',
				 **kwargs):

		print('Initializing trainer...')
		self.gnet_config = yaml.load(open(gnet_config, 'r'))
		self.gpu = gpu
		self.batch_size = batch_size
		self.lr_0 = lr
		self.data_path = data_path
		self.img_size = self.gnet_config['image_size']
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.grdnt_rng = grdnt_rng
		self.grdnt_clip = grdnt_clip
		self.log_step = kwargs.get('log_step', 5000)

		# Define session configuration
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.config = tf.ConfigProto(gpu_options=gpu_options,
									 log_device_placement=True,
									 allow_soft_placement=True)
		self.sess = tf.InteractiveSession(config = self.config)

		print('Initializing generator...')
		# Define batch generator
		self.df = ImageTextBatchGen(direct_batch=False,  
									data_path=self.data_path, 
									shuffle=True,  
									img_size=self.img_size[:2])

		self.iter_per_epoch = int(len(self.df)/self.batch_size)
		
		#model definition under distributed training
		self.mirrored_strategy = tf.distribute.MirroredStrategy()
		self.num_repl = self.mirrored_strategy.num_replicas_in_sync

		#creat dataflow for faster batchgen
		self.gen = BatchData(self.df, int(self.batch_size/self.num_repl))
		self.gen = MultiProcessRunner(self.gen, num_prefetch=512, num_proc=8)
		#self.gen.reset_state()

		def tuple_gen():
			for d in self.gen:
				yield tuple(d)

		def train_step(dist_inputs):
			with tf.variable_scope('train/flags'):
				global_step = tf.get_variable("global_step",shape=(),trainable=False,dtype='int64')

			def step_fn(inputs):
				image_input, text_input = inputs
				#model = tf.saved_model.loader.load(self.sess,'','./models/text_models/ELMo_Module')
				#model = tf.saved_model.load(self.sess,
				#					tags='',
				#					export_dir='./models/text_models/ELMo_Module_0',
				#					import_scope='ELMo')

				def elmo_call(dist):
					return hub.Module("./models/text_models/ELMo_Module/", trainable=False, name='GroundNet/text_block/ELMo')
				elmo = tf.distribute.get_replica_context().merge_call(elmo_call)

				gnet = GroundNet(gnet_config=self.gnet_config,elmo=elmo)

				end_points = gnet(image_input=image_input,
								  text_input=text_input,
								  training=True)

				with tf.variable_scope('train_ops'):
					loss = self._define_loss(gnet)

					lr = tf.train.exponential_decay(learning_rate = self.lr_0,
													global_step = global_step,
													decay_steps = self.decay_steps,
													decay_rate = self.decay_rate,
													staircase = True)

					vis_model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GroundNet/image_block')
					train_vars = list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - set(vis_model_vars))
					train_Op = self._train_Op(var_list=train_vars,lr=lr,loss=loss,grdnt_rng=self.grdnt_rng,grdnt_clip=self.grdnt_clip)

					with tf.control_dependencies([train_Op]):
						loss_Op = tf.identity(loss)

				return loss_Op

			per_replica_loss = self.mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
			mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

			return mean_loss

		#initialization
		print('Initializing distributed training...')
		with self.mirrored_strategy.scope():
			self.dataset = tf.data.Dataset.from_generator(tuple_gen,output_types=(tf.float32, tf.string),output_shapes=([None,self.img_size[0],self.img_size[1],self.img_size[2]],[None]))
			self.dataset = self.dataset.repeat(1000)
			self.dist_dataset = self.mirrored_strategy.experimental_distribute_dataset(self.dataset)
			self.input_iterator = self.dataset.make_initializable_iterator()
			self.sess.run(self.input_iterator.initializer)
			self.loss_Op = train_step(self.input_iterator.get_next())

			self.flags = {}
			for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train/flags'):
				name = var.name.replace(':0','').split('/')
				self.flags[name[2]] = var	

			vis_model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GroundNet/image_block')
			self.saver_vis = tf.train.Saver(var_list=vis_model_vars)
			self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GroundNet'))

			print('Initializing variables...')
			with tf.variable_scope('init'):
				#self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
				global_vars = tf.global_variables()
				self.sess.run(tf.tables_initializer())
				uninitialized   = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
				uninitialized_vars = [v for (v, f) in zip(global_vars, uninitialized) if not f]
				self.sess.run(tf.variables_initializer(uninitialized_vars))
				print('Restoring visual model...')
				self.saver_vis.restore(self.sess, './models/visual_models/'+self.gnet_config['image_model']+'.ckpt')
		print('All initializations done.')
	
	def _train_Op(self,
				  var_list,
				  lr,
				  loss,
				  grdnt_rng,
				  grdnt_clip):
	
		opt = tf.train.AdamOptimizer(learning_rate=lr)
		gvs = opt.compute_gradients(loss, var_list=var_list)
		if grdnt_clip:
			gvs = [(grad if grad is None else tf.clip_by_value(grad, -grdnt_rng, grdnt_rng), var) for grad, var in gvs]
		
		trainOp = opt.apply_gradients(gvs)
		return trainOp

	def _define_loss(self,gnet):
		loss_ = gnet.matching_loss(gnet.end_points['wrd_embedding'],
									gnet.end_points['stacked_v'], 
									gnet.end_points['sen_embedding']) + tf.losses.get_regularization_loss()
		return loss_

	def train(self,epochs,ckpt_path='./models/groundnet',save=True):
		print('Training...')
		for e in range(epochs):
			loss = 0
			print('\n======Epoch: {}/{}'.format(e+1,epochs))
			for cnt in range(self.iter_per_epoch):
				loss_ = self.sess.run(self.loss_Op)
				loss += loss_
				AvgLoss = loss/(cnt+1)
				if cnt%self.log_step==0:
					log_str = 'loss_log_{}_{}_{}.txt'.format(self.gnet_config['image_model'],
															 self.gnet_config['conv_type'],
															 self.gnet_config['text_model'])
					open(log_str, 'a').write('train, e: {}, iter: {}, loss: {:0.5f}\n'.format(e,cnt,AvgLoss))
				prnt_AvgLoss = '{}/{} === loss: {:0.4f}\r'.format(cnt+1, self.iter_per_epoch, AvgLoss)
				sys.stdout.write(prnt_AvgLoss)
				sys.stdout.flush()

			if save:
				step = e+1
				prnt_loc = self.saver.save(self.sess, ckpt_path, global_step=step)
				print('\n\nSaving model to ' + prnt_loc + '\n')