import tensorflow as tf
import numpy as np
from dataflow import MultiProcessRunner, BatchData
import sys, os
from groundnet import GroundNet
from gen import ImageTextAnnotBatchGen
import yaml
from utils import *

class gnet_evaluator():
	def __init__(self,
				 ckpt_path,
				 data_path,
				 gpu='0',
				 batch_size=32,
				 gnet_config='./configs/pnas_elmo_3x3.yml',
				 query_level='referral'):

		print('Initializing evaluator...')
		self.gnet_config = yaml.load(open(gnet_config, 'r'))
		self.img_size = self.gnet_config['image_size']
		self.gpu = gpu
		self.data_path = data_path
		self.ckpt_path = ckpt_path
		self.batch_size = batch_size
		print('Initializing generator...')
		# Define batch generator
		self.df = ImageTextAnnotBatchGen(data_path=self.data_path, 
										 img_size=self.img_size[:2],
										 query_level=query_level)

		self.iter_per_epoch = int(len(self.df)/self.batch_size)

		#create dataflow for faster batchgen
		self.gen = BatchData(self.df, self.batch_size)
		self.gen = MultiProcessRunner(self.gen, num_prefetch=512, num_proc=8)
		#self.gen.reset_state()

		self.gnet_infer = gnet_inference(ckpt_path=self.ckpt_path,
										 gpu=self.gpu,
										 gnet_config=gnet_config)

	
	def __call__(self):
		return self.start_validation()

	def start_validation(self):
		cnt_overall = 0
		cnt_correct = 0
		cnt_correct_hit = 0
		att_correct = 0
		cat_cnt_overall = {}
		cat_cnt_correct = {}
		cat_cnt_correct_hit = {}
		cat_att_correct = {}
		cat_lvl_scores = {}
		wrd_idx_list = []
		sen_idx_list = []
		cnt = 0

		endpoints = ['heatmap_word', 
					 'score_word', 
					 'score_sentence', 
					 'level_index_word', 
					 'level_index_sentence',
					 'level_score_word']

		for img,txt,annots in self.gen:
			#print(cnt, txt, annots)
			cnt+=1
			eval_tensors = self.gnet_infer(img,txt,endpoints)
			qry_heats, qry_scores, sen_score, wrd_idx, sen_idx, lvl_scores = eval_tensors

			#checking correctness
			sen_idx_list.extend(sen_idx)
			for c, sen in enumerate(txt):
				wrds = sen.split()
				wrd_idx_list.extend(wrd_idx[c,:len(wrds)])
				for annot in annots[c]:
					orig_img_shape = annot['image_size'][:2]
					query = annot['query']
					idx = annot['idx']
					#print("We have {} words at positions {} for query `{}` and {} word scores for sentence #{}: `{}`".format(len(wrds), idx, query, qry_scores[c,:].shape, c, sen))
					if len(query.split())==0 or len(idx)==0:
						print('query zero')
					category = list(annot['category'])
					#skip non-groundable queries
					if 'notvisual' in category or len(annot['bbox_norm'])==0:
						continue
					if not check_percent(union(annot['bbox_norm'])):
						continue

					cnt_overall+=1
					for cat in category:
						if cat not in cat_cnt_overall:
							cat_cnt_overall[cat] = 0
						cat_cnt_overall[cat]+=1    
					try:
						tmp = qry_scores[c,idx]
					except:
						print("Are you running the evaluation on Flickr30K without query_level='sentence'?")
						print(c, idx, query)
					if np.mean(qry_scores[c,idx])==0:
						pred = {}
					else:
						heatmap = np.average(qry_heats[c,idx,:], weights = qry_scores[c,idx], axis=0)
						bbox_c,hit_c,att_c = calc_correctness(annot,heatmap,orig_img_shape)
						cnt_correct+=bbox_c
						cnt_correct_hit+=hit_c
						att_correct+=att_c
						for cat in category:
							#per-cat lvl score
							if cat not in cat_lvl_scores:
								cat_lvl_scores[cat] = []
							cat_lvl_scores[cat].append(lvl_scores[c,idx,:])
							
							#per-cat bbox acc
							if cat not in cat_cnt_correct:
								cat_cnt_correct[cat] = 0
							cat_cnt_correct[cat] += bbox_c
							
							#per-cat hit acc
							if cat not in cat_cnt_correct_hit:
								cat_cnt_correct_hit[cat] = 0
							cat_cnt_correct_hit[cat] += hit_c
							
							#per-cat att acc
							if cat not in cat_att_correct:
								cat_att_correct[cat] = []
							cat_att_correct[cat].append(att_c)
						
			var = [cnt,self.iter_per_epoch,100.*cnt_correct/cnt_overall,100.*cnt_correct_hit/cnt_overall,100.*att_correct/cnt_overall]
			prnt = 'Sample {}/{}, IoU: {:.2f}, Pointing Accuracy: {:.2f}, Attention Correctness: {:.2f} \r'.format(var[0],var[1],var[2],var[3],var[4])
			sys.stdout.write(prnt)                
			sys.stdout.flush()
		
		#overall acc
		hit_acc = cnt_correct_hit/cnt_overall
		iou_acc = cnt_correct/cnt_overall
		att_crr = att_correct/cnt_overall

		#cat-wise acc
		for cat in cat_cnt_correct:
			cat_cnt_correct[cat]/=cat_cnt_overall[cat]
		for cat in cat_cnt_correct_hit:
			cat_cnt_correct_hit[cat]/=cat_cnt_overall[cat]
		for cat in cat_att_correct:
			cat_att_correct[cat]=np.mean(cat_att_correct[cat])

		return iou_acc,hit_acc,att_crr,wrd_idx_list,sen_idx_list,cat_lvl_scores,cat_cnt_correct,cat_cnt_correct_hit,cat_att_correct


class gnet_inference():
	def __init__(self,
				 ckpt_path,
				 gpu='0',
				 gnet_config='./configs/pnas_elmo_3x3.yml'):

		self.gnet_config = yaml.load(open(gnet_config, 'r'))
		self.gpu = gpu
		self.ckpt_path = ckpt_path
		self.img_size = self.gnet_config['image_size']
		# Define session configuration
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.config = tf.ConfigProto(gpu_options=gpu_options,
									 log_device_placement=True,
									 allow_soft_placement=True)
		self.sess = tf.InteractiveSession(config = self.config)

		self.image_input = tf.placeholder(shape=(None,)+self.img_size, dtype=tf.float32, name='image_input')
		self.text_input = tf.placeholder(shape=(None),dtype=tf.string, name='text_input')

		self.gnet = GroundNet(gnet_config=self.gnet_config)
		self.end_points = self.gnet(image_input=self.image_input,
									text_input=self.text_input,
									training=False)

		self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GroundNet'))

		print('Restoring variables from checkpoint...')
		self.sess.run([tf.initialize_all_tables(),tf.initialize_all_variables()])
		self.saver.restore(self.sess, self.ckpt_path)

	def __call__(self,img,txt,endpoint_names):
		tensorlist = [self.end_points[n] for n in endpoint_names]
		return self.sess.run(tensorlist, feed_dict={self.image_input: img,
													self.text_input: txt})




