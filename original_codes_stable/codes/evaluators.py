import tensorflow as tf
import numpy as np
from dataflow import MultiProcessRunner, BatchData, RNGDataFlow
import sys, os, cv2, lmdb
import pickle
import random
from utils import *

def calc_correctness(annot,heatmap,orig_img_shape):
    bbox_dict = heat2bbox(heatmap,orig_img_shape)
    bbox, bbox_norm, bbox_score = filter_bbox(bbox_dict=bbox_dict, order='xyxy')
    bbox_norm_annot = union(annot['bbox_norm'])
    bbox_annot = annot['bbox']
    bbox_norm_pred = union(bbox_norm)
    bbox_correctness = isCorrect(bbox_norm_annot, bbox_norm_pred, iou_thr=.5)
    hit_correctness = isCorrectHit(bbox_annot,heatmap,orig_img_shape)
    att_correctness = attCorrectness(bbox_annot,heatmap,orig_img_shape)
    return bbox_correctness,hit_correctness,att_correctness
    
class gnet_evaluator():
	def __init__(self,
				 ckpt_path,
				 data_path,
				 gpu='0',
				 batch_size=32,
				 query_level='referral'):

		print('Initializing evaluator...')
		self.gpu = gpu
		self.data_path = data_path
		self.ckpt_path = ckpt_path
		self.batch_size = batch_size
		self.img_size = (299,299,3)

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
										 gpu=self.gpu)

	
	def __call__(self):
		return self.start_validation()

	def update_data(self,data_path,query_level):
		del self.df
		del self.gen
		self.df = ImageTextAnnotBatchGen(data_path=data_path, 
										 img_size=self.img_size[:2],
										 query_level=query_level)

		self.iter_per_epoch = int(len(self.df)/self.batch_size)

		#create dataflow for faster batchgen
		self.gen = BatchData(self.df, self.batch_size)
		self.gen = MultiProcessRunner(self.gen, num_prefetch=512, num_proc=8)

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
			prnt = 'Sample {}/{}, IoU:{:.2f}, Pointing Accucary:{:.2f}, Attention Correctness:{:.2f} \r'.format(var[0],var[1],var[2],var[3],var[4])
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
				 gpu='0'):

		self.gpu = gpu
		self.ckpt_path = ckpt_path
		self.img_size = (299,299,3)
		# Define session configuration
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.config = tf.ConfigProto(gpu_options=gpu_options,
									 log_device_placement=True,
									 allow_soft_placement=True)
		self.sess = tf.InteractiveSession(config = self.config)

		self.gnet = LoadGroundNet(ckpt_path=self.ckpt_path,sess=self.sess)
		self.end_points = self.gnet.end_points
		self.image_input = self.end_points['image_input']
		self.text_input = self.end_points['text_input']

	def __call__(self,img,txt,endpoint_names):
		tensorlist = [self.end_points[n] for n in endpoint_names]
		return self.sess.run(tensorlist, feed_dict={self.image_input: img,
													self.text_input: txt})


class LoadGroundNet():
	def __init__(self,
				 ckpt_path,
				 sess,
				 **kwargs):
		
		self.sess = sess
		self.ckpt_path = ckpt_path
		self.end_points = {}
		self.image_size = (299,299,3)
		self._load_gnet()

	def _load_gnet(self):
		#loading grounding pretrained model
		print('Loading grounding pretrained model...')
		saver = tf.train.import_meta_graph(self.ckpt_path+'.meta')
		_ = self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
		saver.restore(self.sess, self.ckpt_path)
		self.end_points['image_input'] = self.sess.graph.get_tensor_by_name("input_img:0")
		self.end_points['text_input'] = self.sess.graph.get_tensor_by_name("text_input:0")
		self.end_points['mode'] = self.sess.graph.get_tensor_by_name("mode:0")
		self.end_points['score_word'] = self.sess.graph.get_tensor_by_name("attention/score_word:0")
		self.end_points['score_sentence'] = self.sess.graph.get_tensor_by_name("attention/score_sentence:0")
		self.end_points['heatmap_word'] = self.sess.graph.get_tensor_by_name("attention/heatmap_word:0")
		self.end_points['heatmap_sentence'] = self.sess.graph.get_tensor_by_name("attention/heatmap_sentence:0")
		#heatmap_wk = self.sess.graph.get_tensor_by_name("attention/level_heatmap_word:0")
		#heatmap_sk = self.sess.graph.get_tensor_by_name("attention/level_heatmap_sentence:0")
		self.end_points['level_index_word'] = self.sess.graph.get_tensor_by_name('attention/level_index_word:0')
		self.end_points['level_score_word'] = self.sess.graph.get_tensor_by_name('attention/level_score_word:0')
		self.end_points['level_index_sentence'] = self.sess.graph.get_tensor_by_name('attention/level_index_sentence:0')
		print('Loading done.')

	def __call__(self):
		return self.end_points

		
class ImageTextAnnotBatchGen(RNGDataFlow):
	def __init__(self,
				 data_path=None,  
				 img_size=(299,299),
				 query_level='referral'):

		self.data_path = data_path
		self.img_size = img_size
		self.entries = self._list_entries()
		self.nbSamples = self._get_stats()
		self.per_sen_qry = query_level=='sentence'
		if self.per_sen_qry:
			#make sure if "source_sen" actually exists
			assert self.per_sen_qry == self._isPerSen()

	def _get_stats(self):
		nbSamples = 0
		for entry in self.entries:
			nbSamples+=len(self.annotations[entry]['annotations'])
		return nbSamples

	def _list_entries(self):
		lmdb_path = self.data_path['lmdb']
		lmdb_env = lmdb.open(lmdb_path, map_size=int(1e11), readonly=True, lock=False)
		self.txn = lmdb_env.begin(write=False)

		annt_path = self.data_path['annotations']
		with open(annt_path, 'rb') as f:
			self.annotations = pickle.load(f, encoding='latin1')
		return set(self.annotations.keys())

	def _isPerSen(self):
		smpl_key = random.sample(self.entries,1)[0]
		smpl_qry = self.annotations[smpl_key]['annotations'][0]
		return 'source_sen' in smpl_qry

	def _get_unq_sen(self,annots):
		dict_ = {}
		for annot in annots:
			sen = annot['source_sen']
			if sen not in dict_:
				dict_[sen] = []
			dict_[sen].append(annot)
		return dict_

	def __len__(self):
		return self.nbSamples

	def __iter__(self):
		while(True):
			if len(self.entries)==0:
				break
			for entry in self.entries:
				if len(self.annotations[entry]['queries'])==0:
					continue
				imgbin = self.txn.get(entry.encode('utf-8'))
				if imgbin!=None:
					buff = np.frombuffer(imgbin, dtype='uint8')
				else:
					continue
			
				imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
				img = cv2.resize(imgbgr[:,:,[2,1,0]], self.img_size)

				if self.per_sen_qry:
					#return unique sentence with all annotations available
					tmp_dict_ = self._get_unq_sen(self.annotations[entry]['annotations'])
					for sen in tmp_dict_:
						annot = np.array(tmp_dict_[sen])
						#each query is a sentence, it has many annotations
						yield [img, sen, annot]

				else:
					for annot_entity in self.annotations[entry]['annotations']:
						sen = annot_entity['query']
						annot = np.array([annot_entity])
						#each query is a simple referral expression, paired witth only one annotation
						yield [img, sen, annot]
