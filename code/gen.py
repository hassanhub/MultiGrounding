from dataflow import RNGDataFlow
import os, cv2, lmdb
import numpy as np
import pickle
import random

#Plans: 1. add augmentation option
#Plans: 4. 

class ImageTextBatchGen(RNGDataFlow):
	def __init__(self, 
				 direct_batch=False, 
				 data_path=None, 
				 shuffle=True, 
				 file_type= 'jpg', 
				 img_size=(299,299)):

		self.data_path = data_path
		self.file_type = file_type
		self.direct_batch = direct_batch
		self.shuffle = shuffle
		self.img_size = img_size
		self.entries = self._list_entries()
		if self.direct_batch:
			self.nbSamples = len(self.entries)
		else:
			self.nbSamples = self._get_stats()

	def _get_stats(self):
		nbSamples = 0
		#for entry in self.entries:
		#	nbSamples+=len(self.annotations[entry]['queries'])
		nbSamples = len(self.entries)
		return nbSamples

	def _list_entries(self):
		if self.direct_batch:
			return self._list_files()
		else:
			lmdb_path = self.data_path['lmdb']
			lmdb_env = lmdb.open(lmdb_path, map_size=int(1e11), readonly=True, lock=False)
			self.txn = lmdb_env.begin(write=False)

			annt_path = self.data_path['annotations']
			with open(annt_path, 'rb') as f:
				self.annotations = pickle.load(f, encoding='latin1')
			return set(self.annotations.keys())

	def _list_files(self):
		l=[]
		for root, dirs, files in os.walk(self.data_path):
			for file in files:
				if isfile(os.path.join(root, file)) and self.file_type in file:
					file = '.'.join(file.split('.')[:-1])
					l.append(os.path.join(root, file))
		return l

	def __len__(self):
		return self.nbSamples

	def __iter__(self):
		#if self.shuffle:
		#	self.rng.shuffle(self.entries)

		if self.direct_batch:
			#direct batch assumes that each images comes with only one discription
			for entry in self.entries:
				img_path = entry+'.'+self.file_type
				txt_path = entry+'.txt'
				img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,[2,1,0]]
				img = cv2.resize(img, self.img_size)
				with open(txt_path, 'r') as f:
					lines = f.readlines()
					queries = [line.strip('\n') for line in lines if len(line)>2]
				sen = self.rng.choice(queries)

				yield img, sen
		else:
			# while(True):
			# 	if len(self.entries)==0:
			# 		break
			# 	for entry in self.entries:
			# 		if len(self.annotations[entry]['queries'])==0:
			# 			continue
			# 		imgbin = self.txn.get(entry.encode('utf-8'))
			# 		if imgbin!=None:
			# 			buff = np.frombuffer(imgbin, dtype='uint8')
			# 		else:
			# 			continue
				
			# 		imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
			# 		img = cv2.resize(imgbgr[:,:,[2,1,0]], self.img_size)

			# 		sen = self.annotations[entry]['queries'].pop()

			# 		yield [img, sen]
			while(True):
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

					sen = np.random.choice(self.annotations[entry]['queries'])

					yield [img, sen]


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