import xml.etree.cElementTree as etree
import os, sys, re, pickle, cv2, lmdb, json
from pycocotools.coco import COCO
from data.refer import REFER

class Data():
	def __init__(self):
		self.supported = ['flickr30k_entities', 'coco', 'visual_genome', 'referit']
		self.coco_version = '2014'

	def __call__(self,
				 raw_path,
				 save_path,
				 store_lmdb,
				 name,
				 **kwargs):

		name = name.lower()
		assert name in self.supported

		if name=='flickr30k_entities':
			self._flickr30k(raw_path,save_path,store_lmdb)
		elif name=='coco':
			self._coco(raw_path,save_path,store_lmdb,kwargs['version'])
		elif name=='visual_genome':
			self._vg(raw_path,save_path,store_lmdb)
		elif name=='referit':
			self._referit(raw_path,save_path,store_lmdb)

	def help(self):
		name = input('Please choose one of the following:\n flickr30k_entities\n coco\n visual_genome\n referit\n\n')
		name = name.lower()
		assert name in self.supported
		with open('./data/readme/'+name+'.txt', 'r') as f:
			lines = f.readlines()
			lines = [l.strip('\n') for l in lines]
		for line in lines:
			print(line)

	def _sen2qry(self,sen,annot_dict,size):
		#an interanl function for flickr30k
		sen_list = []
		q_dicts = []
		splits_sen = re.split(r'\[|\]', sen)
		for split_s in splits_sen:
			if split_s == '':
				continue
			elif '/EN#' in split_s:
				tokens = split_s.split()
				annots = tokens[0].split('/')
				obj_id = annots[1].strip('EN#')
				category = annots[2:]
				query = tokens[1:]
				if obj_id == '0':
					bboxes = []
					bboxes_norm = []
				else:
					bboxes = annot_dict[obj_id]
					bboxes_norm = []
					for bbox in bboxes:
						bb_x_n = [float(bb)/size[1] for bb in bbox[0::2]]
						bb_y_n = [float(bb)/size[0] for bb in bbox[1::2]]
						bboxes_norm.append([bb_x_n[0],bb_y_n[0],bb_x_n[1],bb_y_n[1]])
				idx = list(range(len(sen_list),len(query)+len(sen_list)))
				q_dicts.append({'category': set(category),
							   'obj_id': obj_id,
							   'bbox': bboxes,
							   'bbox_norm': bboxes_norm,
							   'idx': idx,
							   'query': ' '.join(query),
							   'image_size': size})
			else:
				query = split_s.split()
			sen_list.extend(query)
		sen_clean = ' '.join(sen_list)
		for q_dict in q_dicts:
			q_dict['source_sen'] = sen_clean
		return q_dicts, sen_clean


	def _get_dict_flickr30k(self,keys,xml_files,annot_path,sen_path,imgs_path,store_lmdb):
		#an internal function for flickr30k
		data_dict = {}
		for k,key in enumerate(keys):
			sys.stdout.write('{}/{} \r'.format(k,len(keys)))
			xml_file = key+'.xml'
			if xml_file not in xml_files:
				print(xml_file)
				continue
			xmlDoc = open(annot_path+xml_file, 'r')
			xmlDocData = xmlDoc.read()
			xmlDocTree = etree.XML(xmlDocData)
			tmp_dict = {}
			for obj in xmlDocTree:
				if obj.tag=='size':
					size = [int(sz.text) for sz in obj]
					size = size[1::-1] #H,W format
				elif obj.tag=='object': #each object has one bbox and could have multiple obj_id
					obj_ids = []
					for item in obj.iter('name'):
						obj_id = item.text
						if obj_id not in tmp_dict:
							tmp_dict.update({obj_id: []})
						obj_ids.append(obj_id)
					for item in obj.iter('bndbox'):
						for obj_id in obj_ids:
							tmp_dict[obj_id].append([int(bb.text) for bb in item])
			
			if store_lmdb:
				img_path = imgs_path+key+'.jpg'
				img = cv2.imread(img_path, cv2.IMREAD_COLOR)
				img_vec = cv2.imencode('.jpg', img)[1]
				with store_lmdb.begin(write=True) as lmdb_txn:
					lmdb_txn.put(key.encode(), img_vec)
			data_dict[key] = {'size': size, 'queries': [], 'annotations': []}
			senDoc = open(sen_path+key+'.txt', 'r').readlines()
			sens = [line.strip('\n') for line in senDoc]
			for sen in sens:
				q_dict, sen_clear = self._sen2qry(sen,tmp_dict,size)
				data_dict[key]['annotations'].extend(q_dict)
				data_dict[key]['queries'].append(sen_clear)
			data_dict[key]['captions'] = data_dict[key]['queries']
		return data_dict

	def _flickr30k(self,raw_path,save_path,store_lmdb):
		imgs_path = os.path.join(raw_path,'Flickr30k_Images/')
		annot_path = os.path.join(raw_path,'Flickr30k_Entities/Annotations/')
		sen_path = os.path.join(raw_path,'Flickr30k_Entities/Sentences/')
		split_path = os.path.join(raw_path,'Flickr30k_Splits/')
		xml_files = os.listdir(annot_path)

		annot_dict_path = os.path.join(save_path,'annotations/')
		os.system('mkdir -p '+annot_dict_path)
		if store_lmdb:
			lmdb_path = os.path.join(save_path,'images.lmdb')
			lmdb_env = lmdb.open(lmdb_path, map_size=int(1e11), lock=False)
			store_lmdb = lmdb_env

		for split in ['train', 'test', 'val']:
			print('Processing '+split+' split...')
			with open(split_path+split+'.txt', 'r') as f:
				lines = f.readlines()
				lines = [l.strip('\n') for l in lines]
			keys = set(lines)
			dict_ = self._get_dict_flickr30k(keys,xml_files,annot_path,sen_path,imgs_path,store_lmdb)
			pickle.dump(dict_,open(annot_dict_path+split+'.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
			print('\n')
		print('Processing done.')

		if store_lmdb:
			store_lmdb.close()

	def _get_split_coco(self,annot_path,split,version):
		#an internal coco function
		instancesAnnFile = '{}/instances_{}{}.json'.format(annot_path,split,version)
		captionsAnnFile = '{}/captions_{}{}.json'.format(annot_path,split,version)
		# initialize COCO api for instance annotations
		coco_obj = COCO(instancesAnnFile)
		# initialize COCO api for caption annotations
		caption_objects = COCO(captionsAnnFile)
		categories = coco_obj.loadCats(coco_obj.getCatIds())
		# building proper objects
		names = {}
		supercats = {}
		for cat in categories:
			names[cat['id']] = cat['name']
			supercats[cat['id']] = cat['supercategory']
		# get all image ids in the dataset
		imageIds = coco_obj.getImgIds();
		image_objects = coco_obj.loadImgs(imageIds)

		return coco_obj, image_objects, caption_objects, names, supercats
	
	def _get_dict_coco(self, imgs_path, coco_obj, image_objects, caption_objects, names, supercats, store_lmdb):
		#an internal coco function
		dictionary = {}
		for n,img_obj in enumerate(image_objects):
			img_id, height, width, filename = (img_obj['id'], img_obj['height'], img_obj['width'], img_obj['file_name'])
			
			if store_lmdb:
				image = cv2.imread(os.path.join(imgs_path, filename), cv2.IMREAD_COLOR)
				img_encoded = cv2.imencode('.jpg', image)[1]
				save_key = str(img_id)
				with store_lmdb.begin(write=True) as lmdb_txn:
					lmdb_txn.put(save_key.encode(), img_encoded)
			
			annotations = []
			bbox_annIds = coco_obj.getAnnIds(imgIds=img_id)
			bbox_anns = coco_obj.loadAnns(bbox_annIds)

			for bbox_ann in bbox_anns:
				bbox = bbox_ann['bbox']
				x_min, x_max, y_min, y_max = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]
				bbox = [x_min, y_min, x_max, y_max]
				bbox_norm = [x_min/width, y_min/height, x_max/width, y_max/height]
				category_id = bbox_ann['category_id']
				supercategory = supercats[category_id]
				category_name = names[category_id]
				object_id = bbox_ann['id']
				segmentation = bbox_ann['segmentation']
				#mask = coco.annToMask(bbox_ann)
				bbox_entity = {'image_id': img_id,
								'bbox': bbox, 
								'bbox_norm': bbox_norm, 
								'supercategory': supercategory,
								'category': category_name, 
								'category_id': category_id, 
								'obj_id': object_id,
								'segmentation': segmentation}
				annotations.append(bbox_entity)

			capsAnnIds = caption_objects.getAnnIds(imgIds=img_id);
			caps = caption_objects.loadAnns(capsAnnIds)
			sentences = [cap['caption'] for cap in caps]
			dictionary[str(img_id)] = {'size': (height, width, 3),
									   'queries': sentences, 
									   'captions': sentences, 
									   'annotations': annotations}

			sys.stdout.write('%d/%d \r' %(n+1,len(image_objects)))
			sys.stdout.flush()

		return dictionary

	def _coco(self,raw_path,save_path,store_lmdb,version):
		imgs_path = os.path.join(raw_path,'COCO_Images/')
		annot_path = os.path.join(raw_path,'COCO_Annotations/')
		annot_dict_path = os.path.join(save_path,'annotations/')
		os.system('mkdir -p '+annot_dict_path)

		if store_lmdb:
			lmdb_path = os.path.join(save_path,'images.lmdb')
			lmdb_env = lmdb.open(lmdb_path, map_size=int(1e11), lock=False)
			store_lmdb = lmdb_env

		for split in ['train','val']: #we only process train and val splits
			print('Processing '+split+' split...')
			coco_obj, image_objects, caption_objects, names, supercats = self._get_split_coco(annot_path, split, version)
			dict_ = self._get_dict_coco(imgs_path, coco_obj, image_objects, caption_objects, names, supercats, store_lmdb)
			pickle.dump(dict_,open(annot_dict_path+split+'.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
			if split=='train':
				train_val_dict_ = dict_
			elif split=='val':
				train_val_dict_.update(dict_)
				pickle.dump(train_val_dict_,open(annot_dict_path+'train_val.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
			print('\n')
		print('Processing done.')
		
		if store_lmdb:
			store_lmdb.close()

	def _get_split_refer(self,splits_path,split):
		#an internal refer function
		with open(os.path.join(splits_path, 'referit_'+split+'_imlist.txt')) as f:
			lines = f.readlines()
			img_ids = [int(l.strip('\n')) for l in lines]
		return img_ids

	def _get_dict_refer(self, imgs_path, caps_path, refer_obj, img_ids_all, img_ids_split, store_lmdb):
		#an intternal refer function
		dictionary = {}
		for n, img_id in enumerate(img_ids_split):
			if img_id not in img_ids_all:
				continue
			img_file_name = refer_obj.Imgs[img_id]['file_name']
			img_obj =  refer_obj.Imgs[img_id]
			height, width = int(img_obj['height']), int(img_obj['width'])

			cap_file_name = os.path.join(img_file_name.split("/")[0], str(img_id) + '.eng')
			cap_file_path = os.path.join(caps_path, cap_file_name)
			cations = []
			if os.path.exists(cap_file_path):
				with open(cap_file_path, encoding = "ISO-8859-1") as f:
					lines = f.readlines()
					captions = lines[3][13:-16].split(';')
					captions = [c for c in captions if len(c)>2]

			if store_lmdb:
				image = cv2.imread(os.path.join(imgs_path, img_file_name), cv2.IMREAD_COLOR)
				img_encoded = cv2.imencode('.jpg', image)[1]

				with store_lmdb.begin(write=True) as lmdb_txn:
					lmdb_txn.put(str(img_id).encode(), img_encoded)

				height_i, width_i = image.shape[:2]
				if int(height_i) != int(height) or int(width_i) != int(width):
					self.wrong += 1
					#print('Image H,W mismatch for image_id: '+str(img_id))
					continue
			

			refs = refer_obj.imgToRefs[img_id]
			annotations = []
			queries = []
			count = 0
			for ref in refs:
				split = ref['split']
				ref_id = ref['ref_id']
				sentences = ref['sentences']
				query_list = []
				for sentence in sentences:
					query_list.append(sentence['sent'])

				query_list = list(set(query_list)) #just take unique queries
				category_id = ref['category_id']
				bbox = refer_obj.getRefBox(ref_id)
				x1, x2, y1, y2 = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]
				bbox = [int(x1), int(y1), int(x2), int(y2)]
				bbox_norm = [x1/width, y1/height, x2/width, y2/height]
				if max(bbox_norm)>1:
					continue

				ann = refer_obj.refToAnn[ref_id]
				segmentation = ann['segmentation']
				for query in query_list:
					ref_entity = {'ref_id': ref_id,
								  'image_id': img_id,
								  'query': query,
								  'category': set([category_id]),
								  'bbox': [bbox],
								  'bbox_norm':[bbox_norm],
								  'segmentation':segmentation,
							   	  'idx': list(range(len(query.split()))),
							   	  'image_size': (height,width,3)}
					annotations.append(ref_entity)
					queries.append(query)
				
			dictionary[str(img_id)] = {'image_id': img_id,
										'size': (height,width,3),
										'queries': queries,
										'annotations':annotations,
										'captions': captions}

			sys.stdout.write('%d/%d \r' %(n+1,len(img_ids_split)))
			sys.stdout.flush()
		return dictionary

	def _referit(self,raw_path,save_path,store_lmdb):
		imgs_path = os.path.join(raw_path,'ReferIt_Images/')
		splits_path = os.path.join(raw_path,'ReferIt_Splits/')
		caps_path = os.path.join(raw_path, 'RefClef_Captions/')
		annot_dict_path = os.path.join(save_path,'annotations/')
		os.system('mkdir -p '+annot_dict_path)
		self.wrong = 0

		if store_lmdb:
			lmdb_path = os.path.join(save_path,'images.lmdb')
			lmdb_env = lmdb.open(lmdb_path, map_size=int(1e11), lock=False)
			store_lmdb = lmdb_env

		#initialization
		dataset = 'refclef'
		splitBy = 'unc'
		refer_obj = REFER(raw_path, dataset, splitBy)
		img_ids_all = refer_obj.getImgIds()

		for split in ['train', 'test', 'val']:
			print('Processing '+split+' split...')
			img_ids_split = self._get_split_refer(splits_path, split)
			dict_ = self._get_dict_refer(imgs_path, caps_path, refer_obj, img_ids_all, img_ids_split, store_lmdb)
			pickle.dump(dict_,open(annot_dict_path+split+'.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
			if split=='train':
				train_val_dict_ = dict_
			elif split=='val':
				train_val_dict_.update(dict_)
				pickle.dump(train_val_dict_,open(annot_dict_path+'train_val.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
			print('\n')
		print('Processing done.')
		print('Discarded {} mismatches in images and annotations'.format(self.wrong))
		if store_lmdb:
			store_lmdb.close()

	def _words_preprocess(self, phrase):
		""" preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
		#this is an iternal function of vg
		replacements = {u'½': u'half',
						u'—' : u'-',
						u'™': u'',
						u'¢': u'cent',
						u'ç': u'c',
						u'û': u'u',
						u'é': u'e',
						u'°': u' degree',
						u'è': u'e',
						u'…': u''}

		for k, v in replacements.items():
			phrase = phrase.replace(k, v)
		return str(phrase)

	def _get_meta_vg(self,anns_path,splits_path):
		print('Loading VG metadata to memory...')
		with open(os.path.join(anns_path, 'imgs_data.pickle'), 'rb') as f:
			imgs_data = pickle.load(f, encoding='latin1')

		with open(os.path.join(anns_path, 'region_descriptions.json'), 'r') as f:
			regions_ = json.load(f)
		
		regions_data = {}
		for region_ in regions_:
			regions_data[region_['id']] = region_['regions']

		with open(os.path.join(splits_path, 'data_splits.pickle'), 'rb') as f:
			img_ids_split = pickle.load(f, encoding='latin1')
		
		print('Loading done.')
		return imgs_data, regions_data, img_ids_split

	def _get_dict_vg(self, imgs_path, img_ids_split, imgs_data, regions_data, store_lmdb):
		dictionary = {}
		wrong_xw, wrong_yw, wrong_ww, wrong_hw = [0]*4

		for n, img_id in enumerate(img_ids_split):
			if store_lmdb:
				img_path = os.path.join(imgs_path, str(img_id)+'.jpg')
				if os.path.exists(img_path):
					img = cv2.imread(img_path, cv2.IMREAD_COLOR)
				else:
					continue

				img_encoded = cv2.imencode('.jpg', img)[1]
				save_key = img_id
				with store_lmdb.begin(write=True) as lmdb_txn:
					lmdb_txn.put(str(img_id).encode(), img_encoded)
			
			height, width = imgs_data[img_id]['height'], imgs_data[img_id]['width']
			regions = regions_data[img_id]
			queries = []
			annotations = []
			query_entities = {}

			for region in regions:
				query = self._words_preprocess(region['phrase'])
				split_query = query.split()
				if len(split_query) >= 10 or len(split_query) == 0:
					continue
				if '\n' in query:
					continue

				x, y = region['x'], region['y']
				w, h = region['width'], region['height']
				
				# clamp to image
				if x < 1: x = 1
				if y < 1: y = 1
				if x > width - 1: 
					x = width - 1
					wrong_xw += 1
				if y > height - 1: 
					y = height - 1
					wrong_yw += 1
				if x + w > width: 
					w = width - x
					wrong_ww += 1
				if y + h > height: 
					h = height - y
					wrong_hw += 1
				
				bbox = [x, y, x + w, y + h]

				bbox_norm = [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]
				
				if max(bbox_norm) > 1:
					continue
				#    print (str('img_id'), + '\n')

				#region_id = region['region_id']

				if query not in query_entities:
					query_entities[query] = {'bbox': [bbox], 'bbox_norm': [bbox_norm]}
				else:
					query_entities[query]['bbox'].append(bbox)
					query_entities[query]['bbox_norm'].append(bbox_norm)
			
			for query in query_entities:
				bbox_entity = {'image_id':str(img_id), 
							   'bbox': query_entities[query]['bbox'], 
							   'bbox_norm': query_entities[query]['bbox_norm'], 
							   'query': query,
							   'category': set(['All']),
							   'idx': list(range(len(query.split()))),
							   'image_size': (height,width,3)}
				annotations.append(bbox_entity)

			queries = list(query_entities.keys())
			dictionary[str(img_id)] = {'size': (height,width,3),
									   'queries': queries,
									   'annotations': annotations}
			
			sys.stdout.write('%d/%d \r' %(n+1,len(img_ids_split)))
			sys.stdout.flush()

		return dictionary

	def _vg(self,raw_path,save_path,store_lmdb):
		imgs_path = os.path.join(raw_path,'VG_Images/')
		splits_path = os.path.join(raw_path,'VG_Splits/')
		anns_path = os.path.join(raw_path, 'VG_Annotations/')
		annot_dict_path = os.path.join(save_path,'annotations/')
		os.system('mkdir -p '+annot_dict_path)

		if store_lmdb:
			lmdb_path = os.path.join(save_path,'images.lmdb')
			lmdb_env = lmdb.open(lmdb_path, map_size=int(1e11), lock=False)
			store_lmdb = lmdb_env

		imgs_data, regions_data, img_ids_split = self._get_meta_vg(anns_path,splits_path)

		for split in ['train', 'test']:
			print('Processing '+split+' split...')
			dict_ = self._get_dict_vg(imgs_path, img_ids_split[split], imgs_data, regions_data, store_lmdb)
			pickle.dump(dict_,open(annot_dict_path+split+'.pickle','wb'),protocol=pickle.HIGHEST_PROTOCOL)
			print('\n')
		print('Processing done.')
		if store_lmdb:
			store_lmdb.close()