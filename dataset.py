import os;
import math;
import numpy as np;
import pandas as pd;
import pickle as pickle;

from words import *;
from coco.coco import *;

class Dataset():
	def __init__(self,images,files,captions = None,masks = None,batch_size = 1,train = False):
		self.images = np.array(images);
		self.files = np.array(files);
		self.captions = np.array(captions);
		self.masks = np.array(masks);
		self.batch_size = batch_size;
		self.train = train;
		self.count = len(self.images);
		self.batches = int(self.count*1.0/self.batch_size);
		self.index = 0;
		self.indices = list(range(self.count));
		print('Dataset built......');
		
	def reset(self):
		self.index = 0
		np.random.shuffle(self.indices);

	def next_batch(self):
		if(self.index+self.batch_size<=self.count):
			start = self.index;
			end = self.index+self.batch_size;
			current = self.indices[start:end];
			images = self.files[current];
			if(self.train):
				captions = self.captions[current];
				masks = self.masks[current];
				self.index += self.batch_size;
				return images,captions,masks;
			else:
				self.index += self.batch_size;
				return images;

def train_data(args):
	caption_file = args.train_caption;
	annotation_file = args.train_annotation;
	word_file = args.word_file;
	sentence_length = args.sentence_length;
	vocabulary_size = args.vocabulary_size;
	word_embed = args.word_embed;
	batch_size = args.batch_size;
	wordtable = WordTable(vocabulary_size,word_embed,sentence_length,word_file);
	wordtable.load();
	coco = COCO(caption_file);
	coco.filter_by_cap_len(sentence_length);
	coco.filter_by_words(wordtable.all_words());
	annotations = pd.read_csv(annotation_file);
	images = annotations['image_id'].values;
	files = annotations['image_file'].values;
	captions = annotations['caption'].values;
	captions, masks = embed_captions(captions,wordtable);
	dataset = Dataset(images,files,captions,masks,batch_size,True);
	return coco,dataset;

def val_data(args):
	image_dir = args.val_image;
	caption_file = args.val_caption;
	coco = COCO(caption_file);
	images = list(coco.imgs.keys());
	files= [];
	for image in images:
		files.append(os.path.join(image_dir,coco.imgs[image]['file_name']));
	dataset = Dataset(images,files);
	return coco,dataset;

def test_data(args):
	image_dir = args.test_image;
	files = os.listdir(image_dir);
	images = list(range(len(files)));
	dataset = Dataset(images,files);
	return dataset;

def embed_captions(captions,wordtable):
	embedded_captions = [];
	masks = [];
	for caption in captions:
		embedding,mask = wordtable.embed_sentence(caption);
		embedded_captions.append(embedding);
		masks.append(mask);
	return np.array(embedded_captions),np.array(masks);

