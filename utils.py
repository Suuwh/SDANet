import glob
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from parser import get_args


def init_weights(m):
	
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform(m.weight)
		# m.bias.data.fill_(0.0001)
	elif isinstance(m, nn.BatchNorm1d):
		pass
	elif isinstance(m, nn.LayerNorm):
		pass
	else:
		if hasattr(m, 'weight'):
			torch.nn.init.kaiming_normal_(m.weight, a=0.01)
		else:
			pass


def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_eer(Pfa,Pmiss):
	"""
	Calculate EER
	"""
	idxeer=np.argmin(np.abs(Pfa-Pmiss))
	return 0.5*(Pfa[idxeer]+Pmiss[idxeer])


def get_min_dcf(Pfa, Pmiss, p_tar=0.01, normalize=True):#
	"""
	input:
		p_tar: a vector of target priors
		normalize: normalize DCFs
	output:
		Values of minDCF, one for each value of p_tar
	"""
	p_tar = np.asarray(p_tar)
	p_non = 1 - p_tar
	# CDet = CMiss x PTarget x PMiss|Target + CFalseAlarm x (1-PTarget) x PFalseAlarm|NonTarget
	cdet = np.dot(np.vstack((p_tar, p_non)).T, np.vstack((Pmiss,Pfa)))
	idxdcfs = np.argmin(cdet, 1)
	dcfs = cdet[np.arange(len(idxdcfs)), idxdcfs]

	if normalize:
		mins = np.amin(np.vstack((p_tar, p_non)), axis=0)
		dcfs /= mins
	return float(dcfs.squeeze())


def get_utt_list(src_dir):

	l_utt = []
	for r, ds, fs in os.walk(src_dir):
		for f in fs:
			if f[-3:] != 'wav':
				continue
			l_utt.append(r+'/'+f)


	# for idx, line in enumerate(l_utt):
	# 	if '\\' in line:
	# 		label = line.split("\\")
	# 		line_cat = ''
	# 		for i in range(len(label)):
	# 			if i != len(label) - 1:
	# 				line_cat += label[i] + '/'
	# 			else:
	# 				line_cat += label[i]
	#
	# 	l_utt[idx] = line_cat

	return l_utt


def get_trials(args):
	dic_trials = {}

	#Vox-O、Vox-E、Vox-H
	if args.eval_extend:
		#args.trial_path : ./DB/VoxCeleb1/vox1_trials
		for r, ds, fs in os.walk(args.trial_path):
			for f in fs:
				if f[-3:] != 'txt' or f[0] == '.':
					continue
				with open(r+'/'+f, 'r') as ff:
					dic_trials[f[:-4]] = ff.readlines()
					print('{0} number of pair: {1}'.format(f[:-4],len(dic_trials[f[:-4]])))
	else:
		with open(args.eval_trial, 'r') as ff:
			dic_trials[args.eval_trial.split('/')[3][:-4]] = ff.readlines()
			print('{0} number of pair: {1}'.format(args.eval_trial.split('/')[3][:-4],len(dic_trials[args.eval_trial.split('/')[3][:-4]])))

	return dic_trials

def get_spk(fn):
	'''
	input	: (str)file name
	output	: (str)speaker id
	'''

	chunk = fn.strip().split('/')
   
	if len(chunk) == 6:
		c = chunk[-1].strip().split('-')
		if c[3] == 'src':
			return c[4]
		else:
			return c[5]
	elif len(chunk) == 8:
		return chunk[5]
	elif len(chunk) == 7:
		return chunk[4]
	else:
		raise ValueError('data format unknown, got:{}'.format(fn))


def make_d_label(lines):
	idx = 0
	dic_label = {}
	list_label = []
	for line in lines:
		spk = get_spk(line)
		if spk not in dic_label:
			dic_label[spk] = idx
			list_label.append(spk)
			idx += 1
		
	return dic_label, list_label

def dic_embd(all_ID, all_embeddings):
	d_embeddings = {}

	for i in range(len(all_ID)):
		if all_ID[i] in d_embeddings:
			pass
		else:
			d_embeddings[all_ID[i]] = all_embeddings[i]

	return d_embeddings


def make_d_label_spk2uttr(lines):
	idx = 0
	dic_label = {}
	list_label = []
	dic_spk2utt = {}
	for line in lines:

		spk = get_spk(line)
		#label
		if spk not in dic_label:

			dic_label[spk] = idx
			#label
			list_label.append(spk)
			idx += 1
		#utt
		if spk not in dic_spk2utt:

			dic_spk2utt[spk] = []
		dic_spk2utt[spk].append(line)

	return dic_label, list_label, dic_spk2utt

def split_utt_lines(nb_spk_per_batch, nb_utt_per_spk, iter, nb_split, ngpus_per_node, gpu, loader_args):#
	l_return = []
	for i in range(nb_split): l_return.append([loader_args, nb_spk_per_batch, nb_utt_per_spk, ngpus_per_node, iter, gpu])
	return l_return

def zipdir(path, ziph):
	for root, dirs, files in os.walk(path):
		for file in files:
			fn, ext = os.path.splitext(file)
			if ext != ".py":
				continue
			#print(file)
			ap = '/'.join(os.path.abspath(file).split('/')[:-1])
			
			ziph.write(os.path.join(ap, root, file))


def convert():
	files = glob.glob('/media/omnisky/data/suwh/DANet/DB/VoxCeleb2/*/*/*/*.m4a')
	files.sort()
	files = files[113306:]
	print('Converting files from AAC to WAV')
	for fname in tqdm(files):
		outfile = fname.replace('.m4a','.wav')
		out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fname,outfile), shell=True)
		if out != 0:
			print('Conversion failed %s.'%fname)
