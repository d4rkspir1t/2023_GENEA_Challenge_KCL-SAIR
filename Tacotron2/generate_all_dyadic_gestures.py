import argparse
import librosa
from time import time
import os
import time
import math
import h5py
import numpy as np
from tqdm import tqdm
import joblib as jl
import torch
from pymo.viz_tools import *
from pymo.writers import *
from common.model import Tacotron2
from common.hparams_dyadic import create_hparams

from scipy.signal import savgol_filter


parser = argparse.ArgumentParser()
parser.add_argument('-f', "--h5file", type=str, default="../tst_main-agent_v0.h5")
parser.add_argument('-fi', "--h5file_interlocutor", type=str, default="../tst_interloctr_v0.h5")
parser.add_argument('-ch', "--checkpoint_path", type=str, required=True)
parser.add_argument('-ch2', "--checkpoint_path_listener", type=str, required=True)
parser.add_argument('-o', "--output_dir", type=str, default="outputs")
parser.add_argument('-t', "--track", type=str, default="full", help="The track for the bvh files. Can only be either 'full' or 'upper'")
args = parser.parse_args()

print("Predicting {} body motion.".format(args.track))

hparams = create_hparams()
torch.cuda.set_device("cuda:{}".format(hparams.device))
torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)
configname = args.checkpoint_path.split("/")[0]
# configname_listen = args.checkpoint_path_listener.split("/")[0]
args.output_dir = os.path.join(args.output_dir, configname+'_tst')
os.makedirs(args.output_dir, exist_ok=True)

if args.track == "full":
	hparams.n_acoustic_feat_dims = 78
else:
	hparams.n_acoustic_feat_dims = 57


### Load Tacotron2 Model
model = Tacotron2(hparams)
if args.checkpoint_path is not None:
	model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu")['state_dict'])
model.cuda().eval()

model_listen = Tacotron2(hparams)
if args.checkpoint_path_listener is not None:
	model_listen.load_state_dict(torch.load(args.checkpoint_path_listener, map_location="cpu")['state_dict'])
model_listen.cuda().eval()

# Load postprocessing pipeline
npy_root = ".."
mel_mean = np.load(os.path.join(npy_root, "mel_mean.npy"))
mel_std = np.load(os.path.join(npy_root, "mel_std.npy"))
mfcc_mean = np.load(os.path.join(npy_root, "mfcc_mean.npy"))
mfcc_std = np.load(os.path.join(npy_root, "mfcc_std.npy"))
prosody_mean = np.load(os.path.join(npy_root, "prosody_mean.npy"))
prosody_std = np.load(os.path.join(npy_root, "prosody_std.npy"))

h5_interlocutor = h5py.File(args.h5file_interlocutor, "r")
h5 = h5py.File(args.h5file, "r")
for index in tqdm(range(len(h5.keys()))):
	### Load input
	mel = torch.FloatTensor((h5[str(index)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std)
	mfcc = torch.FloatTensor((h5[str(index)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
	prosody = torch.FloatTensor((h5[str(index)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
	text = torch.FloatTensor(h5[str(index)]["text"][:])
	speaker = torch.zeros([mel.shape[0], 17])
	speaker[:, h5[str(index)]["speaker_id"][:]] = 1
	audiotext = torch.cat((mel, mfcc, prosody, text, speaker), axis=-1)
	audiotext = audiotext.transpose(0, 1).unsqueeze(0).cuda()

	### Interlocutor
	mel_interlocutor = torch.FloatTensor((h5_interlocutor[str(index)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std)
	mfcc_interlocutor = torch.FloatTensor((h5_interlocutor[str(index)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
	prosody_interlocutor = torch.FloatTensor((h5_interlocutor[str(index)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
	text_interlocutor = torch.FloatTensor(h5_interlocutor[str(index)]["text"][:])
	speaker_interlocutor = torch.zeros([mel_interlocutor.shape[0], 17])
	speaker_interlocutor[:, h5_interlocutor[str(index)]["speaker_id"][:]] = 1
	audiotext_interlocutor = torch.cat((mel_interlocutor, mfcc_interlocutor, prosody_interlocutor, text_interlocutor, speaker_interlocutor), axis=-1)
	audiotext_interlocutor = audiotext_interlocutor.transpose(0, 1).unsqueeze(0).cuda()

	### Interlocutor motion
	if args.track == "full":
		motion_interlocutor = torch.FloatTensor(h5_interlocutor[str(index)]["motion"]["expmap_full"][:])
		motion_interlocutor = motion_interlocutor.transpose(0, 1).unsqueeze(0).cuda()
	else:
		motion_interlocutor = torch.FloatTensor(h5_interlocutor[str(index)]["motion"]["expmap_upper"][:])
		motion_interlocutor = motion_interlocutor.transpose(0, 1).unsqueeze(0).cuda()

	### Concatenating speaker and interlocutor inputs
	seqlen = audiotext.shape[-1]
	audiotext_interlocutor = audiotext_interlocutor[..., :seqlen]
	motion_interlocutor = motion_interlocutor[..., :seqlen]

	seqlen_interlocutor = audiotext_interlocutor.shape[-1]
	indices = np.clip(np.arange(seqlen), 0, seqlen_interlocutor - 1)
	motion_interlocutor = motion_interlocutor[..., indices]
	audiotext_interlocutor = audiotext_interlocutor[..., indices]

	x = torch.cat((audiotext, audiotext_interlocutor, motion_interlocutor), dim=1)

	### Inference
	txt_sum_below = False
	for idx, txt in enumerate(text):
		# print('TEXT', sum(txt))
		if sum(txt) <= 0:
			txt_sum_below = True

	with torch.no_grad():
		if txt_sum_below:
			y_pred = model_listen.inference(x)
		else:
			y_pred = model.inference(x)
		_, predicted_gesture, _, _ = y_pred
		predicted_gesture = predicted_gesture.squeeze(0).transpose(0, 1).cpu().detach().numpy()

	# todo: convert to bvh and save to output folder
	predicted_gesture = savgol_filter(predicted_gesture, 9, 3, axis=0)
	# print(predicted_gesture.shape)
	# exit()
	
	if args.track == "full":
		predicted_gesture[:, 21:24] = np.mean(predicted_gesture[:, 21:24], axis=0)
		predicted_gesture[:, 27] = np.clip(predicted_gesture[:, 27], -9999, 0.6)
		predicted_gesture[:, 39] = np.clip(predicted_gesture[:, 39], -9999, -2.0)
		predicted_gesture[:, 40] = np.clip(predicted_gesture[:, 40], -9999, 0.4)
		predicted_gesture[:, 41] = np.clip(predicted_gesture[:, 41], -9999, 0.4)
		predicted_gesture[:, 45] = np.clip(predicted_gesture[:, 45], -9999, 0.6)
		predicted_gesture[:, 12] = np.clip(predicted_gesture[:, 12], -9999, 1.4)
		predicted_gesture[:, 3] = np.clip(predicted_gesture[:, 3], -9999, 1.4)
		pipeline = jl.load("../pipeline_expmap_full.sav")

	else:
		predicted_gesture[:, 21-18:24-18] = np.mean(predicted_gesture[:, 21-18:24-18], axis=0)
		predicted_gesture[:, 27-18] = np.clip(predicted_gesture[:, 27-18], -9999, 0.6)
		predicted_gesture[:, 39-18:42-18] = np.mean(predicted_gesture[:, 39-18:42-18], axis=0)
		predicted_gesture[:, 45-18] = np.clip(predicted_gesture[:, 45-18], -9999, 0.6)
		pipeline = jl.load("../pipeline_expmap_upper.sav")

	bvh_data = pipeline.inverse_transform([predicted_gesture])[0]
	writer = BVHWriter()
	with open(os.path.join(args.output_dir, "tst_2023_v0_{:03d}_main-agent.bvh".format(index)), 'w') as f:
		writer.write(bvh_data, f, framerate=30)

h5_interlocutor.close()
h5.close()

if __name__ == "__main__":
	pass
