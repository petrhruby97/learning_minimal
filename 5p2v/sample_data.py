import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import gc
import sys
import copy

def generate(input_folder, output_file, goal_samples):
	print("generating " + name)
	
	#find the number of samples
	first_eval = "./bin/data_sampler " + input_folder + " 10 " + mode + " 1 " + " > " + output_file
	first_samples = os.system(first_eval)
	first_samples = sum(1 for line in open(output_file))
	samples = 10*goal_samples / first_samples
	
	#generate the problems
	second_eval = "./bin/data_sampler " + input_folder + " " + str(int(samples)+1) + " " + mode + " 1 " + " > " + output_file
	print(second_eval)
	#return
	final_samples = os.system(second_eval)
	final_samples = sum(1 for line in open(output_file))
	
	#add the number of samples to the beginning of the file
	sed_eval = "sed -i '1s/^/" + str(final_samples) + "\\n/' " + output_file
	print(sed_eval)
	os.system(sed_eval)
	
	#print the result
	print(str(final_samples) + " p-s pairs generated")

if __name__ == "__main__":
	argc = len(sys.argv)
	if argc < 3:
		print("Run as:")
		print("python3 sample_data.py COLMAP_folder output_file num_samples")
		exit()
	input_folder = sys.argv[1]+"/"
	output_file = sys.argv[2]
	goal_samples = sys.argv[3]+"/"
	generate(input_folder, output_file, goal_samples)
	
	
