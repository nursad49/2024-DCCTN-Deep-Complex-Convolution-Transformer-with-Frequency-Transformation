# 2024-DCCTN-Deep-Complex-Convolution-Transformer-with-Frequency-Transformation


———————————********—————————-
Speech Enhancement for Cochlear Implant Listeners
Author: Dr. Nursadul Mamun, Dr. John H.L. Hansen
Center of Robust Speech Systems-CI Lab
University of Texas at Dallas, USA
———————————********—————————-

# -----------------------------------------------------
# Implementation of three networks (CFTNet, DCCTN, and DATCFTNet) for speech enhancement.
# ******************* Network-1: CFTNet ******************
# Complex-valued Frequency Transformation Network for Speech Enhancement
# Authors: Nursadul Mamun, John H.L. Hansen “CFTNet: Complex-valued frequency transformation
# network for speech enhancement, INTERSPEECH, Dublin, Ireland, 2023.

# ******************* Network-2: DCCTN *******************
# Deep Complex Convolution Transformer Network for Speech Enhancement
# Authors: Nursadul Mamun, John H.L. Hansen “Speech Enhancement for Cochlear Implant using Deep Complex
# Convolution Transformer with Frequency Transformation, IEEE Transaction on Audio, Speech, and Language Processing, 2024.


# ******************* Network-2: DATCFTNet ***************
# DAT-CFTNet: A Dual-Path Attention-based Complex-valued Frequency Transformation Network for Speech Enhancement
# Authors: Nursadul Mamun, John H.L. Hansen “DAT-CFTNet: A Dual-Path Attention-based Complex-valued Frequency
# Transformation Network for Speech Enhancement, ICASSP, Korea, 2024.
#-----------------------------------------------------------

# Architecture
DCCTN network is four-fold. (1) propose a fully complex-valued deep complex convolution transformer network, DCCTN, that uses a complex audio transformer and complex frequency transformation network (2) complex FTL in the encoder to leverage correlation among harmonics to capture global cor- relations over frequency for more effective T-F representations (3) a complex audio transformer within the bottleneck layer of the network. This transformer offers several advantages: the self-attention mechanism captures long-range relationships in speech by focusing on key input sequence features; its parallel processing capabilities outperform RNNs by enabling simultaneous use of multiple processing units for expedited computation [41]; accurately captures both local and global contexts, and the multi-head attention mechanism improves



![Fig-1- Basic block diagram of DCCTN with the complex frequency transformed module Model Overview](https://github.com/nursad49/2024-DCCTN-Deep-Complex-Convolution-Transformer-with-Frequency-Transformation/assets/45471274/2a6c2eee-253f-49e5-a376-03b520e2bb1b)




The folder contains:

AudioDataGeneration.py: This generates the noisy audio samples for different SNRs for the Train, Dev, and Test folders. It saves audio and noisy audio files 			in the Database>Original_Samples>Train/Dev/Test folder. Change the name of the noise and SNRs in the function.

Write_scp_files.py: This generates the .scp files for Dataprep.py. Change the noise name as like AudioDataGenerator.py. 

Dataprep.py: This splits all clean and noisy audio files into 4-second chunks and generates training samples to use to train the system. The generated files 	    are saved in the Database>Training_Samples folder

Network.py: This file contains all proposed networks (CFTNet, DCCTN, and DATCFTNET). You can test each network by running this files seperately. 

Train.py: This file contains all files related to train the model. This import training data from Database>Training_Samples folder and related model from the 	Network.py file.

Test.py: This evaluates the network by using the test samples from Database>Original_Samples>Test folder and generate the enhanced files in the same folder.


Dependencies:

dataloader.py: This helps the train.py function to load all training samples (train and Dev) from Database>Training_Samples folder.

modules.py: It contains all the functions required to design the model.

utils.py: It contains all functions required for the whole system.

objective_metrics.py: This contains all objective speech intelligibility and quality metrics, and loss functions.


How to Run:
												
	    # Part 1: If running for the first time or folders are not available for this database:

Step 1: Execute AudioDataGeneration.py to generate noisy samples corresponding to clean samples for different noise types and SNRs. Ensure you have clean files in Database>Original_Samples>Clean and noise in Database>Original_Samples>Different_Noise folder to generate rthe equired noisy files. Modify AudioDataGeneration.py according to your specifications for noisy samples and make necessary edits. 
   		*If you already have noisy samples for corresponding clean samples, you can skip this step.*

Step 2: Run `Write_scp_files.py` to generate `Train.scp`, `Dev.scp`, and `Test.scp` files. These files will be utilized in the `Dataprep.py` function to segment all training files.
   		*If you already have noisy samples for corresponding clean samples, you can skip this step.*

Step 3: Execute `Dataprep.py` to segment audio files and create the `Database > Training_Samples` folder.
   		*If you already have these files, you can skip this step.*

	    # Part 2:  Run these steps to train your model every time**



Step 4: To train the model run this function in the Python terminal:-

			python3 train.py --model "$model name" --b "$batch size" --e "$Num of epoch" --loss "$Loss Function" --gpu "$GPUs"
			# For Example: python3 train.py --model CFTNet --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
			# Simple Example: python3 train.py --model CFTNet


This will save a .ckpt file in the Saved_Models>$Model Name folder.
			Repeat step 4 every time to run the desired model.


Step 5: To test the model run this function in the Python terminal:-

			“python3 test.py”

Please change the model name, and model file from the “Saved_Models>$Model Name” folder in the main file of test.py before testing.


