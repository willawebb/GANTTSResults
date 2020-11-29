# GANTTSResults
This repository contains the training/testing data, and generated audio.

---

The GAN-TTS model we used is a fork of Yanggeng1995's GAN-TTS implementation, linked here: https://github.com/yanggeng1995/GAN-TTS. 

This model was created using a system with a 2080, 32GB of RAM, and a 5600X. Training was done through Anaconda, with the following packages installed;

CUDA 11.0 Version 1
CuDNN 8.0.5
librosa 0.8.0
numpy 1.19.2
torch 1.7.0+cu110
tensorboardx 2.1

Yang's implementation had several bugs on my setup that required me to edit his code to make it work.

The problem itself was that PyTorch would refuse to run, on the condition there existed two devices, my CPU and my GPU, when there should have only been one.

I traced this problem to lines 15, 69, and 70 in loss.py, where the "window" variable appeared to be causing the issues. To make this code work, I appended '.cuda()' to each instance of window in the code to force PyTorch to only consider the available CUDA devices.

The training itself was done with the default settings in train.py, though batch-size was adjusted to 16 due to an OoM error. The model was trained off of only 230 samples, speaker p225 from the University of Edinburgh's Centre for Speech Technology Research VCTK Corpus, linked here: https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning. For training, simply grab that speaker's folder from the kaggle archive and follow yanggeng's steps to process the audio and train the model.

If you have a checkpoint, simply put it in the logdir folder, and create a 'checkpoint' file (no suffix) with "model.chkpt-[NUMBER].pt" written inside. For example, I have left this file filled with "model.ckpt-235000.pt", which will attempt to resume from that checkpoint should that file be there. To use the resume feature, simply add the --resume command when you execute train.py, pointing at "logdir".

---

