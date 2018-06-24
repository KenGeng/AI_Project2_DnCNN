# AI_Project2_DnCNN
Use tensorflow to implement DnCNN and modify it for my AI course project(image restoration)

#Environment

Python3.6
Tensorflow1.8

#Usage
train example: python main.py --channel 1 --input A --percent 0.8 --train 1

test example: python main.py --channel 1 --input B --percent 0.8 --train 0
(I only use the program to denoise a grey image i.e. channel=1. Need modification for rgb images)

#Result
update when I finish other tasks...

#Reference
[1] Zhang, Kai, et al. "Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising." IEEE Transactions on Image Processing 26.7 (2017): 3142-3155.
[2] https://github.com/crisb-DUT/DnCNN-tensorflow
[3] https://github.com/CKCZZJ/ImgRecovery/
