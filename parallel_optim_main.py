# import necessary packages
from __future__ import print_function
import argparse
from os.path import join, exists, isfile, realpath, dirname

try:
    import gradslam as gs
except ImportError:
    raise Exception("Install gradslam from github")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import os
import json
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import ICL
from gradslam.slam import PointFusion
from torch.utils.data import DataLoader
from chamferdist import ChamferDistance
from chamferdist.chamfer import knn_points

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='RGB-D-Recovery-GradSLAM-Parallel-Optimization')
parser.add_argument('--experiment', type=str, default='uniform_noise', help='Experiment', choices=['semantic', 'uniform_noise', 'slight_noise', 'constant_value', 'salt_pepper'])
parser.add_argument('--save_dir', type=str, default='none', help='Directory to save results')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--seed', type=int, default=123, help='Seed')

def RGBD_Reconstruction_GradSLAM(iter_dataloader):
  prev_frame = None

  for i in range(4):
    if i != 3:
      # Run normal RGB-D Images through GradSlam
      colors, depths, intrinsics, poses, *_ = next(iter_loader)
      rgbdimages = RGBDImages(colors, depths, intrinsics, poses, device=device) 
      live_frame = rgbdimages
      if i == 0:
        pointclouds, live_frame.poses = slam(live_frame)
      else:
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame)
      prev_frame = live_frame
    else:
      # Run corrupted RGB-D Images through GradSlam
      adv_rgbdimages = RGBDImages(adv_colors.requires_grad_(True),
                            adv_depths.requires_grad_(True),
                            adv_intrinsics.requires_grad_(False),
                            adv_poses.requires_grad_(False), device=device)
      pointclouds, adv_rgbdimages.poses = slam.step(pointclouds, adv_rgbdimages)

  return pointclouds, adv_rgbdimages

def loss_fn(gt_cloud, gt_pc_color, pert_cloud, pert_pc_color):
    # dist1: distance between closest points between clouds
    # idx1: index of gt_cloud's closest points to pert_cloud's points
    _KNN = knn_points(pert_cloud, gt_cloud)
    dist1, idx1 = _KNN.dists.squeeze(-1), _KNN.idx.squeeze(-1).detach()

    chamferDist = ChamferDistance()

    cloud_loss = 0.5*chamferdist(pert_cloud,gt_cloud,bidirectional=True)
    color_loss = ((pert_pc_color[0] - gt_pc_color[0, idx1[0].long()]).abs().mean())

    return cloud_loss, color_loss

if __name__ == "__main__":
	opt = parser.parse_args()

	print(opt)

	if opt.save_dir.lower() == 'none':
		raise Exception('Please give path to save results')
	else:
		save_dir = opt.save_dir

	ICL_data_path = './ICL/living_room_traj1_frei_png/'
	adversarials_path = './ICL/living_room_traj1_frei_png/adversarial_data/living_room_traj1_frei_png/'

	if opt.experiment.lower() == 'semantic':
		seg_mask = np.load(adversarials_path+'segmentation_mask.npy') # Load Semantic Segmentation mask
		pillow_mask = seg_mask==17 # Get Pillow only mask

		# Overlay RGB Pillow Mask on orginal rgb after shifting by 200 pixels

		rgb = imageio.imread(adversarials_path+'rgb/3_org.png')
		pillows = pillow_mask.reshape(rgb.shape[0],rgb.shape[1],1)*rgb

		rgb_mask = pillows
		y1, y2 = 0, 0 + rgb.shape[0]
		x1, x2 = 200, 200 + rgb.shape[1]

		alpha_s = pillow_mask[:,:(rgb.shape[1]-200)].reshape(rgb.shape[0],rgb.shape[1]-200,1)
		alpha_l = 1.0 - alpha_s

		adv_rgb = rgb

		adv_rgb[:, x1:] = ((alpha_s * rgb_mask[:,:(rgb.shape[1]-200)])+
					      alpha_l * rgb[:, x1:])

		imageio.imwrite(adversarials_path+'rgb/3.png',adv_rgb)

		# Overlay Depth Pillow Mask on orginal depth after shifting by 200 pixels

		depth = cv2.imread(adversarials_path+'depth/3_org.png',cv2.IMREAD_UNCHANGED)
		depth_mask = pillow_mask*depth
		y1, y2 = 0, 0 + depth.shape[0]
		x1, x2 = 200, 200 + depth.shape[1]

		alpha_s = pillow_mask[:,:(depth.shape[1]-200)]
		alpha_l = 1.0 - alpha_s

		adv_depth = depth

		adv_depth[:, x1:] = ((alpha_s * depth_mask[:,:(depth.shape[1]-200)])+
					      alpha_l * depth[:, x1:])

		cv2.imwrite(adversarials_path+'depth/3.png',adv_depth.astype(np.uint16))

	elif opt.experiment.lower() == 'uniform_noise':

		# Creating RGB Uniform Noise Image

		img = cv2.imread(adversarials_path+'rgb/3_org.png')[...,::-1]/255.0
		noise =  np.random.normal(loc=0, scale=1, size=img.shape)
		noise_image = np.clip(noise,0,1)
		noise_image = (noise_image*255)
		imageio.imwrite(adversarials_path+'rgb/3.png',noise_image)

		# Creating 16-bit Depth Uniform Noise Image

		image = cv2.imread(adversarials_path+'depth/3_org.png', cv2.IMREAD_UNCHANGED)
		uniform_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint16)

		cv2.randu(uniform_noise,0,65535)
		cv2.imwrite(adversarials_path+'depth/3.png', uniform_noise)

	elif opt.experiment.lower() == 'slight_noise':

		# Adding Gaussian Noise to RGB Image
		img = cv2.imread(adversarials_path+'rgb/3_org.png')[...,::-1]/255.0
		noise =  np.random.normal(loc=0, scale=1, size=img.shape)

		# noise overlaid over image
		noisy = np.clip((img + noise*0.2),0,1)

		imageio.imwrite(adversarials_path+'rgb/3.png', noisy)

		# Adding 16-bit Gaussian Noise to Depth Image
		image = cv2.imread(adversarials_path+'depth/3_org.png', cv2.IMREAD_UNCHANGED)
		uniform_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint16)

		cv2.randu(uniform_noise,0,65535)
		depth_noisy = np.clip((image + uniform_noise*0.2),0,65535)
		cv2.imwrite(adversarials_path+'depth/3.png', depth_noisy.astype(np.uint16))

	elif opt.experiment.lower() == 'constant_value':

		# Creating Constant Value RGB-D Image
		image = np.zeros((480,640,3))
		image.fill(135)
		imageio.imwrite(adversarials_path+'rgb/3.png', image.astype(np.uint8))

		d_image = np.zeros((480,640),dtype=np.uint16)
		d_image.fill(10015)
		cv2.imwrite(adversarials_path+'depth/3.png', d_image.astype(np.uint16))

	elif opt.experiment.lower() == 'salt_pepper':

		# Creating Salt & Pepper Gaussian Noise RGB Image
		image = cv2.imread(adversarials_path+'rgb/3_org.png')
		uniform_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)

		cv2.randu(uniform_noise,0,255)

		uniform_noise = np.repeat(uniform_noise.astype(np.uint8).reshape(image.shape[0],image.shape[1],1), 3, axis=2)

		cv2.imwrite(adversarials_path+'rgb/3.png', uniform_noise)

		# Original Depth Image
		depth = cv2.imread(adversarials_path+'depth/3_org.png',cv2.IMREAD_UNCHANGED)
		cv2.imwrite(adversarials_path+'depth/3.png',depth.astype(np.uint16))
	else:
		raise Exception('Unknown experiment')

	# Assigning Number of Iterations
	if opt.experiment.lower() == 'uniform_noise':
		iterations = 400
	else:
		iterations = 200

	cuda = not opt.nocuda
	if cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run with --nocuda")

	device = torch.device("cuda" if cuda else "cpu")

	seed = opt.seed
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	# Ground Truth 3D Reconstruction of first 4 ICL-NUIR frames

	gt_icl_path = './ICL/'

	gt_dataset = ICL(gt_icl_path, seqlen=4, height=240, width=320, start=0, end=4)
	gt_loader = DataLoader(dataset=gt_dataset, batch_size=2)
	gt_colors, gt_depths, gt_intrinsics, gt_poses, *_ = next(iter(gt_loader))

	gt_rgbdimages = RGBDImages(gt_colors, gt_depths, gt_intrinsics, gt_poses, device=device)

	odometry = "gt" # Ground Truth Poses for Odometry

	gt_slam = PointFusion(odom=odometry, device=device)
	gt_pointclouds, gt_recovered_poses = gt_slam(gt_rgbdimages)

	# Setting things up for data containing corrupted rgb-d image
	icl_path = './ICL/living_room_traj1_frei_png/adversarial_data/'

	dataset = ICL(icl_path, seqlen=1, height=240, width=320, start=0, end=4)
	loader = DataLoader(dataset=dataset, batch_size=1)
	iter_loader = iter(loader)

	# Initalize adversarial/corrupted RGB-DImage
	for i in range(4):
	  if i != 3:
	    _ = next(iter_loader)
	  else:
	    adv_colors, adv_depths, adv_intrinsics, adv_poses, *_ = next(iter_loader)

	# Using Ground-Truth Odometry for Corrupted Reconstruction since pose has direct effect on quality of depth reconstruction
	odometry = 'gt'
	slam = PointFusion(odom=odometry, device=device)
	pointclouds = Pointclouds(device=device)

	# Setting parameters for backpropagating to corrupted RGB-D Image
	lr = 0.1
	optimizer = torch.optim.Adam([adv_colors, adv_depths], lr=lr)

	gt_cloud = gt_pointclouds.points_list[0].unsqueeze(0).contiguous().detach().to(device)
	gt_pc_color = gt_pointclouds.colors_list[0].unsqueeze(0).contiguous().detach().to(device)

	print('===> Starting RGB-D Completion', flush=True)

	save_path = save_dir+'/parallel-results/'+opt.experiment+'/'
	os.makedirs(save_path+'rgb/',exist_ok=True)
	os.makedirs(save_path+'depth/',exist_ok=True)

	chamfer_dist = []
	recon_rgb_ssim = []
	recon_depth_mse = []

	for iteration in range(0,iterations+1):
	  iteration_cdist = 0

	  # Forward pass  GradSLAM for corrupted 3D Reconstruction

	  iter_loader = iter(loader)
	  pointclouds, adv_rgbdimages = RGBD_Reconstruction_GradSLAM(iter_loader)

	  pert_cloud = pointclouds.points_list[0].unsqueeze(0).contiguous()
	  pert_pc_color = pointclouds.colors_list[0].unsqueeze(0).contiguous()

	  optimizer.zero_grad()

	  # Calculate Bi-directional Chamfer distance between noisy pointcloud and gt pointcloud and backpropagate
	  pt_cdist, color_cdist = loss_fn(gt_cloud, gt_pc_color, pert_cloud, pert_pc_color)
	  cdist = (pt_cdist + color_cdist)

	  cdist.backward()
	  optimizer.step()

	  iteration_cdist = cdist.item()

	  # Calculate SSIM between groundtruth rgb and optimized rgb
	  ground_truth_rgb = gt_rgbdimages.rgb_image[0,3].detach().cpu().numpy()
	  reconstructed_rgb = adv_rgbdimages.rgb_image[0,0].detach().cpu().numpy()

	  if reconstructed_rgb.max() - reconstructed_rgb.min() == 0:
	    data_range = 1
	  else:
	    data_range = reconstructed_rgb.max() - reconstructed_rgb.min()

	  ssim_noise = ssim(ground_truth_rgb, reconstructed_rgb,
			    data_range=data_range, multichannel=True)

	  # Save Optimized RGB Image
	  imageio.imwrite(save_path+'rgb/'+str(iteration)+'.png',reconstructed_rgb.astype(np.uint8))

	  # Calculate MSE between groundtruth rgb and optimized rgb
	  ground_truth_depth = gt_rgbdimages.depth_image[0,3].detach().cpu().numpy()
	  reconstructed_depth = adv_rgbdimages.depth_image[0,0].detach().cpu().numpy()

	  mse_noise = mean_squared_error(ground_truth_depth, reconstructed_depth)

	  # Save Optimized Depth Image
	  plt.imsave(save_path+'depth/'+str(iteration)+'.png',reconstructed_depth[:,:,0])

	  chamfer_dist.append(iteration_cdist)
	  recon_rgb_ssim.append(ssim_noise)
	  recon_depth_mse.append(mse_noise)

	  if (iteration) % 100 == 0:
	    print("===> Iteration {} Complete: Chamfer Distance: {:.4f}".format(iteration, iteration_cdist), 
			flush=True)
	    print("===> Iteration {} Complete: RGB SSIM: {:.4f}".format(iteration, ssim_noise), 
			flush=True)
	    print("===> Iteration {} Complete: Depth MSE: {:.4f}".format(iteration, mse_noise), 
			flush=True)
	    print("----")

	  del color_cdist, pt_cdist, cdist, iteration_cdist, ssim_noise, mse_noise

	# Save Chamfer Distance, SSIM, MSE values of all iterations
	with open(save_path+"chamfer_dist.txt", "w") as fp:
	  json.dump(chamfer_dist, fp, indent=2) 

	with open(save_path+"RGB_SSIM.txt", "w") as fp:
	  json.dump(recon_rgb_ssim, fp, indent=2)

	with open(save_path+"Depth_MSE.txt", "w") as fp:
	  json.dump(recon_depth_mse, fp, indent=2)
