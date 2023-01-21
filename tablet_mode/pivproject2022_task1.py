#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob
import os
import zipfile as zf
import pickle
from pathlib import Path
import random
import math
import scipy.ndimage
import scipy.io
import matplotlib.pyplot as plt
from scipy import optimize as opt
import matplotlib.image as img
import sys

# global variables:
MIN_MATCH_COUNT = 24 # minimum number of matches required 
MAX_ITER = 600 # number of iterations in ransac 
NUM_MATCH = 500 # total number of matches that we want to subsample from different locations of the image.
                # For every region in the image analysed, if it contains at least a descriptor, and 
                # if the number of descriptors subsampled is fixed to N = 1, there will be 500 matches. 
                # Nevertheless, some regions of the image can have no descriptors, so the total number of matches will be reduced. 
                # For solving this, we can sample N > 1 descriptors from the regions of the image analysed 

threshold_error = 4 # threshold error for points to be considered inliers, when performing ransac 

subsampling = 1 # variable that defines if subsampling of the source image descriptors is to 
                   # be performed (set to 1, 0 otherwise), when performing the matching


# In[2]:


def match_features(des_src, des_dest, threshold):
    '''
    Implements the Nearest Neighbor Distance Ratio Test (NNDR) - Equation 4.18 in Section 4.1.3 of 
    Szeliski - to assign matches between interest points in two images. It also searches for mutual 
    matches and applies the NNDR test
  
    A match is between a feature in des_src and a feature in des_dest. We can
    represent this match as a the index of the feature in des_src and the index
    of the feature in des_dest
    
    :params:
    :des_src: an np array of features for interest points in source image
    :des_dest: an np array of features for interest points in destination image
    
    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into des_src and the second column is an index into des_dest
    '''
    
    global MIN_MATCH_COUNT

    matches = []
    
    # Re-normalize
    des_dest_normalize = des_dest / np.linalg.norm(des_dest, axis = 0)
    
    des_src_normalize = des_src / np.linalg.norm(des_src, axis = 0)
    
    # cosine similarity (descriptors are L2 normalized) 
    matrix_similarity = des_src_normalize.T @ des_dest_normalize
    
    ind_col_matches = np.argmax(matrix_similarity, axis = 1)
        
    matches = np.concatenate((np.arange(0, des_src.shape[1]).reshape(-1,1), ind_col_matches.reshape(-1, 1)), axis = 1)
    final_matches = matches
    
    # FIND GOOD MATCHES:
    # Retrieve top 2 nearest neighbors 1->2.
    index_sorted = np.argsort(-matrix_similarity, axis = 1)[:, 0:2]

    matrix_distances = np.sqrt(2 - 2 * matrix_similarity)
    
    mask_good_matches = matrix_distances[list(range(0,matrix_distances.shape[0])), index_sorted[:, 0]] / matrix_distances[list(range(0,matrix_distances.shape[0])), index_sorted[:, 1]] < threshold
    
    if np.any(mask_good_matches):
        good_matches = matches[mask_good_matches, :]
        
        print("good matches/matches - %d/%d" % (good_matches.shape[0],matches.shape[0]))
        
        if good_matches.shape[0] > MIN_MATCH_COUNT:
            final_matches = good_matches
    
    # FIND MUTUAL AND GOOD MATCHES: 
    # Retrieve top 2 nearest neighbors 1->2.
    matches_12_top2 = np.argsort(-matrix_similarity, axis = 1)[:, 0:2]
    matches_12 = matches_12_top2[:, 0] # Save first NN and match similarity.
    
    matrix_distances = np.sqrt(2 - 2 * matrix_similarity)
    
    # Compute Lowe's ratio.
    mask1_good_matches = matrix_distances[list(range(0,matrix_distances.shape[0])), matches_12_top2[:, 0]] / matrix_distances[list(range(0,matrix_distances.shape[0])), matches_12_top2[:, 1]] < threshold

    # Retrieve top 2 nearest neighbors 1->2.
    matches_21_top2 = np.argsort(-matrix_similarity.T, axis = 1)[:, 0:2]
    matches_21 = matches_21_top2[:, 0] # Save first NN and match similarity.
    
    matrix_distances_T = np.sqrt(2 - 2 * matrix_similarity.T)
    
    # Compute Lowe's ratio.
    mask2_good_matches = matrix_distances_T[list(range(0,matrix_distances_T.shape[0])), matches_21_top2[:, 0]] / matrix_distances_T[list(range(0,matrix_distances_T.shape[0])), matches_21_top2[:, 1]] < threshold
        
    final_mask_good_matches = mask1_good_matches & mask2_good_matches[matches_12]
    
    # Mutual NN + symmetric ratio test.
    ids1 = np.arange(0, matrix_similarity.shape[0])
    
    mask_mutual_matches = (ids1 == matches_21[matches_12]) & final_mask_good_matches
    
    if np.any(mask_mutual_matches):
        mutual_matches = matches[mask_mutual_matches, :]
        
        if mutual_matches.shape[0] > MIN_MATCH_COUNT:
            final_matches = mutual_matches
        
        print("mutual and good matches/matches - %d/%d" % (mutual_matches.shape[0],matches.shape[0]))

    return final_matches


# In[3]:


def siftMatch(img1, sift_path_ref, sift_path_image, threshold = 0.75, N = 1):
    
    global NUM_MATCH, subsampling

    data_ref = scipy.io.loadmat(sift_path_ref)
    dst = data_ref['p'] # (2,N) numpy array, where N is the total number of keypoints
    des_dest = data_ref['d'] # (128,N) numpy array, where N is the total number of keypoints
        
    data_image = scipy.io.loadmat(sift_path_image)
    src = data_image['p'] # (2,N) numpy array, where N is the total number of keypoints
    des_src = data_image['d'] # (128,N) numpy array, where N is the total number of keypoints
        
    if subsampling:
        h, w, _ = img1.shape
            
        h_subsampling = math.floor(h/4) 
            
        w_subsampling = math.floor(w * h/(NUM_MATCH*h_subsampling)) 
            
        regions_h = range(0, h+1, h_subsampling)
        regions_w = range(0, w+1, w_subsampling)
            
        des_src_subsampling = np.array([], dtype=np.int64).reshape(des_src.shape[0],0)
        src_subsampling = np.array([], dtype=np.int64).reshape(src.shape[0],0)
            
        id_descriptor = np.arange(des_src.shape[1])
            
        for i in range(len(regions_h)-1):
            h_region_min = regions_h[i]
            h_region_max = regions_h[i+1]-1
            
            for j in range(len(regions_w)-1):
                w_region_min = regions_w[j]
                w_region_max = regions_w[j+1]-1
                    
                ind_keypoints_region = (src[0,:] > w_region_min) & (src[0,:] < w_region_max) & (src[1,:] > h_region_min) & (src[1,:] < h_region_max)
                    
                if np.any(ind_keypoints_region):
                    if len(ind_keypoints_region[ind_keypoints_region == True]) < N:
                        num_sampling = len(ind_keypoints_region[ind_keypoints_region == True])
                            
                    else:
                        num_sampling = N 
                        
                    ind_d_des = random.sample(list(id_descriptor[ind_keypoints_region]), num_sampling)
                        
                    des_src_subsampling = np.concatenate((des_src_subsampling, des_src[:, ind_d_des]), axis = 1)
                    src_subsampling = np.concatenate((src_subsampling, src[:, ind_d_des]), axis = 1)
            
        des_src = des_src_subsampling
        src = src_subsampling
            
    m =match_features(des_src,des_dest, threshold)
        
    matches_coords = np.concatenate((src[:, m[:,0]], dst[:, m[:, 1]]))
        
    src_pts = matches_coords[0:2, :].T
    dst_pts = matches_coords[2:4, :].T
            
    return src_pts, dst_pts


# In[4]:


def FitHomography(selected_matches, N = 4):
    """ Compute the fitted homography matrix by using N match pairs
  
   [u]     [X]
   [v] = H [Y], 
   [1]     [1]
   being H a 3x3 matrix 
  
   This can be arranged in a system Ax = 0, where x is a column vector with 
   the parameters of the homography, and A is given by:
   A = [X Y 1 0 0 0 -u.X -u.Y -u]
       [0 0 0 X Y 1 -v.X -v.Y -v]

   For N matches, the above matrix is vertically stacked, with 2 rows per match 
  """
    
    X = selected_matches[:,0]
    Y = selected_matches[:,1]
    u = selected_matches[:,2]
    v = selected_matches[:,3]
    
    A = []
    
    for i in range(N):
        row_1 = np.array([X[i], Y[i], 1, 0, 0, 0, -X[i]*u[i], -Y[i]*u[i], -u[i]])
        row_2 = np.array([0, 0, 0, X[i], Y[i], 1, -X[i]*v[i], -Y[i]*v[i], -v[i]])
        
        A.append(row_1)
        A.append(row_2)
    
    A = np.array(A)
    
    # V = eigvec(A.T @ A), being V.T obtained through Singular Value Decomposition (SVD)
    _, _, vT = np.linalg.svd(A)

  # vT is a 9×9 matrix
  # the solution x is the eigenvector corresponding to the smallest eigenvalue, 
  # that is, the eigenvector corresponding to the minimum singular value, 
  # leading to a row vector of 9 columns. Thus, to obtain the calibrated 
  # homography H, the final solution is to reshape the obtained vector into a 
  # 3x3 matrix 
    
    H = np.reshape(vT[-1,:], (3,3))
    
    # normalized homography, dividing by the element at (2,2)
    H = H/H[2,2]
    
    return H 


# In[5]:


def get_errors(all_matches, H):
    
    """Compute error or distance between original points and transformed by H. 
   Return an array of errors for all points"""
    
    num_matches = len(all_matches)
    
    X = all_matches[:,0].reshape(-1, 1)
    Y = all_matches[:,1].reshape(-1, 1)
    u = all_matches[:,2].reshape(-1, 1)
    v = all_matches[:,3].reshape(-1, 1)
    
    # all matching points in source image
    all_p1 = np.concatenate((X, Y, np.ones((len(all_matches),1))), axis = 1)
    
    # all matching points in template image
    all_p2 = np.concatenate((u, v), axis = 1)
    
    # Transform every point in p1 to estimate p2
    estimate_p2homogeneous = H @ all_p1.T
    
    estimate_p2euclidean = (estimate_p2homogeneous/(estimate_p2homogeneous[-1]))[0:2]
    
    # Compute error of each matching pair
    errors = np.linalg.norm(all_p2 - estimate_p2euclidean.T, axis = 1) 
    
    return errors


# In[6]:


def GetHomographyRANSAC(match_coords):
    
    """Function that computes linear (2D) Homography Calibration, implementing RANSAC
  for eliminating outliers and align correspondent matches. The main output concerns 
  a single transformation H that gets the most inliers in the course of all the 
  iterations. 
  
    Args:
        match_coords(numpy.ndarray): In dims (#matched pixels, 4).

    Returns:
        H(numpy.ndarray): Homography matrix, dims (3, 3).
    """
    
    global MAX_ITER, threshold_error
    
    N = 4 # four matches to initialize the homography in each iteration
    
    max_inliers = 0 
    
    # RANSAC procedure
    for itr in range(MAX_ITER): 
        
        # Randomly select 4 matched pairs
        idx_rand_inliers = random.sample(range(match_coords.shape[0]), N)
        
        selected_matches = match_coords[idx_rand_inliers, :]
        
        # compute the homography H by DLT from the N = 4 matched pairs 
        H = FitHomography(selected_matches)
        
        # Find inliners 
        errors = get_errors(match_coords, H)
        
        idx_inliers = np.where(errors < threshold_error)[0]
        
        num_inliers = len(idx_inliers) 
        
        # Analise current solution, and if it contains the maximum number of inliers
        # amongst all homographies until now fitted, save the current inliers for 
        # further refinement of the homography in the last step 
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = match_coords[idx_inliers]
            
    # compute the homography H by DLT from best_inliers  
    H = FitHomography(best_inliers, max_inliers)
    
    return H


# In[7]:


def Check_Homography(image, H, sift_paths, i):
    """
  Check if homography is reasonable, according to certain criteria:
  - If the determinant of the homography det(H) is very close to 0, H is 
   close to singular;
  
  - If condition number of H (ratio of the first-to-last singular value) is
    infinite, the matrix H is singular, and if it is too large, H is 
    ill-conditioned. In non-mathematical terms, an ill-conditioned problem 
    is one where, for a small change in the inputs, there is a large 
    change in the output, that is, H is very sensitive to changes or errors 
    in the input. This means that the correct solution/answer to the 
    equation becomes hard to find;
        
  - If det(H) < 0, the homography is not conserving the orientation, 
    being orientation-reversing. This is not suitable, except if we are 
    watching the object in a mirror. Nevertheless, sift descriptors 
    are not done to be mirror invariant, so if it was the case we would 
    probably not have good maches. 
        
  An exactly singular matrix means that it is not invertible. If the above 
  criteria is verified, more pratically the matrix H is non-invertible. 
  In the context of homographies, it means that points in one 2D image are mapped
  to a less-than-2D subspace in the other image (a line, a point). A 
  nearly singular matrix is indicative of a rather extreme warp.  

  """
    
    global last_H_good, last_iter_good
    
    #Conditions to accertain that the resultant homography H is free of 
    #singularities. If one of the condition is satisfied, the Homography H from 
    #image space to template image space is not reasonable, according 
    #to the defined criteria 
    if  np.linalg.det(H) <= 20 or np.linalg.cond(H[0:2, 0:2]) >= 3.25:
        print('searching for a reasonable H')
        
        # In the condition number, only the top-left 2x2 matrix is considered, 
        # thus omitting the z-dependence of the transformation, which should be 
        # irrelevant because we know that z will always be fixed to 1 on the input
        
        if last_iter_good != 0:
            # if in the current iteration no reasonable homography was estimated, 
            # the last homography that was found reasonable is considered 
            
            sift_path2 = sift_paths[last_iter_good]
            
            sift_path1 = sift_paths[i]
            
            # homography from I2 to template 
            H_I2_template = last_H_good
            
            # match points between image (source space) and I2 (destination space)
            m_coords_img, m_coords_temp = siftMatch(image, sift_path2, sift_path1, N = 4)
            
            match_coords = np.append(m_coords_img, m_coords_temp, axis = 1)
            
            # homography from image space to I2 space 
            H_image_I2 = GetHomographyRANSAC(match_coords)
            
            # Conjugate all the performed transformations from original to final warped
            # image (image space--> H_image_I2 --> I2 space --> H_I2_template --> final warp image)
            H = H_I2_template@H_image_I2
            
        # check if obtained homography is reasonable 
        if np.linalg.det(H) > 20 and np.linalg.cond(H[0:2, 0:2]) < 3.25:
            last_iter_good = i 
            last_H_good = H
            
            # If the above strategy was not successful in estimating an homography between
            # image and template, set H to 0 
        else:
            H = np.zeros((3,3))
        
    # Homography H from image space to template space is reasonable, according 
    # to the defined criteria 
    else:
        last_iter_good = i 
        last_H_good = H
    
    return H


# In[8]:


def pivproject2022_task1(path_to_template_folder, path_to_input_folder, path_to_output_folder):
    """
    Compute the homography between images in a directory and a template

    path_to_template_folder:  string with the path to a folder with both the 
                               a jpg file for the template image
                               and a mat file with the sift descriptors
    
    path_to_input_folder: string with the path to the input folder, where input images 
                           and keypoints are stored. Images are named rgb_number.jpg 
                           (or rgb_number.png) and corresponding keypoints are named 
                           rgbsift_number.mat

    path_to_output_folder: string with the path where homographies between images and 
    the template are stored
    """
    
    global last_H_good, last_iter_good

    # Check if path_to_input_folder was passed. If not, "No_path" is assigned
    if not('path_to_input_folder' in locals()):
        path_to_input_folder = "No_path";
        
    # Check if output directory exists. If not, output directory is created 
    if not(os.path.isdir(path_to_output_folder)):
        os.mkdir(path_to_output_folder)
    
    # Get input rgb images
    rgb_paths = []
    sift_paths = []

    for im_path in glob.glob(path_to_input_folder+'/*.jpg'):
        rgb_paths.append(im_path)
    
    if len(rgb_paths) == 0:
        print('ERROR: In the specified path there aren\'t image input files')
        return 
    
    else: 
        #Ordering the rgb_paths array, in such a way that consecutive frames follow 
        #each other  
        image_paths = sorted(rgb_paths)  
    
    print('Searching for sift .mat files')
        
    for im_path in glob.glob(path_to_input_folder+'/*.mat'):
        sift_paths.append(im_path)
    
    if len(sift_paths) != 0:
        sift_paths_ordered = sorted(sift_paths)
    
    else:
        print('ERROR: In the specified path there aren\'t sift input files.')
        return  

    # Get template image
    try:
        print('Searching for template')
        
        for path in glob.glob(path_to_template_folder+'/*.jpg'):
            template_path = path
            template = img.imread(template_path)
            
    except:
        print('ERROR: ERROR: In the specified path there isn\'t a template image.')
        return
    
    try:
        sift_template_path = template_path[:-4] + '.mat'

    except:
        print('ERROR: In the specified path there is not a sift input file for the template.')
        return
        
    last_H_good = 0 
    last_iter_good = 0 # variable for saving the last iteration where a reasonable 
                      # homography was estimated 
    
    print('Calculating projections')

    for i in range(len(image_paths)):
        print(str(i + 1), '/', str(len(image_paths)), 'Image to be processed')
        
        image_path = image_paths[i]
        image = img.imread(image_path)
        
        sift_path = sift_paths_ordered[i]
        
        try:
            # coordinates of the matches between image and template 
            m_coords_img, m_coords_temp = siftMatch(image, sift_template_path, sift_path, N = 4) 
            match_coords = np.append(m_coords_img, m_coords_temp, axis = 1)
        
        except:
            print('ERROR: check format of directory, as OpenCV only accepts ASCII characters for image paths')
            return 
        
        try:
            H = GetHomographyRANSAC(match_coords)

            print("np.linalg.det(H):", np.linalg.det(H))
            print("np.linalg.cond(H[0:2, 0:2]):", np.linalg.cond(H[0:2, 0:2]))
            
            # Check if homography is reasonable, according to certain criteria
            H = Check_Homography(image, H, sift_paths_ordered, i)
            
            # Saving outputs 
            file_name = os.path.split(image_path)[1]
            H_output_path = path_to_output_folder + '/' + 'H_' + file_name[4:8] + '.mat'
            scipy.io.savemat(H_output_path, {'H':H})
        
        except: 
            print('ERROR: RANSAC failed to compute homography. Check if there are enough matching keypoints.')
    
    print('All projections calculated.')


# In[9]:


path_to_template_folder = sys.argv[1]
path_to_input_folder = sys.argv[2]
path_to_output_folder = sys.argv[3]

pivproject2022_task1(path_to_template_folder, path_to_input_folder, path_to_output_folder)


