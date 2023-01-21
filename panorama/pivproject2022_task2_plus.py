#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
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
import matplotlib.image as img
import sys 

# global variables:
MIN_MATCH_COUNT = 24 # minimum number of matches required 
MAX_ITER = 5000 # number of iterations in ransac
NUM_MATCH = 500 # total number of matches that we want to subsample from different locations of the image.
                # For every region in the image analysed, if it contains at least a descriptor, and 
                # if the number of descriptors subsampled is fixed to N = 1, there will be 500 matches. 
                # Nevertheless, some regions of the image can have no descriptors, so the total number of matches will be reduced. 
                # For solving this, we can sample N > 1 descriptors from the regions of the image analysed

threshold_error = 4 # threshold error for points to be considered inliers, when performing ransac


# In[3]:


def match_features(des_src, des_dest, threshold):
    '''
    Implements the Nearest Neighbor Distance Ratio Test (NNDR) - Equation 4.18 in Section 4.1.3 of 
    Szeliski - to assign matches between interest points in two images. It also searches for mutual 
    matches and applies the NNDR test
  
    A match is between a feature in des_src and a feature in des_dest. We can
    represent this match as the index of the feature in des_src and the index
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


# In[4]:


def siftMatch(img1, img2, sift_path_ref, sift_path_image, threshold = 0.75, N = 1):
    
    global extract_sift, NUM_MATCH, subsampling
    
    if extract_sift:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        m =match_features(des1.T,des2.T, threshold)
        src_pts = np.float32([kp1[i].pt for i in m[:,0]]).reshape(-1,2)
        dst_pts = np.float32([kp2[i].pt for i in m[:,1]]).reshape(-1,2)
    
    else:
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


# In[5]:


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
    
    # normalized homography, dividing by the element at (3,3)
    H = H/H[2,2]
    
    return H 


# In[6]:


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


# In[7]:


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


# In[8]:


def compute_H_wrt_reference(H_all, ref_image):
    """
  Function that computes new homographies H_map that map every other image *directly* to
  the reference image by composing H matrices in H_all. 
  The homography in H_map that is associated with the reference image
  should be the identity matrix, created using eye(3). The homographies in
  H_map for the other images (both before and after the reference image)
  are computed by using already defined matrices in H_map and H_all. 

  Args: 
      H_all(cell array) 

      ref_image(int): index of the reference image (the first image has index 1)


  Returns:
      H_map(cell array): 3x3 homographies matrices that map each image into the reference image's
  coordinate system.

  """
    num_imgs = len(H_all)+1
    
    H_map = {}
    
    key = "H{}{}".format(ref_image-1, ref_image-1) 
    H_map[key] = np.eye(3)
    
    for i in range(0, ref_image-1): 
        key = "H{}{}".format(i, ref_image-1)  
        H_aux = np.eye(3)
        
        j = i
        
        while j < ref_image - 1:
            key_t = "H{}{}".format(j, j+1)
            H_aux = H_all[key_t] @ H_aux
            j += 1 
        
        H_map[key] = H_aux
    
    for i in range(ref_image, num_imgs):
        key = "H{}{}".format(i, ref_image-1)  # H10
        H_aux = np.eye(3)
        
        j = i -1 
        
        while j>= ref_image-1:
            key_t = "H{}{}".format(j, j+1)
            H_inv = np.linalg.inv(H_all[key_t])
            H_aux = H_inv/H_inv[2,2] @ H_aux
            j -= 1
        
        H_map[key] = H_aux
    
    return H_map 


# In[9]:


def Check_Homography(H):
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
    watching the object in a mirror. Nevertheless, sift/surf descriptors 
    are not done to be mirror invariant, so if it was the case we would 
    probably not have good maches. 
        
  An exactly singular matrix means that it is not invertible. If the above 
  criteria is verified, more pratically the matrix H is non-invertible. 
  In the context of homographies, it means that points in one 2D image are mapped
  to a less-than-2D subspace in the other image (a line, a point).A 
  nearly singular matrix is indicative of a rather extreme warp.  
  """
    
    #Conditions to accertain that the resultant homography H is free of 
    #singularities. If one of the condition is satisfied, the Homography H from 
    #image space to reference image space is not reasonable, according 
    #to the defined criteria 
    if  np.linalg.det(H) < 1 and np.linalg.cond(H[0:2, 0:2]) > 2:
        
        # In the condition number, only the top-left 2x2 matrix is considered, 
        # thus omitting the z-dependence of the transformation, which should be 
        # irrelevant because we know that z will always be fixed to 1 on the input
        
        H = np.zeros((3,3))
    
    return H


# In[10]:


def get_blank_canvas(H_warp, ind_image_warp, image_paths, ref_img):
    """
  Function that computes the size of the panorama using forward warping. Before warping 
  each of the images, the size of the output panorama image is computed and 
  initialized from the range of warped image coordinates for each input image.
  """
    
    num_imgs = len(H_warp)
    
    # Initialize the limits of the output panorama image
    min_crd_canvas = np.array([np.inf, np.inf])
    max_crd_canvas = np.array([-np.inf, -np.inf])
    
    limits_all = []
    
    # mapping the coordinates of the four corners from each source image using forward warping to determine its coordinates in 
    # the output image. 
    
    for i in range(num_imgs):
        ind_image = ind_image_warp[i]
        
        image = img.imread(image_paths[ind_image])
        img_h, img_w, _ = image.shape
        
        key = "H{}{}".format(ind_image, ref_img-1)
        H = H_warp[key]
        
        # create a matrix with the coordinates (homogeneous) of the four corners 
        # of the current image
        corners_img = np.array([[0, 0, 1], [0, img_h,1], [img_w, img_h,1], [img_w, 0,1]])
            
        # Map each of the 4 corner's coordinates into the coordinate system of
        # the reference image
        canvas_crd_corners = H @ corners_img.T
        canvas_crd_corners = (canvas_crd_corners / canvas_crd_corners[-1, :])[0:2, :]
            
        limits_all.append(canvas_crd_corners.T)
            
        # Limits of the current warped image 
        min_crd_canvas_cur = np.amin(canvas_crd_corners.T, axis=0) # min_x, min_y
        max_crd_canvas_cur = np.amax(canvas_crd_corners.T, axis=0) # max_x, max_y
            
        # Update the limits of the output image 
        min_crd_canvas = np.floor(np.minimum(min_crd_canvas_cur, min_crd_canvas)) # min_x, min_y
        max_crd_canvas = np.ceil(np.maximum(max_crd_canvas_cur, max_crd_canvas)) # max_x, max_y
        
    # Compute output image size 
    min_x = min_crd_canvas[0]
    max_x = max_crd_canvas[0]
    min_y = min_crd_canvas[1]
    max_y = max_crd_canvas[1]
    
    width_canvas = max_x - min_x + 1
    height_canvas = max_y - min_y + 1
    
    # output image array initialized to all black pixels
    canvas_img = np.zeros((int(height_canvas), int(width_canvas), 3), dtype=np.int64)
    
    # Compute offset of the upper-left corner of the reference image relative
    # to the upper-left corner of the output image
    offset = min_crd_canvas.astype(np.int64) # [x_offset, y_offset]
    
    # Find limits of panorama
    lims = np.concatenate(limits_all,axis=0)
    
    for i in range(int(lims.shape[0]/4)):
        lims_i = np.concatenate((lims[4*i:4 + 4*i], lims[None, 4*i, :]), axis = 0)
        plt.plot(lims_i[:, 0], -lims_i[:,1])

    plt.show()

    return canvas_img, offset


# In[11]:


def image_warping(panorama_height, panorama_width, offset, H, img):
    """
  Function that warps every input image on to the panorama, using inverse warping
  to map each pixel in the output image into the planes defined by the source images. 
  If forward warping was used to map every pixel from each source image, there will 
  be holes (i.e., some pixels in the output image will not be assigned an RGB 
  value from any source image, and remain black) in the final output image.
  """
    
    x_offset = -offset[0]
    y_offset = -offset[1]
    
    # Create a list of all pixels' coordinates in output image
    x,y = np.meshgrid(range(panorama_width), range(panorama_height))
    
    # Create homogeneous coordinates for each pixel in output image, considering 
    # the translation offset vector 
    x_coords = x.flatten().reshape(1,-1) - x_offset
    y_coords = y.flatten().reshape(1,-1) - y_offset
    
    grid_coords = np.concatenate((x_coords, y_coords, np.ones((1, x_coords.shape[1]))))
    
    # Perform inverse warp to compute coordinates in current input image
    image_coords = np.linalg.solve(H, grid_coords)
    
    # To get the warped coordinates, we must divide the first and second coordinates by 
    # z to obtain the new x and y (euclidean coordinates)
    z = image_coords[None, 2, :]  
    warp_coords = image_coords[0:2, :]/np.concatenate((z,z))
    
    # Reshape the pixel grid to have the same size as the panorama 
    x_warp = np.reshape(warp_coords[None, 0, :], (panorama_height, panorama_width))
    y_warp = np.reshape(warp_coords[None, 1, :], (panorama_height, panorama_width))
    # Note:  Some values will return as NaN ("not a number") because they
    # map to points outside the domain of the input image
    
    # Warped Image array that will contain RGB color maps obtained through inverse 
    # mapping 
    I_WarpColorMaps = np.zeros((panorama_height, panorama_width, 3))
    
    # Color interpolation, by sampling a color value for each pixel in source image. 
    # By doing this we won't have any black pixels or gaps in the warpped image, 
    # (a kind of undersampling artifact)
    
    for channel in range(3):
        # When mapping pixel locations,  some pixels in the output warped image will not 
        # map to a pixel in a given source image because the output pixel’s coordinates 
        # map outside the domain of the source image.  For solving this, we use bilinear 
        # interpolation (order = 1) for assigning the color value in the warped image
        
        I_WarpColorMaps[:, :, channel] = scipy.ndimage.map_coordinates(img[:, :, channel].astype(float), [y_warp, x_warp], order = 1)
    
    # color pixel warping, by converting I_WarpColorMaps into an unsigned 8-bit integer, 
    # with the elements of an uint8 ranging from 0 to 255
    warped_image = I_WarpColorMaps.astype('uint8')

    return warped_image


# In[12]:


def alpha_channel(img, epsilon=0.001):
    """
  Function that computes the alpha channel of an RGB image.

  Args:
     img is an RGB image. 

     epsilon (float): value to guarantee that the alpha channel has non-zero
     values, otherwise a division-by-zero error will be encounter 
     when performing blending 
  
  Returns:
     im_alpha has the same size as im_input. Its intensity is between
     epsilon and 1, inclusive.
  """
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # binary image that has 1s within the warped image and 0s beyond the edges of 
    # the warped image 
    im_bw = cv2.threshold(img_gray, 0.5, 255, cv2.THRESH_BINARY)[1]
    
    # alpha channel where the value of alpha for the input image is 1 at its 
    # center pixel and decreases linearly to epsilon  at all the border pixels
    im_alpha = scipy.ndimage.distance_transform_edt(im_bw)
    
    # normalize the distances to be in the interval [epsilon, 1].
    im_alpha = (im_alpha+epsilon)/np.max(im_alpha) 
    
    return im_alpha


# In[16]:


def blending(img1, img2):
    """
  Function that blends two warped images together, overlapping pixel color values. 
  The simplest way to create a final composite is by averaging the pixel values 
  where the two images overlap, or by using the pixel values from one of the 
  images.
  Simple averaging usually does not work very well, since exposure differences, ´
  misregistrations, and scene movement are all very visible. 
  A better approach to averaging is to weight pixels near the center of the 
  image more heavily and to down-weight pixels near the edges, being this 
  enconded into a alpha channel. 
  This is called feathering (Section 9.3.2 in the Szeliski book)

  Each pixel (x, y) in image Ii is represented as 
  Ii(x, y) = (αi*R, αi*G, αi*B, αi) where (R,G,B) are the color values at the 
  pixel and αi its alpha channel 
  
  Pixel value of (x, y) in the stitched output image is computed has:
  [(α1*R, α1*G, α1*B) + (α2*R, α2*G,α2*B) ] / (α1+α2).

  Args:
  img1 and img2 are both RGB images of the same size, having 
  been warped to the same coordinate frame 
  
  Output:
  im_blended has the same size and data type as the input images

  """
    
    if feathering:
        # alpha channel that contains weights for blending the images 
        alpha1 = alpha_channel(img1)
        alpha2 = alpha_channel(img2)
        
        im_blended = np.zeros(img1.shape)
        
        red_blending = (alpha1 * img1[:, :, 0] + alpha2 * img2[:,:,0])/(alpha1 + alpha2)
        green_blending = (alpha1 * img1[:, :, 1] + alpha2 * img2[:,:,1])/(alpha1 + alpha2)
        blue_blending = (alpha1 * img1[:, :, 2] + alpha2 * img2[:,:,2])/(alpha1 + alpha2)
        
        im_blended[:,:,0] = red_blending 
        im_blended[:,:,1] = green_blending 
        im_blended[:,:,2] = blue_blending
        
        # convert into an unsigned 8-bit integer, for the values of each channel to
        # range from 0 to 255
        im_blended = im_blended.astype('uint8')
    
    else:
        # average blending 
        
        mask_a = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        mask_b = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        a_and_b = cv2.bitwise_and(mask_a, mask_b)
        overlap_area_mask = cv2.threshold(a_and_b, 1, 255, cv2.THRESH_BINARY)[1]
        
        overlap_pixels = (cv2.bitwise_and(img1, img1, mask = overlap_area_mask.astype('uint8')) + cv2.bitwise_and(img2, img2, mask = overlap_area_mask.astype('uint8')))/2
        
        im_blended = cv2.bitwise_and(img1, img1, mask = np.logical_not(overlap_area_mask).astype('uint8')) + cv2.bitwise_and(img2, img2, mask = np.logical_not(overlap_area_mask).astype('uint8')) + overlap_pixels
                
        im_blended = im_blended.astype('uint8')
                
    return im_blended


# In[13]:


def pivproject2022_task2_plus(ref_image, path_to_input_folder, path_to_output_folder, extract_sift, subsampling, cv2WarpPerspective, feathering):
    """
  Compute the homographies between images in a directory and a reference image

    ref_image: integer with index number of the frame that will be the reference 
    image. First image is index=1. Ideally, the one in the middle of the sequence 
    of input images so that there is less distortion resulting mosaic.
    
    path_to_input_folder: string with the path to the input folder, where input images 
                           and keypoints are stored. Images are named rgb_number.jpg 
                           (or rgb_number.png) and corresponding keypoints are named 
                           rgbsift_number.mat

    
    path_to_output_folder: string with the path where homographies with respect to the reference
    image are stored

  """
    
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
    
    if not(extract_sift):
        print('Searching for sift .mat files')
        
        for im_path in glob.glob(path_to_input_folder+'/*.mat'):
            sift_paths.append(im_path)
            
            if len(sift_paths) != 0:
                
                sift_paths_ordered = sorted(sift_paths)
            
            else:
                extract_sift = True
                print('In the specified path there aren\'t sift input files. Thus, a sift function will be used to extract matching points')

    # Get Reference image
    try: 
        reference_image = img.imread(image_paths[ref_image - 1])
        
    except:
        print('ERROR: The index for the reference image is out of bounds. Please select an index between 1 and %d' % (len(rgb_paths)))
        return
    
    H_all = {}
        
    # compute homography matrices between adjacent input images. Homography matrices 
    # between adjacent input images are then stored in a cell array H_all.
    
    for i in range(len(image_paths)-1):
        image_1_path = image_paths[i]
        image_2_path = image_paths[i+1]
        
        print("Processing {} & {}".format(image_1_path, image_2_path))
        
        image_1 = img.imread(image_1_path)
        image_2 = img.imread(image_2_path)
        
        if extract_sift:
            sift_path1 = None
            sift_path2 = None
        
        else:
            sift_path1 = sift_paths_ordered[i]
            sift_path2 = sift_paths_ordered[i+1]
        
        key = 'H{}{}'.format(i, i+1)
        
        #try:
            # coordinates of the matches between image and template
        m_coords_img, m_coords_temp = siftMatch(image_1, image_2, sift_path2, sift_path1, N = 4)
        match_coords = np.append(m_coords_img, m_coords_temp, axis = 1)
        #except:
        #    print('ERROR: check format of directory, as OpenCV only accepts ASCII characters for image paths')
        #    return 
        
        try:
            threshold_error = 4
            
            H_all[key] = GetHomographyRANSAC(match_coords)
            
        except: 
            print('ERROR: RANSAC failed to compute homography. Check if there are enough matching keypoints.')
            
    
    # Compute new homographies H_map that map every other image *directly* to
    # the reference image 
    H_map = compute_H_wrt_reference(H_all, ref_image)
     
        
    H_warp = {} # cell array that will contain the reasonable homographies between images and reference image, 
                # so that only these images are warped 
        
    ind_image_warp = [] # list that will contain the index of the images that will be warped 
    
    for i in range(len(H_map)):
        key = "H{}{}".format(i, ref_image-1)
        H = H_map[key]
        
        print("image "+str(i))
        print("np.linalg.det(H):", np.linalg.det(H)) 
        print("np.linalg.cond(H[0:2, 0:2]):", np.linalg.cond(H[0:2, 0:2]))
            
        H = Check_Homography(H)

        if np.array_equal(H, np.zeros((3,3))):
            print("H{}{} is not reasonable".format(i, ref_image-1))
        else:
            print("H{}{} is reasonable".format(i, ref_image-1))
            H_warp[key] = H
            ind_image_warp.append(i)

        # saving homographies with respect to the reference image 
        file_name = os.path.split(image_paths[i])[1]
        H_output_path = path_to_output_folder + '/' + 'H_' + file_name[4:8] + '.mat'
        scipy.io.savemat(H_output_path, {'H':H})
        
    canvas_img, offset = get_blank_canvas(H_warp, ind_image_warp, image_paths, ref_image)
    
    panorama_height, panorama_width, _ = canvas_img.shape
    
    # cell array that contains warped input images on the output canvas panorama 
    warped_images = {}
    
    for i in range(len(H_warp)):
        ind_image = ind_image_warp[i]
                       
        key = "H{}{}".format(ind_image, ref_image-1)
        H = H_warp[key]
        
        image = cv2.imread(image_paths[ind_image])
        
        if cv2WarpPerspective:
            # Combine homography with the translation offset vector 
            translation_mat = np.array([[1, 0, -offset[0]], [0, 1, -offset[1]], [0, 0, 1]])
            H = np.dot(translation_mat, H)
            warped_images[i] = cv2.warpPerspective(image, H, (canvas_img.shape[1], canvas_img.shape[0]), flags = cv2.INTER_NEAREST)
            
        else:
            warped_images[i] = image_warping(panorama_height, panorama_width, offset, H, image)
    
    # Initialize output image to black (0)
    panorama_image = np.zeros((panorama_height, panorama_width,3))
    
    panorama_image = warped_images[0] 
    
    for i in range(1,len(warped_images)):
        panorama_image = blending(np.float32(panorama_image), np.float32(warped_images[i]))
    
    plt.title("Panorama")
    plt.imshow(panorama_image)
    plt.show()
    
    # saving mosaic
    image_output_path = path_to_output_folder + '/mosaic_' + str(ref_image) + '.png'
    cv2.imwrite(image_output_path, panorama_image)
    


# In[14]:


ref_image = int(sys.argv[1])
path_to_input_folder = sys.argv[2]
path_to_output_folder = sys.argv[3]

extract_sift = int(sys.argv[4]) # variable that defines if the extraction of sift keypoints and descriptors is to 
                       # be performed (set to 1, 0 otherwise)

subsampling = int(sys.argv[5]) # variable that defines if subsampling of the source image 
                               # descriptors is to be performed (set to 1, 0 otherwise), when 
                               # performing the matching 
    
cv2WarpPerspective = int(sys.argv[6]) # variable that defines if image warping is to be performed by 
                                      # cv2WarpPerspective built-in function (set to 1) or a function 
                                      # developed by the group (set to 0)  

feathering = int(sys.argv[7]) # variable that defines if feathering is going to be performed (set to 1) or not 
                              # (set to 0) not for blending the warped images 


# In[48]:


pivproject2022_task2_plus(ref_image, path_to_input_folder, path_to_output_folder, extract_sift, subsampling, cv2WarpPerspective, feathering)

