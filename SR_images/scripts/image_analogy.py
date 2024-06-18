import numpy as np
import matplotlib.pyplot as plt
import copy
from annoy import AnnoyIndex
import random
import time
from skimage import data, io, filters, color, transform, util
from skimage.transform import pyramid_gaussian
from scipy import ndimage
import cv2

def create_image_pyramid(img, num_layers=5):
    return list(pyramid_gaussian(img, downscale=2, channel_axis=-1, max_layer=num_layers))[::-1]

def feature_vector(img_prime_hr, img_prime_lr, img_hr, img_lr, row, column, nighbourhood_size_hr=5, nighbourhood_size_lr=3):
    fv = []
    
    fv.extend(full_neighbourhood_feature_vector(img_lr, row//2, column//2, nighbourhood_size_lr))
    fv.extend(full_neighbourhood_feature_vector(img_hr, row, column, nighbourhood_size_hr))
    
    fv.extend(full_neighbourhood_feature_vector(img_prime_lr, row//2, column//2, nighbourhood_size_lr))
    fv.extend(half_neighbourhood_feature_vector(img_prime_hr, row, column, nighbourhood_size_hr))
    
    # g_fv = get_gaussian_feature_vector()
    
    return fv # np.array(fv) * np.array(g_fv)

def full_neighbourhood_feature_vector(img, row, column, neighbourhood_size, feature_extractor=lambda x : [x[0]], normalise=False):
    if neighbourhood_size % 2 == 0:
        raise ValueError("Neighbourhood size must be positive odd integer")
        
    img_height, img_width, channel_length = img.shape
        
    radius = neighbourhood_size // 2
    vec = []
    
    for i in range(row-radius, row+radius+1):
        for j in range(column-radius, column+radius+1):
            vec.extend(feature_extractor(img[i, j, :] if 0 <= i and i < img_height and 0 <= j and j < img_width else [0] * channel_length))

    #return np.array(vec) #/ len(vec)
    return vec # / np.linalg.norm(vec) if normalise and np.linalg.norm(vec) > 0.0 else vec

def half_neighbourhood_feature_vector(img, row, column, neighbourhood_size, feature_extractor=lambda x : [x[0]], normalise=False):
    if neighbourhood_size % 2 == 0:
        raise ValueError("Neighbourhood size must be positive odd integer")
        
    img_height, img_width, channel_length = img.shape
        
    radius = neighbourhood_size // 2
    vec = []
    
    for i in range(row-radius, row+radius+1):
        for j in range(column-radius, column+radius+1):
            if i == row and j == column:
                #return np.array(vec) #/ len(vec)
                return vec # / np.linalg.norm(vec) if normalise and np.linalg.norm(vec) > 0.0 else vec
            vec.extend(feature_extractor (img[i, j, :] if 0 <= i and i < img_height and 0 <= j and j < img_width else [0] * channel_length))
    raise Exception("Something went wrong")

class ANN_Searcher:
    def __init__(self):
        self.img_sizes = [0]
        self.img_widths = []
        
    def count_to_coord(self, count):
        i = 1
        while count >= self.img_sizes[i]:
            i+=1
        i -= 1
        
        count_within_image = count - self.img_sizes[i]
        return count_within_image // self.img_widths[i], count_within_image % self.img_widths[i], i
        ## RETURN: row, column, image_index
    
    def coord_to_count(self, width, row, column, img_id):
        return row * width + column + self.img_sizes[img_id]
    
    def create_ann_feature_structure(self, A_images, feature_length, metric="euclidean", number_trees=30):
        self.ai = AnnoyIndex(feature_length, metric)
        
        for i in range(len(A_images)):
            height, width, channels = A_images[i].shape
            self.img_widths.append(width)
            for row in range(height):
                for column in range(width):
                    count = self.coord_to_count(width, row, column, i)
                    self.ai.add_item(count, full_neighbourhood_feature_vector(A_images[i], row, column, 5, normalise=False))
            self.img_sizes.append(height * width + self.img_sizes[-1])
            
        self.ai.build(number_trees)
    
    def query(self, B_feature):
        count = self.ai.get_nns_by_vector(B_feature, 1)[0]
        return self.count_to_coord(count)

def coord_to_count(row, column, width):
    return row * width + column

def count_to_coord(count, width):
    return count // width, count % width

def create_ann_feature_structure(img, feature_length, metric="euclidean", number_trees=30):
    ai = AnnoyIndex(feature_length, metric)  # Length of item vector that will be indexed
    height, width, _ = img.shape
    # g_fv = get_gaussian_feature_vector()
    for row in range(height):
        for col in range(width):
            count = coord_to_count(row, col, width)
            ai.add_item(count, full_neighbourhood_feature_vector(img, row, col, 5, normalise=False))

    ai.build(number_trees)
    return ai
    #ai.save('test.ann')

def get_gaussian_feature_vector(large_neighbourhood_size=5, small_neighbourhood_size=3, std=0):
    large_kernel = np.zeros((large_neighbourhood_size, large_neighbourhood_size, 3))
    large_kernel[:, :, :] = (cv2.getGaussianKernel(large_neighbourhood_size, std).dot(cv2.getGaussianKernel(large_neighbourhood_size, std).transpose())).reshape(large_neighbourhood_size, large_neighbourhood_size, 1)
    small_kernel = np.zeros((small_neighbourhood_size, small_neighbourhood_size, 3))
    small_kernel[:, :, :] = (cv2.getGaussianKernel(small_neighbourhood_size, std).dot(cv2.getGaussianKernel(small_neighbourhood_size, std).transpose())).reshape(small_neighbourhood_size, small_neighbourhood_size, 1)
    fv = feature_vector(large_kernel, small_kernel, large_kernel, small_kernel, large_neighbourhood_size // 2, large_neighbourhood_size // 2)
    return fv


def get_dirac_delta(large_neighbourhood_size=5, small_neighbourhood_size=3):
    large_kernel = np.zeros((large_neighbourhood_size, large_neighbourhood_size, 3))
    large_kernel[large_neighbourhood_size//2, large_neighbourhood_size//2, :] = 1
    small_kernel = np.zeros((small_neighbourhood_size, small_neighbourhood_size, 3))
    small_kernel[small_neighbourhood_size//2, small_neighbourhood_size//2, :] = 1
    fv = feature_vector(large_kernel, small_kernel, large_kernel, small_kernel, large_neighbourhood_size // 2, large_neighbourhood_size // 2)
    return fv


def best_approximate_match(ANN, B_feature):
    return ANN.query(B_feature)



def best_coherence_match(A_prime_pyramid, A_pyramid, B_fv, s_pyramid, b_row, b_column, b_width, b_height, l):
    r_min = None
    r_star = None
    
    min_row = max(0, b_row - 2)
    max_row = min(b_height-1, b_row + 2)
    min_column = max(0, b_column - 2)
    max_column = min(b_width-1, b_column + 2)
    
    #A_height, A_width, _ = A_prime_pyramid[l].shape
    #gaussian_fv = get_dirac_delta()
    gaussian_fv = get_gaussian_feature_vector()
    
    for row in range(min_row, max_row+1):
        for column in range(min_column, max_column+1):
            r_row, r_column, r_img_idx = tuple(map(int, s_pyramid[l][row][column]))
            A_column = int(r_column + b_column - column)
            A_row = int(r_row + b_row - row)
            A_height, A_width, _ = A_prime_pyramid[l][r_img_idx].shape
            
            if A_column < 0 or A_column >= A_width or A_row < 0 or A_row >= A_height:
                #print("OOB A")
                continue
            sr_fv = feature_vector(A_prime_pyramid[l][r_img_idx], A_prime_pyramid[l-1][r_img_idx], A_pyramid[l][r_img_idx], A_pyramid[l-1][r_img_idx], A_row, A_column)
            r_min_temp = np.linalg.norm( np.array(sr_fv) - np.array(B_fv) )**2 #B_pyramids
            #r_min_temp = np.linalg.norm( (np.array(sr_fv) - np.array(B_fv)) * gaussian_fv )**2
            if r_min is None or r_min > r_min_temp:
                r_min = r_min_temp
                r_star = (A_row, A_column, r_img_idx)
            if row == b_row and column == b_column:
                return r_star
                
    return r_star

def best_match(A_prime_pyramid, A_pyramid, ANN, B_prime_pyramid, B_pyramid, s_pyramid, l, L, row, column, kappa):#, ana_pyramid):
    B_fv = feature_vector(B_prime_pyramid[l], B_prime_pyramid[l-1], B_pyramid[l], B_pyramid[l-1], row, column)
    b_height, b_width, _ = B_prime_pyramid[l].shape
    
    # A_height, A_width, _ = A_prime_pyramid[l].shape
    B_feature = full_neighbourhood_feature_vector(B_pyramid[l], row, column, 5, normalise=False)
    p_app_row, p_app_col, p_app_image_idx = best_approximate_match(ANN, B_feature)
    
    #if row < 4 or row >= b_height - 4 or column < 4 or column >= b_width - 4:
    #    ana_pyramid[l][row][column] = [255, 0, 0]
    #    return (p_app_row, p_app_col)
    
    p_coh_row, p_coh_col, p_coh_image_idx = best_coherence_match(A_prime_pyramid, A_pyramid, B_fv, s_pyramid, row, column, b_width, b_height, l)
    #p_coh_row, p_coh_col = brute_force_match(A_prime_pyramid, A_pyramid, B_fv, s_pyramid, row, column, b_width, b_height, l)
    
    ## Need to add gaussian weighting to these feature vectors
    gaussian_fv = get_gaussian_feature_vector(std=0.7)
    #gaussian_fv = get_dirac_delta()
    
    A_app_fv = np.array(feature_vector(A_prime_pyramid[l][p_app_image_idx], A_prime_pyramid[l-1][p_app_image_idx], A_pyramid[l][p_app_image_idx], A_pyramid[l-1][p_app_image_idx], p_app_row, p_app_col))
    A_coh_fv = np.array(feature_vector(A_prime_pyramid[l][p_coh_image_idx], A_prime_pyramid[l-1][p_coh_image_idx], A_pyramid[l][p_coh_image_idx], A_pyramid[l-1][p_coh_image_idx], p_coh_row, p_coh_col))
    
    #d_app = np.linalg.norm( np.array(A_app_fv) - np.array(B_fv) )**2
    #d_coh = np.linalg.norm( np.array(A_coh_fv) - np.array(B_fv) )**2
    d_app = np.linalg.norm( (np.array(A_app_fv) - np.array(B_fv)) * np.array(gaussian_fv) )
    d_coh = np.linalg.norm( (np.array(A_coh_fv) - np.array(B_fv)) * np.array(gaussian_fv) )
    
    if d_coh < d_app * (1+kappa*2**(-l-1)):# * (1+kappa*2**(l-L)):
        # ana_pyramid[l][row][column] = [0, 0, 255]
        #global score_coh
        #score_coh += 1
        return (p_coh_row, p_coh_col, p_coh_image_idx)
        #return (p_app_row, p_app_col)
    # ana_pyramid[l][row][column] = [255, 0, 0]
    #global score_app
    #score_app += 1
    #return (p_coh_row, p_coh_col, p_coh_image_idx)
    return (p_app_row, p_app_col, p_app_image_idx)


def create_image_analogy(A_images, B_image):
    L = 8 #5
    A = [A_images[i][0] for i in range(len(A_images))]
    A_prime = [A_images[i][1] for i in range(len(A_images))]
    
    
    A_pyramid = [create_image_pyramid(A_images[i][0], num_layers=L) for i in range(len(A_images))]


    A_pyramid = list(map(list, zip(*A_pyramid)))

    #print(len(A_pyramid))
    #return
    ## A_pyramid, each row is a level in the pyramid, each column is a separate pyramid
    A_prime_pyramid = [create_image_pyramid(A_images[i][1], num_layers=L) for i in range(len(A_images))]
    A_prime_pyramid = list(map(list, zip(*A_prime_pyramid)))
    
    B_pyramid = create_image_pyramid(B_image, num_layers=L)
    
    B_prime = np.zeros(B_image.shape)
    B_prime_copy = copy.copy(B_prime)
    B_prime_pyramid = create_image_pyramid(B_prime, num_layers=L)
    
    xdim, ydim, channels = B_image.shape
    s = np.zeros((xdim, ydim, 3)) ## row, column, image_idx
    s_pyramid = create_image_pyramid(s, num_layers=L)
    
    print("First step")
    height, width, _ = B_prime_pyramid[0].shape
    # A_height, A_width, _ = A_prime_pyramid[0].shape
    ANN= ANN_Searcher()
    ANN.create_ann_feature_structure(A_pyramid[0], 1 * 25)
    for row in range(height):
        for column in range(width):
            # p = best_match(A_pyramid, A_prime_pyramid, B, B_prime_pyramid, s, l, (row, column))
            #i, j = best_approximate_match(A_features, B_fv, row, column, A_width)
            
            B_feature = full_neighbourhood_feature_vector(B_pyramid[0], row, column, 5, normalise=False)
            # i, j, idx = best_approximate_match(A_features, B_feature, A_width)
            i, j, idx = ANN.query(B_feature)
            # best_match(A_prime_pyramid, A_pyramid, A_features, B_prime_pyramid, B_pyramid, s_pyramid, l, L-1, row, column, 2)

            #best_coherence_match

            B_prime_pyramid[0][row][column] = B_pyramid[0][row][column]
            B_prime_pyramid[0][row][column][0] = A_prime_pyramid[0][idx][i, j][0]
            s_pyramid[0][row][column] = [i, j, idx]
            #ana_pyramid[0][row][column] = [255, 0, 0]
    print("Completed first step")
    
    
    #print("Scores:")
    #print(score_coh)
    #print(score_app)
    
    for l in range(1, L+1):
        height, width, _ = B_prime_pyramid[l].shape
        # A_height, A_width, _ = A_prime_pyramid[0].shape
        ANN= ANN_Searcher()
        print("Stop")
        print(l)
        ANN.create_ann_feature_structure(A_pyramid[l], 1 * 25)
        for row in range(height):
            for column in range(width):

                B_feature = full_neighbourhood_feature_vector(B_pyramid[l], row, column, 5, normalise=False)
                i, j, idx = best_match(A_prime_pyramid, A_pyramid, ANN, B_prime_pyramid, B_pyramid, s_pyramid, l, L, row, column, 0.5)#0.5)
                B_prime_pyramid[l][row][column] = B_pyramid[l][row][column]
                B_prime_pyramid[l][row][column][0] = A_prime_pyramid[l][idx][i, j][0]
                s_pyramid[l][row][column] = [i, j, idx]
                #ana_pyramid[0][row][column] = [255, 0, 0]
        print("Completed first step")
        #print("Scores:")
        #print(score_coh)
        #print(score_app)
    
    return B_prime_pyramid, B_pyramid
    #     ana = np.zeros(B.shape)# np.random.rand(xdim, ydim, channels)
    #     ana_pyramid = create_image_pyramid(ana, num_layers=L)
