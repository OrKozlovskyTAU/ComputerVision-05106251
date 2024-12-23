"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        right_image_padded=np.pad(right_image,((0,0),(dsp_range,dsp_range),(0,0)),mode='constant',constant_values=0)
        for disparity_value in disparity_values:
        ## disparity value runs from 
            col_start = dsp_range+disparity_value
            col_end   = col_start+num_of_cols
            #print("working from %d to %d"%(col_start,col_end))
            dsp_image_diff=left_image-right_image_padded[:,col_start:col_end,:]
            #print(dsp_image_diff[:,:,0])
            #kernel = np.ones((win_size, win_size, 3))
            kernel = np.zeros((win_size, win_size, 3))
            kernel[:,:,1] = np.ones((win_size, win_size))
            result=convolve(dsp_image_diff**2, kernel, mode='constant', cval=0.0)
            ### bug - disparity_value runs from negative - this causes a shift in labels by dsp_range
            #ssdd_tensor[:,:,disparity_value] = np.sum(result, axis=2)
            ssdd_tensor[:,:,col_start] = np.sum(result, axis=2)
            """INSERT YOUR CODE HERE"""
            ssdd_tensor -= ssdd_tensor.min()
            ssdd_tensor /= ssdd_tensor.max()
            ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        #label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        label_no_smooth = np.argmin(ssdd_tensor[:,:,:,],axis=2)
        """INSERT YOUR CODE HERE"""
        return label_no_smooth


    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        """INSERT YOUR CODE HERE"""
        # Initialize the l_slice and m_slice matrices
        l_slice = np.zeros_like(c_slice)
        
        # Initialize the first column (l_slice and m_slice are initialized with the values of Cslice)
        l_slice[:, 0] = c_slice[:, 0]

        for col in range(1, num_of_cols):
            # m_slice is of square of size [num_labels x num_labels]
            m_slice = np.transpose(np.tile(l_slice[:, col-1], [num_labels, 1]))
            # prepare penalty map:
            #    has p2 on all location other the main diagonal and the 2 adjacent to it
            #    has p1 on the 2 diagonals adjacent to the main
            #    has 0 (no penalty) on the main diagonal
            penalty_mat = np.full((num_labels, num_labels), p2)
            np.fill_diagonal(penalty_mat[1:,:], p1)
            np.fill_diagonal(penalty_mat[:,1:], p1)
            np.fill_diagonal(penalty_mat, 0)
            m_slice += penalty_mat
            l_slice[:, col] = c_slice[:, col] + np.min(m_slice, axis=0) - np.min(l_slice[:, col - 1])
            
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        #######################################################################
        """INSERT YOUR CODE HERE"""
        c_slices = np.split(ssdd_tensor, ssdd_tensor.shape[0], axis=0)
        l_slices = [self.dp_grade_slice(c_slice.squeeze().T, p1, p2) for c_slice in c_slices]
        l = np.stack(l_slices, axis=2).T
        #######################################################################
        return self.naive_labeling(l) 

    @staticmethod
    def extract_slices(ssdd, direction):
        """ 
            :param ssdd: a (HxWxD) Tensor 
            :param direction: an int in range (1..8)
                   1 is west, 2 is north-west and it's continuing clockwise
            :return (slice_dict, index_dict) where:
                1. slice_dict is a dictionary that holds the 2d slice (keys are 0,1,..)
                2. index_dict is a dictionary thet holds the ssdd indices of the matching slice
        """ 
        (H,W,D) = np.shape(ssdd)
        (xx,yy) = np.meshgrid(range(W),range(H))
        # slicing is done with either: 
        #     1. python's array slicing in case the slice is 'straight' (i.e n/e/s/w)
        #     2. np.diagonal in case the slice is diagonal (i.e ne/nw/se/sw)
        # the ssdd matrix is flipped and/or transposed for that purpose
        if (direction==1):      # starting from row 0
            warp_xx = xx
            warp_yy = yy
        elif (direction==2):    # range(-H+1,W): starting from bottom-left-corner
            warp_xx = xx
            warp_yy = yy
        elif (direction==3):    # starting from col 0
            warp_xx = xx.T
            warp_yy = yy.T
        elif (direction==4):    # range(-H+1,W): starting from bottom-right-corner
            warp_xx = np.fliplr(xx)
            warp_yy = np.fliplr(yy)
        elif (direction==5):    # starting from row 0
            warp_xx = np.fliplr(xx)
            warp_yy = np.fliplr(yy)
        elif (direction==6):    # range(-H+1,W): starting from top-right-corner
            warp_xx = np.flip(xx,axis=(0,1))
            warp_yy = np.flip(yy,axis=(0,1))
        elif (direction==7):    # starting from col 0
            warp_xx = np.fliplr(xx.T)
            warp_yy = np.fliplr(yy.T)
        elif (direction==8):    # range(-H+1,W): starting from top-left-corner
            warp_xx = np.flipud(xx)
            warp_yy = np.flipud(yy)
        else:
            print ("Error in direction, should be 1..8")
        slice_dict ={}
        index_dict ={}
        (warp_H, warp_W) = np.shape(warp_xx)        # coule be yy - same shape 
        if ((direction % 2) == 1):                  # up,down,left,right
            slice_range = range(warp_H)
            for slice in slice_range:
                slice_dict[slice] = ssdd[warp_yy[slice], warp_xx[slice],:]
                index_dict[slice] = np.array([warp_yy[slice], warp_xx[slice]])
        else:                                       # diagonal
            slice_range = range((-warp_H+1),warp_W)
            for slice in slice_range:
                slice_dict[slice] = ssdd[np.diagonal(warp_yy,slice), np.diagonal(warp_xx,slice),:]
                # indices are as a (2,slice_size) array - row 0 is pixel row, row 1 is pixel column
                index_dict[slice] = np.array((np.diagonal(warp_yy,slice), np.diagonal(warp_xx,slice)))
        return (slice_dict, index_dict)


    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        #######################################################
        """INSERT YOUR CODE HERE"""
        for direction in range(1,num_of_directions+1):
            ld = np.zeros_like(ssdd_tensor)
            (slice_dict, index_dict) = self.extract_slices(ssdd_tensor, direction)
            for slice in slice_dict.keys():
                l_slice = self.dp_grade_slice(slice_dict[slice].T, p1, p2)
                #print(slice_dict[slice])
                #print(ssdd[index_dict[slice][0],index_dict[slice][1]])
                ld[index_dict[slice][0],index_dict[slice][1],:] = l_slice.T
            direction_to_slice[direction] = self.naive_labeling(ld) 
        return direction_to_slice


    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        #######################################################
        """INSERT YOUR CODE HERE"""
        for direction in range(1,num_of_directions+1):
            ld = np.zeros_like(ssdd_tensor)
            (slice_dict, index_dict) = self.extract_slices(ssdd_tensor, direction)
            for slice in slice_dict.keys():
                l_slice = self.dp_grade_slice(slice_dict[slice].T, p1, p2)
                #print(slice_dict[slice])
                #print(ssdd[index_dict[slice][0],index_dict[slice][1]])
                ld[index_dict[slice][0],index_dict[slice][1],:] = l_slice.T
            l += ld
        #######################################################
        return self.naive_labeling(l)



"""
def slice_plot_3d(self, ssdd_slice: np.ndarray):
    ax = plt.figure().add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

    ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()
"""