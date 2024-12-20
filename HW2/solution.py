"""Stereo matching."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray, right_image: np.ndarray, win_size: int, dsp_range: int) -> np.ndarray:
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
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows, num_of_cols, len(disparity_values)))
        #######################################################################
        """INSERT YOUR CODE HERE"""
        # Pad the right image to handle disparity shifts and border cases.
        # Height: win_size//2 padding; Width: (dsp_range + win_size//2) padding.
        right_image_padded = np.pad(
            right_image,
            ((win_size // 2, win_size // 2), (dsp_range + win_size // 2, dsp_range + win_size // 2), (0, 0)),
            mode='constant',
        )

        # Create a tensor of right-image slices shifted for each disparity value.
        # Shape: Hx(W+2*win_size//2)x3x(2*dsp_range+1).
        right_image_shifted_tensor = np.stack(
            [right_image_padded[:, i : i + (num_of_cols + (win_size // 2) * 2)] for i in range(len(disparity_values))],
            axis=3,
        )

        # Pad the left image to handle window computations.
        # Height and width: win_size//2 padding.
        left_image_padded = np.pad(
            left_image, ((win_size // 2, win_size // 2), (win_size // 2, win_size // 2), (0, 0)), mode='constant'
        )

        # Compute squared differences between the left image and shifted right slices.
        # Shape: Hx(W+2*win_size//2)x3x(2*dsp_range+1).
        sdd_tensor = (left_image_padded[:, :, :, np.newaxis] - right_image_shifted_tensor) ** 2

        # Compute SSD over local windows using sliding_window_view.
        # Shape: HxWx(2*dsp_range+1).
        ssdd_tensor = np.sum(sliding_window_view(sdd_tensor, (win_size, win_size), (0, 1)), axis=(2, 4, 5))

        # ##### non-vectorized implementation ####
        # for i in range(num_of_rows):
        #     for j in range(num_of_cols):
        #         for d in range(2*dsp_range+1):
        #             ssdd_tensor[i, j, d] = np.sum((left_image_padded[i:i+win_size, j:j+win_size, :] - right_image_padded[i:i+win_size, j+d:j+d+win_size, :]) ** 2)

        # print(f"All pixels are equal: {np.allclose(ssdd_tensor, ssdd_tensor2)}")

        #######################################################################
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
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))

        #######################################################################
        """INSERT YOUR CODE HERE"""

        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        #######################################################################

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

        #######################################################################
        """INSERT YOUR CODE HERE"""

        l_slice_padded = np.pad(l_slice, ((0, num_labels - 1), (0, 0)), mode='constant', constant_values=np.inf)

        for col in range(num_of_cols):
            if col == 0:
                l_slice_padded[0:num_labels, col] = c_slice[:, col]
                continue

            a = l_slice_padded[0:num_labels, col - 1]

            t = np.hstack(
                [np.roll(l_slice_padded[:, col - 1 : col], k)[0:num_labels] for k in range(-num_labels + 1, num_labels)]
            )
            zero_disparity_index = num_labels - 1
            b = t[:, [zero_disparity_index - 1, zero_disparity_index + 1]].min(axis=1) + p1
            c = (
                t[:, list(range(0, zero_disparity_index - 1)) + list(range(zero_disparity_index + 2, len(t)))].min(
                    axis=1
                )
                + p2
            )

            M = np.stack([a, b, c], axis=1).min(axis=1)

            l_slice_padded[0:num_labels, col] = c_slice[:, col] + M - a.min()

        l_slice = l_slice_padded[0:num_labels, :]
        #######################################################################

        return l_slice

    def dp_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float) -> np.ndarray:
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

    def dp_labeling_per_direction(self, ssdd_tensor: np.ndarray, p1: float, p2: float) -> dict:
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
        """INSERT YOUR CODE HERE"""
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
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)


if __name__ == "__main__":
    c_slice = np.random.rand(7, 5)
    p1 = 0.1
    p2 = 0.3
    solution = Solution()
    l_slice = solution.dp_grade_slice(c_slice, p1, p2)
