import numpy as np
import matplotlib.pyplot as plt
from tools.utils import *

# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance = []
        for k in range(K):
            d = np.linalg.norm(X[i] - centroids[k])
            distance.append(d)
        idx[i] = np.argmin(distance)

    return idx


# UNQ_C2
# GRADED FUNCTION: compute_centpods

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    for k in range(K):
        point_ls = X[idx == k]
        centroids[k] = np.mean(point_ls, axis=0)
    return centroids


def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx


if __name__ == '__main__':
    """
    压缩图像，将RGB数千种颜色聚类成16种，只需要存储16种特定颜色的值，图像中的每个元素，只存储该位置的颜色索引即可
    """
    # Load an image of a bird
    original_img = plt.imread('data/bird_small.png')
    # Visualizing the image
    plt.imshow(original_img)
    print("Shape of original_img is:", original_img.shape)

    # Process data
    # Divide by 255 so that all values are in the range 0 - 1
    original_img = original_img / 255

    # Reshape the image into an m x 3 matrix where m = number of pixels
    # (in this case m = 128 x 128 = 16384)
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X_img that we will use K-Means on.

    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16
    max_iters = 10

    # Using the function you have implemented above.
    initial_centroids = kMeans_init_centroids(X_img, K)

    # Run K-Means - this takes a couple of minutes
    centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

    # Represent image in terms of indices
    X_recovered = centroids[idx, :]

    # Reshape recovered image into proper dimensions
    X_recovered = np.reshape(X_recovered, original_img.shape)

    # Display original image
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    plt.axis('off')

    ax[0].imshow(original_img * 255)
    ax[0].set_title('Original')
    ax[0].set_axis_off()

    # Display compressed image
    ax[1].imshow(X_recovered * 255)
    ax[1].set_title('Compressed with %d colours' % K)
    ax[1].set_axis_off()