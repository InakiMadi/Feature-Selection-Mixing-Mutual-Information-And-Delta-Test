import info
from numpy import array
import time
import nose

# Define data array
data = array( [ (0, 0, 1, 1, 0, 1, 1, 2, 2, 2),
                (3, 4, 5, 5, 3, 2, 2, 6, 6, 1),
                (7, 2, 1, 3, 2, 8, 9, 1, 2, 0),
                (7, 7, 7, 7, 7, 7, 7, 7, 7, 7),
                (0, 1, 2, 3, 4, 5, 6, 7, 1, 1)])
# Create object. We only want to compare between points of X (single_entropy, entropy, MI), with no Y.
im_object = info.Informacion(X=data,last_feature_is_Y=False)


# --- Checking single random var entropy

# entropy of  X_1 (3, 4, 5, 5, 3, 2, 2, 6, 6, 1)
t_start = time.time()
print('Entropy(X_1): %f' % im_object.single_entropy_X(index=1, log_base=10))
print('Elapsed time: %f\n' % (time.time() - t_start))

# entropy of  X_3 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
t_start = time.time()
print('Entropy(X_3): %f' % im_object.single_entropy_X(3, 10))
print('Elapsed time: %f\n' % (time.time() - t_start))

# entropy of  X_4 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
t_start = time.time()
print('Entropy(X_4): %f' % im_object.single_entropy_X(4, 10))
print('Elapsed time: %f\n' % (time.time() - t_start))



# --- Checking entropy between two random variables

# entropy of  X_0 (0, 0, 1, 1, 0, 1, 1, 2, 2, 2) and X_1 (3, 4, 5, 5, 3, 2, 2, 6, 6, 1)
t_start = time.time()
print('Entropy(X_0, X_1): %f' % im_object.entropy_XX(0, 1, 10))
print('Elapsed time: %f\n' % (time.time() - t_start))

# entropy of  X_3 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7) and X_3 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
t_start = time.time()
print('Entropy(X_3, X_3): %f' % im_object.entropy_XX(3, 3, 10))
print('Elapsed time: %f\n' % (time.time() - t_start))


# ---Checking Mutual Information between two random variables

# Print mutual information between X_0 (0,0,1,1,0,1,1,2,2,2) and X_1 (3,4,5,5,3,2,2,6,6,1)
t_start = time.time()
print('MI(X_0, X_1): %f' % im_object.mutual_information_XX(0, 1, 10))
print('Elapsed time: %f\n' % (time.time() - t_start))

# Print mutual information between X_1 (3,4,5,5,3,2,2,6,6,1) and X_2 (7,2,1,3,2,8,9,1,2,0)
t_start = time.time()
print('MI(X_1, X_2): %f' % im_object.mutual_information_XX(1, 2, 10))
print('Elapsed time: %f\n' % (time.time() - t_start))



# --- Checking results

# Checking entropy results
for i in range(0,data.shape[0]):
    assert(im_object.entropy_XX(i, i, 10) == im_object.single_entropy_X(i, 10))

# Checking mutual information results
# MI(X,Y) = H(X) + H(Y) - H(X,Y)
n_rows = data.shape[0]
i = 0
while i < n_rows:
    j = i + 1
    while j < n_rows:
        if j != i:
            nose.tools.assert_almost_equal(im_object.mutual_information_XX(i, j, 10),
                        im_object.single_entropy_X(i, 10)+im_object.single_entropy_X(j, 10)-im_object.entropy_XX(i, j, 10))
        j += 1
    i += 1
