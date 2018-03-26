import os
import functools

from loader import load_data_csv
from plotting import plot_points, plot_svm_decision_boundary
from classification import SVM
from kernels import linear, rbf
from metrics import accuracy

__author__ = 'Otilia Stretcu'


# Parameters.
data_folder = '../data/'
dataset_name = 'cluster_harder'                      # Select either 'cluster_easy', 'cluster_harder', 'camel'.
C = 1
# kernel_func = functools.partial(rbf, gamma=100.0)  # Use this for camel data.
kernel_func = linear                                 # Use this for clusters data.
output_path = '../outputs/'                          # Where to save the plots.
class_colors = {-1: 'b', 1: 'r'}                     # Colors for plotting.

# Load data.
x_train, y_train = load_data_csv(os.path.join(data_folder, dataset_name+'_train.csv'))
x_test, y_test = load_data_csv(os.path.join(data_folder, dataset_name+'_test.csv'))

plot_points(x_train, y_train, class_colors=class_colors, title='Train - correct labels')
plot_points(x_test, y_test, class_colors=class_colors, title='Test - correct labels')

# Train the SVM classifier on the training data.
svm = SVM(kernel_func=kernel_func, C=C)
print('Training...')
svm.train(x_train, y_train)

# Plot the decision boundary.
print('Plotting...')
plot_svm_decision_boundary(svm, x_train, y_train,
    title='SVM decision boundary on training data', output_path=output_path,
    file_name=str(dataset_name) + '_support_vectors_train.png',
    class_colors=class_colors)

# Make predictions on train and test data.
y_train_pred = svm.predict(x_train)
y_test_pred = svm.predict(x_test)

plot_points(x_train, y_train_pred, class_colors=class_colors,
    title='Your predictions for training data')
plot_points(x_test, y_test_pred, class_colors=class_colors,
    title='Your predictions for test data')

# Compute the classification accuracy for train and test.
acc_train = accuracy(y_train_pred, y_train)
acc_test = accuracy(y_test_pred, y_test)

print('Train accuracy: %.2f.' % acc_train)
print('Test accuracy: %.2f.' % acc_test)

print('Done.')