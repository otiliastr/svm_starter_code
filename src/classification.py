import numpy as np
from kernels import linear
from scipy.optimize import minimize

__author__ = 'Otilia Stretcu'


class SVM:
    def __init__(self, kernel_func=linear, C=1, tol=1e-3):
        """
        Initialize the SVM classifier.

        :param kernel_func(function): Kernel function, that takes two arguments,
            x_i and x_j, and returns k(x_i, x_j), for some kernel function k.
            If no kernel_function is provided, it uses by default linear.
        :param C(float): Slack tradeoff parameter in the dual function.
        :param tol(float): Tolerance used by the optimizer.
        """
        self.C = C
        self.kernel_func = kernel_func
        self.tol = tol

        # Initialize the information about the support vectors to None, and it
        #  will be updated after training.
        self.support_multipliers = None
        self.bias = None
        self.support_vectors = None
        self.support_vector_labels = None

    def train(self, inputs, targets):
        """
        Use the inputs and targets to learn the SVM parameters.
        :param inputs(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :param targets(np.ndarray): Targets, of shape (num_samples,),
            having values either -1 or 1.
        :return:
        """

        # We train the SVM classifier by solving the dual problem.
        # Calculate the Lagrange multipliers, alphas.
        alphas = self.solve_dual(inputs, targets)

        # Use the Lagrange multipliers to find the support vectors.
        support_vector_indices = self.find_support_vectors(inputs, targets, alphas)

        # Keep only the alpha's, x's and y's that correspond to the support
        # vectors found above.
        self.support_multipliers = alphas[support_vector_indices]
        self.support_vectors = inputs[support_vector_indices, :]
        self.support_vector_labels = targets[support_vector_indices]

        # Calculate the bias.
        self.bias = self.compute_bias(inputs, targets, alphas,
            support_vector_indices, self.kernel_func)

    def compute_kernel_matrix(self, x):
        """
        Uses the kernel function to compute the kernel matrix K for the input
        matrix x, where K(i, j) = kernel_func(x_i, x_j).
        :param x(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :return:
            K(np.ndarray): Kernel matrix, of shape (num_samples, num_samples)
        """
        # TODO: implement this.
        # Tip: Try to use vector operations as much as possible for
        # computation efficiency.
        K = None
        return K

    def solve_dual(self, x, y):
        """
        Computes the Lagrange multipliers for the dual problem.
        :param x(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :param y(np.ndarray): Targets, of shape (num_samples,),
            having values either -1 or 1.
        :return:
             alphas(np.ndarray): Lagrange multipliers, of shape (num_samples,)
        """
        num_samples, num_features = x.shape

        # Use the kernel function to compute the kernel matrix.
        K = self.compute_kernel_matrix(x)

        # Solve the dual problem:
        #    max sum_i alpha_i - 1/2 sum_{i,j} alpha_i * alpha_j * y_i * y_j * k(x_i, x_j)
        #    s.t.
        #       sum_i alpha_i * y_i = 0
        #       C >= alpha_i >= 0
        #       k(x_i, x_j) = phi(x_i) * phi(x_j)
        # by converting it into a quadratic program form accepted by the scipy
        # SLSQP optimizer.
        # See documentation at:
        # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        # Tip: Try to use vector operations as much as possible for
        # computation efficiency.

        # Define the objective function and the gradient wrt. alphas.
        def objective(alphas):
            # TODO: implement this.
            obj = None
            gradient = None
            return (obj, gradient)

        # Define any necessary inequality and equality constraints.
        # TODO: implement this.
        constraints = (
            {'type': add code here,
             'fun': add code here,
             'jac': add code here})

        # Define the bounds for each alpha.
        # TODO: implement this.
        bounds = None

        # Define the initial value for alphas.
        alphas_init = np.zeros((num_samples,))

        # Solve the QP.
        result = minimize(objective, alphas_init, method="SLSQP", jac=True,
            bounds=bounds, constraints=constraints, tol=self.tol,
            options={'ftol': self.tol, 'disp': 2})
        alphas = result['x']

        return alphas

    def find_support_vectors(self, x, y, alphas, tol=1e-5):
        """
        Uses the Lagrange multipliers learnt by the dual problem to determine
        the support vectors that will be used in making predictions.
        :param x(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :param y(np.ndarray): Targets, of shape (num_samples,), having values
            either -1 or 1.
        :param alphas(np.array): Lagrange multipliers, of shape (num_samples,)
        :param tol(float): Tolerance when comparing  values.
        :return:
            support_vector_indices(np.array): Indices of the samples that will
                be the support vectors. This is an array of length
                (num_support_vectors,)
        """
        # Find which of the x's are the support vectors. If you want to compare
        # your values with a threshold, for numerical stability make sure to
        # allow for some tolerance (e.g. tol = 1e-5). Use the parameter tol for
        # setting the tolerance. We will run your code with the same tolerance
        # we use in our implementation.

        # TODO: implement this.
        support_vector_indices = None
        return support_vector_indices

    def compute_bias(self, x, y, alphas, support_vector_indices, kernel_func):
        """
        Uses the support vectors to compute the bias.
        :param x(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :param y(np.ndarray): Targets, of shape (num_samples,),
            having values either -1 or 1.
        :param alphas(np.array): Lagrange multipliers, of shape (num_samples,)
        :param support_vector_indices(np.ndarray): Indices of the support
            vectors in the x and y arrays.
        :return:
            bias(float)
        """
        # Compute the bias that will be used in all predictions. Remember that
        # at test time, we classify each test point x as
        # sign(weights*phi(x) + bias), where the bias does not depend on the
        # test point. Therefore, we can precompute it here.
        # A reference of how to correctly compute the bias in the presence of
        # slack variables can be found at pages 7-8 from
        # http://fouryears.eu/wp-content/uploads/svm_solutions.pdf
        
        # TODO: implement this.
        bias = None
        return bias

    def predict(self, inputs):
        """
        Predict using the trained SVM classifier.
        :param inputs(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :return:
            predictions(np.ndarray): Predictions, of shape (num_samples,),
                having values either -1 or 1.
        """
        # DO NOT CHANGE THIS FUNCTION. Please put your prediction code in
        # self._predict below.
        assert self.support_multipliers is not None, \
            "The classifier needs to be trained before calling predict!"
        return self._predict(inputs, self.support_multipliers,
            self.support_vectors, self.support_vector_labels, self.bias,
            self.kernel_func)

    def _predict(self, inputs, support_multipliers, support_vectors,
                 support_vector_labels, bias, kernel_func):
        # Predict the class of each sample, one by one, and fill in the result
        # in the array predictions.
        num_samples = inputs.shape[0]
        predictions = np.zeros((num_samples,))
        
        # TODO: implement this.

        return predictions

    def decision_function(self, x):
        """
            Calculate f(x) = w.x+b for the given x's.
        :param xs(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :return:
            f(np.ndarray): Array of shape (num_samples,).
        """
        assert self.support_multipliers is not None, \
            "The classifier needs to be trained before applying the decision" \
            "function to new points!"
        return self._decision_function(x, self.support_multipliers,
            self.support_vectors, self.support_vector_labels, self.bias,
            self.kernel_func)

    def _decision_function(self, x, support_multipliers, support_vectors,
                           support_vector_labels, bias, kernel_func):
        """
            Calculate f(x) = w.x+b for the given x's.
        :param xs(np.ndarray): Inputs, of shape (num_samples, num_dims)
        :return:
            f(np.ndarray): Array of shape (num_samples,).
        """
        # TODO: implement this.
        f = None

        return f



