"""Tools that computes various statistics for the given input.

Following statistics are computed for the given input. The input is assumed to be 1-d array.
  and for multi-dimension array, statistics is computed across the axis=0.
  - Mean
  - Mode
  - Variance
  - Median
  - Kutosis
  - Skewness
  - Statistical Outlier
"""

import os
import sys

from typing import List, Dict, Union

import numpy as np
from scipy.stats import kurtokurtosis, skew, modesis


def compute_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
  """Identify statistical outliers in a NumPy ndarray along the 0 axis.

  An outlier is defined as a data point that lies outside a specified number of
  standard deviation from the mean.

  Args:
    data (np.ndarray): A 2d NumPy array containing the input data. The outliers 
      will be computed along the 0 axis.
    threshold(float): The z-score threshold for identifying the outlier. Defaults to 3.0

  Returns:
    np.ndarray of boolen type indicating if the particular value is an outlier or not.

  Raises:
    ValueError if the input is not a np.ndarray.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 100, 6], [7, 8, 9]])
    >>> compute_outliers(data)
    array([[False, False, False],
           [False,  True, False],
           [False, False, False]])

  Notes:
    - Outliers are determined using the z-score method.
    - The default threshold of 3.0 corresponds to approximately 99.7% of data
      within a normal distribution.
  """
  if not isinstance(data, np.ndarray):
    raise ValueError("Input data should be a array of type NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input should be a 2-dimensional array.")

  if not data:
    return np.array([])

  # Compute mean and std-dev along the 0 axis.
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0, ddof=0)

  # Compute z-scores.
  zscores = (data - mean) / std

  # Identify the outliers based on the threshold.
  return np.abs(zscores) > threshold


def compute_iqr(data: np.ndarray) -> np.ndarray:
  """Compute Interquartile range (IQR) of a NumPy array along the axis 0.

  The IQR is the difference between the 75 and 25 percentiles, and it measures the
  spread of the middle 50% of the data.

  Args: 
    data(np.ndarray): A 2d NumPy array containing the input data. The IQR will be 
      computed along the 0 axis.
  
  Returns:
    np.ndarray: A 1D Numpy array with IQR values for each column of the input array.

  Raises:
    ValueError if the input is not a np.ndarray.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_iqr(data)
    array([3., 3., 3.])

  Notes:
    - The IQR is a robust measure of variability that is not influenced by
      outliers or extreme values.
  """
  if not isinstance(data, np.ndarray):
    raise ValueError("Input data should be of type NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  # Compute 25th and 75th percentiles along the 0 axis
  q25 = np.percentile(data, 25, axis=0, interpolation='linear')
  q75 = np.percentile(data, 75, axis=0, interpolation='linear')

  # Compute IQR
  return q75 - q25
    
def compute_mean(data: np.ndarray) -> np.ndarray:
  """Computes the mean of a NumPy ndarray along the axis 0.
  
  Args:
    data (np.ndarray): A n-dimension numpy array for which the mean should be calculated.

  Returns:
    np.ndarray: The computed mean along the axis 0 in a np.ndarray format.

  Raises:
    ValueError if the input is not a np.ndarray.
  """
  if not isinstance(data, np.ndarray):
    raise ValueError("Input should be a array of type NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  if not data:
    return np.array([])
  else:
    return np.mean(data, axis=0)


def compute_kurtosis(data: np.ndarray) -> np.ndarray:
  """
  Compute the kurtosis of a NumPy ndarray along the 0 axis.

  Args:
    data (np.ndarray): A 2D NumPy array containing the input data. The kurtosis 
                      will be computed along the 0 axis.

  Returns:
    np.ndarray: A 1D NumPy array containing the kurtosis values for each column 
                of the input array.

  Raises:
    ValueError: If the input is not a 2D NumPy array.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_kurtosis(data)
    array([-1.5, -1.5, -1.5])

  Notes:
    - Kurtosis is a measure of the "tailedness" of the probability distribution
      of a real-valued random variable.
    - A positive kurtosis indicates heavy tails, while a negative kurtosis 
      indicates light tails relative to a normal distribution.
  """
  # Validate input
  if not isinstance(data, np.ndarray):
    raise ValueError("Input must be a NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  # Compute kurtosis along the 0 axis
  return kurtosis(data, axis=0, fisher=True, nan_policy='omit')


def compute_skewness(data: np.ndarray) -> np.ndarray:
  """
  Compute the skewness of a NumPy ndarray along the 0 axis.

  Args:
    data (np.ndarray): A 2D NumPy array containing the input data. The skewness 
                      will be computed along the 0 axis.

  Returns:
    np.ndarray: A 1D NumPy array containing the skewness values for each column 
                of the input array.

  Raises:
    ValueError: If the input is not a 2D NumPy array.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_skewness(data)
    array([0., 0., 0.])

  Notes:
    - Skewness is a measure of the asymmetry of the probability distribution
      of a real-valued random variable.
    - A positive skew indicates a longer or fatter tail on the right side,
      while a negative skew indicates a longer or fatter tail on the left side.
  """
  # Validate input
  if not isinstance(data, np.ndarray):
    raise ValueError("Input must be a NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  # Compute skewness along the 0 axis
  return skew(data, axis=0, nan_policy='omit')


def compute_mode(data: np.ndarray) -> np.ndarray:
  """
  Compute the mode of a NumPy ndarray along the 0 axis.

  Args:
    data (np.ndarray): A 2D NumPy array containing the input data. The mode 
                      will be computed along the 0 axis.

  Returns:
    np.ndarray: A 1D NumPy array containing the mode values for each column 
                of the input array.

  Raises:
    ValueError: If the input is not a 2D NumPy array.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_mode(data)
    array([1, 2, 3])

  Notes:
    - The mode is the value that appears most frequently in a dataset.
  """
  # Validate input
  if not isinstance(data, np.ndarray):
    raise ValueError("Input must be a NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  # Compute mode along the 0 axis
  return mode(data, axis=0, nan_policy='omit').mode[0]


def compute_median(data: np.ndarray) -> np.ndarray:
  """
  Compute the median of a NumPy ndarray along the 0 axis.

  Args:
    data (np.ndarray): A 2D NumPy array containing the input data. The median 
                      will be computed along the 0 axis.

  Returns:
    np.ndarray: A 1D NumPy array containing the median values for each column 
                of the input array.

  Raises:
    ValueError: If the input is not a 2D NumPy array.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_median(data)
    array([4., 5., 6.])

  Notes:
    - The median is the middle value of a dataset when sorted in ascending order.
  """
  # Validate input
  if not isinstance(data, np.ndarray):
    raise ValueError("Input must be a NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  # Compute median along the 0 axis
  return np.median(data, axis=0)


def compute_variance(data: np.ndarray) -> np.ndarray:
  """
  Compute the variance of a NumPy ndarray along the 0 axis.

  Args:
    data (np.ndarray): A 2D NumPy array containing the input data. The variance 
                      will be computed along the 0 axis.

  Returns:
    np.ndarray: A 1D NumPy array containing the variance values for each column 
                of the input array.

  Raises:
    ValueError: If the input is not a 2D NumPy array.

  Example:
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_variance(data)
    array([6., 6., 6.])

  Notes:
    - The variance measures the spread of the data points around the mean.
  """
  # Validate input
  if not isinstance(data, np.ndarray):
    raise ValueError("Input must be a NumPy ndarray.")
  if data.ndim != 2:
    raise ValueError("Input array must be 2-dimensional.")

  # Compute variance along the 0 axis
  return np.var(data, axis=0, ddof=0)

