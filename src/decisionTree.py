"""
Author: Samuel Bretz (sbretz3)
Date: 8/31/17
Email: bretzsam@gmail.com
"""

import numpy as np
import random
from sklearn import datasets
from sklearn import tree
import pandas
import math

"""

Decision Tree Implementation and Training

"""


"""
Function to prepare and clean data.
RETURN VALUE: Tuple in the following form: (testData, trainingData)
"""

def prepData():
  data = []
  file = open('../data/arrhythmia.data')
  for line in file:
      instance = []
      for item in line.strip('\n').split(','):
        try:
          item = float(item)
        except:
          item = 0.0
        instance.append(item)
      data.append(instance)
  print data[0]




prepData()