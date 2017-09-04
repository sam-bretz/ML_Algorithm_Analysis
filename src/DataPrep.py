"""
Author: Samuel Bretz (sbretz3)
Date: 9/4/17
Email: bretzsam@gmail.com
"""


class DataPrep():
	"""Class container for cleaning datasets for each algorithm."""
	def __init__(self):
		self.decisionTreeData = open('../data/arrhythmia.data')


	"""
	Function for cleaning Decision Tree data.
	RETURN VALUE: Tuple in the following form: (data, classes)
	"""
	def dTreeData(self):
		data = []
		classCodeSamples = []
		for line in self.decisionTreeData:
			instance = []
			for item in line.strip('\n').split(','):
				try:
					item = str(float(item))
				except:
					item = 0.0
				instance.append(item)
			classCode = instance[-1]
			del instance[-1]
			data.append(instance)
			classCodeSamples.append(classCode)
		return (data, classCodeSamples)



if __name__ == '__main__':
  __package__ = '__DataPrep__'