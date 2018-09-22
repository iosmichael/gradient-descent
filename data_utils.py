import pandas as pd

class WineQualityDataset(object):
	def __init__(self, file_path = "winequality-red.csv"):
		self.file_path = file_path
		try:
			df = pd.read_csv(file_path)
			self.features = df.columns.values
			self.dataset = df.copy()
		except Exception as e:
			print("file download exception raised: {}".format(e))

	def __len__(self):
		if self.dataset is None:
			return 0
		return len(self.dataset)

	def __getitem__(self, index):
		if self.dataset is None:
			print('no dataset found inside the WineQualityDataset instance')
			return None
		return self.dataset.iloc[index, :].tolist()

	def get_data(self, lines = 100):
		return self.dataset.iloc[:lines, :].values.tolist()

	def get_features(self):
		return self.features

	def head(self, num = 5):
		return self.dataset.iloc[:num, :]

	def description(self):
		print("number of features: {}".format(len(self.features)))
		print("features are: {}".format(self.features))
		print("number of data rows: {}".format(len(self.dataset)))
		print("first 5 data rows are: \n {}".format(self.dataset.iloc[:5, :]))

if __name__ == "__main__":
	'''
	testing cases for data file import
	'''
	dataset = WineQualityDataset("winequality-red.csv")
	dataset.description()
	print(dataset[10])