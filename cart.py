'''
Author: Michael Liu
Classification and Regression Tree Development Practice
09/22/2018

this repo uses uci dataset: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009#winequality-red.csv
for regression and classification purposes (CART)
CART - Classification and Regression Trees

online reference: 
siraj ravel - random forest youtube
How to implement the decision tree algorithm from scratch in python
'''
from __future__ import division
from data_utils import WineQualityDataset

'''
Basic procedurals of building CART:
- Gini index
- Create split
- Build a tree based on split
- Make a prediction
'''

#if the regression tree is binary, we will see two groups in the split_data
def gini_index(split_data, labels, feature):
	'''
	gini(t) = 1 - sum_(i=0)^(c-1) [P(i|t) ** 2]
	- t refers to individual groups
	- c refers to classes (label)
	- each group's gini is weighted average
	the more gini_index approach 0, the better the split is
	'''
	total = float(sum([len(group) for group in split_data]))
	gini_total = []
	for group in split_data:
		size = len(group)
		if size == 0:
			continue
		class_proportion = []
		for label in labels:
			#calculate the probability of label existing in the groups
			class_proportion.append([row[-1] for row in group].count(label) / size)
		gini_score = sum([p**2 for p in class_proportion])
		#gini(t) done
		gini_total.append((1.0 - gini_score, size / total))
	#calculate weighted average for total gini index
	weighted_gini_index = sum([w * score for score, w in gini_total])
	return weighted_gini_index

'''
creating a split requires choosing a specific attribute and breaking point
'''
def split_with_attribute_value(dataset, attr_index, value):
	#binary split
	left_node, right_node = [], []
	for row in dataset:
		if row[attr_index] < value:
			left_node.append(row)
		else:
			right_node.append(row)
	return [left_node, right_node]

#dataset assumes that label class is the last columns!
#n_features including label feature
def split(dataset, n_features):
	labels = list(set(row[-1] for row in dataset))
	#gini score should not exceed 1
	min_gini_score = 1.0
	min_split = {'attribute': 0, 'value': 0, 'groups': []}
	#walking through all the attributes
	for index in range(n_features - 1):
		#going through all the rows to find the optimal value split
		for row in dataset:
			split_data = split_with_attribute_value(dataset, index, row[index])
			gini = gini_index(split_data, labels, index)
			if gini < min_gini_score:
				min_gini_score = gini
				min_split = {'attribute': index, 'value': row[index], 'groups': split_data, 'gini': min_gini_score}
	return min_split

def recursive_split(node, max_depth, min_size, depth, n_features):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = leaf(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = leaf(left), leaf(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = leaf(left)
	else:
		node['left'] = split(left, n_features)
		recursive_split(node['left'], max_depth, min_size, depth + 1, n_features)
	# process right child
	if len(right) <= min_size:
		node['right'] = leaf(right)
	else:
		node['right'] = split(right, n_features)
		recursive_split(node['right'], max_depth, min_size, depth + 1, n_features)

'''
no more splitting beyond this point. return a summary of the split data.
'''
def leaf(split_data):
	#finding the majority label inside the group
	labels = [row[-1] for row in split_data]
	return max(set(labels), key=labels.count)

'''
build split hierarchy
'''
def build_tree(dataset, n_features, max_depth, min_size):
	root = split(dataset, n_features)
	recursive_split(root, max_depth, min_size, 1, n_features)
	return root

def print_tree(node, feature_map, depth='-'):
	if isinstance(node, dict):
		print('{}[{} < {}]'.format(depth, feature_map[int(node['attribute'])], node['value']))
		depth += depth
		print_tree(node['left'], feature_map, depth)
		print_tree(node['right'], feature_map, depth)
	else:
		depth = '-' + depth
		print('{}[{}]'.format(depth, node))

if __name__ == "__main__":
	dataset = WineQualityDataset()
	tree = build_tree(dataset.get_data(), len(dataset.features), 4, 1)
	print_tree(tree, dataset.features)
	pass