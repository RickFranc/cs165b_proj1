from math import log

class Tree:
    leaf = True
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None

def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if (point.values[i] < tree.threshold):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)

def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]

def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total

def split_data(data, feature, threshold):
    left = []
    right = []

    for i in range(len(data)):
        point = data[i]
        if(point.values[feature] < threshold):
            left.append(point)
        else:
            right.append(point)
            
    # TODO: split data into left and right by given feature.
    # left should contain points whose values are less than threshold
    # right should contain points with values greater than or equal to threshold
    return (left, right)

def count_labels(data):
    counts = {}

    for i in range(len(data)):
        point = data[i]
        label = point.label
        if label in counts:
            counts[label] = counts[label] + 1
        else:
            counts[label] = 1

    # TODO: counts should count the labels in data
    # e.g. counts = {'spam': 10, 'ham': 4}
    return counts

def counts_to_entropy(counts):
    entropy = 0.0

    elements = 0
    for w,i in counts.items():
        elements += i
    for w,i in counts.items():
        prob = (i*1.0)/elements
        entropy -= prob*log(prob,2);
    # TODO: should convert a dictionary of counts into entropy
    return entropy
    
def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

# This is a correct but inefficient way to find the best threshold to maximize
# information gain.
def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for point in data:
        left, right = split_data(data, feature, point.values[feature])
        curr = (get_entropy(left)*len(left) + get_entropy(right)*len(right))/len(data)
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = point.values[feature]
    return (best_gain, best_threshold)

def find_best_threshold_fast(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    # TODO: Write a more efficient method to find the best threshold.
    data.sort(key = lambda p: p.values[feature])
    for i in range(len(data)):
        curr = (get_entropy(data[0:i])*i + get_entropy(data[i:len(data)])*(len(data)-i))/len(data)
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = data[i].values[feature]
    
    return (best_gain, best_threshold)

def find_best_split(data):
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
     
    for f in range(len(data[0].values)):
        gain, threshold = find_best_threshold_fast(data,f)
        if gain>=best_gain and gain != 0:
            best_feature = f
            best_threshold = threshold
            best_gain = gain
    # TODO: find the feature and threshold that maximize information gain.

    return (best_feature, best_threshold)

def make_leaf(data):
    tree = Tree()   
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label])/len(data)
    tree.prediction = prediction
    return tree

def c45(data, max_levels):
    if max_levels <= 0:
        return make_leaf(data)
    # TODO: Construct a decision tree with the data and return it.
    # Your algorithm should return a leaf if the maximum level depth is reached
    # or if there is no split that gains information, otherwise it should greedily
    # choose an feature and threshold to split on and recurse on both partitions
    # of the data.
    return make_leaf(data)

def submission(train, test):
    # TODO: Once your tests pass, make your submission as good as you can!
    tree = c45(train, 4)
    predictions = []
    for point in test:
        predictions.append(predict(tree, point))
    return predictions

# This might be useful for debugging.
def print_tree(tree):
    if tree.leaf:
        print "Leaf", tree.prediction
    else:
        print "Branch", tree.feature, tree.threshold
        print_tree(tree.left)
        print_tree(tree.right)


