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

    return counts

def counts_to_entropy(counts):
    entropy = 0.0

    elements = 0
    for w,i in counts.items():
        elements += i
        
    for w,i in counts.items():
        prob = (i*1.0)/elements
        if(prob != 0):
            entropy -= prob*log(prob,2);
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
    data.sort(key = lambda p: p.values[feature])
    counts_left = count_labels(data[0:1])
    counts_right = count_labels(data[1:len(data)])
    for i in range(1,len(data)-1):
        if(data[i].values[feature] > data[i-1].values[feature]): #can only split if different threshold
            curr = (counts_to_entropy(counts_left)*i + counts_to_entropy(counts_right)*(len(data)-i))/len(data)
            gain = entropy - curr
            if gain > best_gain:
                best_gain = gain
                best_threshold = data[i].values[feature]

        label = data[i].label
        if(label in counts_left):
            counts_left[label] += 1
        else:
            counts_left[label] = 1
        counts_right[label] -= 1
        
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
    best_feature, best_threshold = find_best_split(data)
    if best_feature == None:
        return make_leaf(data)

    left,right = split_data(data, best_feature, best_threshold)
    left_tree = c45(left, max_levels-1)
    right_tree = c45(right, max_levels-1)
    tree = Tree()
    tree.leaf = False
    tree.left = left_tree
    tree.right = right_tree
    
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label])/len(data)
    tree.prediction = prediction

    tree.feature = best_feature
    tree.threshold = best_threshold
    return tree

#In order to increase accuracy, I increased the maximum number of levels within the decision tree
#This affects the accuracy because additional boundaries can be added to decisions
#Each split allow additional information to be gained for the real data and that will allow
# for more precise decisions
def submission(train, test):
    tree = c45(train, 9)
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


