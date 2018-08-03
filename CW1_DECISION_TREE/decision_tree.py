import scipy.io
import numpy as np
import math
import random
# from graphviz import Digraph
import pickle, pprint

INDEX_LAST_ELEMENT = -1


def chooseEmotion(example, emotion):
    """Example contains an array of integers. A binary array is returned with a 1 where the corresponding element is equal to the value of emotion and 0 otherwise."""
    result_array =  []
    for element in example:
        if element == emotion:
            result_array.append(1)
        else:
            result_array.append(0)
    return np.asarray(result_array)

def count(column, target):
    """Returns number of items in column equal to target"""
    count = 0
    if column.size == 0:
        return 0
    for element in range(column.size):
        if column[element] == target:
            count += 1
    return count

def merge(all_data, example):
    """Combines two arrays all_data and example into one matrix"""
    i = 0
    complete_array = []
    for row in all_data:
        row = np.append(row,example[i])
        i += 1
        complete_array.append(row)
    return np.asarray(complete_array)

def entropy(binary_targets):
    """Returns the entropy of binary_targets"""
    num_one = count(binary_targets,1)
    num_zero = count(binary_targets,0)
    summation = num_one + num_zero
    left_side = num_one/summation
    right_side = num_zero/summation

    if left_side == 0:
        return -((right_side)*math.log(right_side,2))
    elif right_side ==0 :
        return (-1*(left_side)*math.log(left_side,2))
    else:
        return (-1*(left_side)*math.log(left_side,2))-((right_side)*math.log(right_side,2))

def gain(data_merge,col_attribute,binary_targets):
    """Returns the information gain of data_merge for col_attribute"""
    return entropy(binary_targets)-remainder(data_merge,col_attribute,binary_targets)

def remainder(data_merge,col_attribute,binary_targets):
    """Returns the remainder of data_merge for col_attribute"""
    left_array = []
    right_array = []
    binary_left = []
    binary_right = []

    for index, row in enumerate(data_merge):
        if row[col_attribute] == 1:
            left_array.append(row)
            binary_left.append(binary_targets[index])
        else:
            right_array.append(row)
            binary_right.append(binary_targets[index])

    left_array = np.asarray(left_array)
    right_array = np.asarray(right_array)
    binary_left = np.asarray(binary_left)
    binary_right = np.asarray(binary_right)
    if left_array.size == 0:
        left_entropy = 0
        p0_left = 0
        n0_left = 0
    else:
        left_entropy = entropy(binary_left)

    p0_n0 = left_array.size

    if right_array.size == 0:
        right_entropy = 0
        p1_right = 0
        n1_right = 0
    else:
        right_entropy = entropy(binary_right)

    p1_n1 = right_array.size

    number_parent = p0_n0 + p1_n1

    return  (p0_n0/number_parent)*left_entropy + (p1_n1/number_parent)*right_entropy

def chooseBestAttr(data_merge,attr_header,binary_targets):
    """Returns the best attribute from data_merge and removes it from the list attr_header"""
    col_best_attr = 0
    max_gain = 0

    if data_merge.size == 0:
        return

    for col_attr in range(data_merge.shape[1]):
        if col_attr not in attr_header:
            continue
        attr_gain = gain(data_merge,col_attr,binary_targets)
        #print('col attr1: ',col_attr, " : values1 = ",attr_gain)

        if attr_gain > max_gain:
            max_gain = attr_gain
            col_best_attr = col_attr

    return col_best_attr


def majorityValue(binary_targets):
    """Returns the majority value of binary_targets"""
    p = count(binary_targets, 1)
    n = count(binary_targets, 0)

    if p > n:
        return 1
    else:
        return 0


def hasSameValueEmotion(data_merge):
    """Returns True if all emotion examples have same emotion value"""
    first_item = data_merge[0]
    for row in data_merge:
        if row != first_item:
            return False
    return True


def getDataSample(data_merge, col_best_attr, binary_targets, val):
    """ Splitting Method that
    Returns two arrays that match best_attr value to val(0 or 1):
    1.arrays of example
    2.array of binary_targets"""
    array_row = []
    binary_row = []
    for index, row in enumerate(data_merge):
        if row[col_best_attr] == val:
            array_row.append(row)
            binary_row.append(binary_targets[index])
    return np.asarray(array_row), np.asarray(binary_row)


def isSameSample(data_merge):
    """Returns true if sample has same value"""
    temp_data_merge = np.copy(data_merge[0])
    i = 0
    for row in data_merge:
        if i == 0:
            i += 1
            continue
        if not np.array_equal(temp_data_merge,row):
            return False
    return True

def decisionTree(examples, attributes, binary_targets):
    """Returns a decision tree for examples, with a list of attributes and binary_targets
    examples is the training data, a matrix of 1s and 0s
    attributes is a list of the attribute headers for each column of the examples matrix
    binary_targets is an array of 1s and 0s, the correct classification for each example."""
    x = tree()
    if isSameSample(examples):
        x.addLeaf(majorityValue(binary_targets))
        return x
    if hasSameValueEmotion(binary_targets):
        x.addLeaf(binary_targets[0])
        return x
    elif len(attributes) == 0:
        value = (majorityValue(binary_targets))
        x.addLeaf(value)
        return x
    else:
        best_attribute = chooseBestAttr(examples, attributes, binary_targets)
        x.addRoot(best_attribute)
        for i in range(2):

            subset_examples, subset_binary = getDataSample(examples, best_attribute, binary_targets, i)
            #print(subset_examples)

            if len(subset_examples) == 0:
                y = tree()
                y.addLeaf(majorityValue(subset_binary))
                x.addKids(y)
            else:
                subset_attribute = attributes[:]
                subset_attribute.remove(best_attribute)
                x.addKids(decisionTree(subset_examples, subset_attribute, subset_binary))
    return x

class tree:
    """class for the tree data structure.
    op = attribute number for this node the root of the (sub)tree
    kids = a list containing the two children of the node, i.e. the root nodes of subtrees. List is empty if this is a leaf node.
    leaf = if the node is a leaf, contains 1 or 0. Otherwise this attribute is empty"""
    def __init__(self):
        self.op = None #attribute number for root
        self.kids = [] #subtreees
        self.leaf = None #value is 1 or 0 if leaf node, otherwise None

    def addRoot(self, attribute):
        """add root node to tree, empty for leaf node"""
        self.op = attribute
        num = str(attribute)
        self.name = num

    def addKids(self, kid):
        """add subtrees to the tree"""
        self.kids.append(kid)

    def addLeaf(self, value):
        """add leaf to the tree"""
        self.leaf = value
        num = str(value)
        self.name = num

    def printtree(self, dot, name):
        """print the tree as plain text"""

        if(self.op is not None):
            dot.node(name, self.name)
            for i in range(len(self.kids)):
                if((self.kids[i].op) is not None):
                    num = str(i)
                    conc = name + self.kids[i].name + num
                    dot.node(conc, self.kids[i].name)
                    if(i == 0):
                        dot.edge(name, conc, constraint='true', label="N")
                    else:
                        dot.edge(name, conc, constraint='true', label="Y")
                    self.kids[i].printtree(dot, conc)
                else:
                    num = str(i)
                    conc = name + self.kids[i].name + num
                    if self.kids[i].leaf == 0:
                        dot.node(conc, "No", shape="box", color="red")
                        if(i == 0):
                            dot.edge(name, conc, constraint='true', label="N")
                        else:
                            dot.edge(name, conc, constraint='true', label="Y")
                    else:
                        dot.node(conc, "Yes", shape="box", color="green")
                        if(i == 0):
                            dot.edge(name, conc, constraint='true', label="N")
                        else:
                            dot.edge(name, conc, constraint='true', label="Y")
        else:
            return
def getResult( attributes, tree, recursion):
    """ Return depth of trees and value in leaf of tree (1 or 0)"""
    if (tree.op == None):
        return tree.leaf, recursion

    if (attributes[tree.op] == 0):
        recursion += 1
        return(getResult(attributes, tree.kids[0],recursion))
    else:
        recursion += 1
        return(getResult(attributes, tree.kids[1],recursion))

def testTrees(T, x2):
    """Tests the list of trees T with data x2.
    Each data item is classified as belonging to an element of T
    an array of classifications as integers is returned.
    Ambiguous data points are classified according to the tree with the shortest branch.
    Code can be altered to use random classification to solve ambiguity."""
    predictions = np.zeros((len(x2), 6))
    predicted = []
    depth_tree = np.ones((len(x2),6))
    for i in range(len(x2)):
        for j in range(len(T)):
            x ,depth_tree[i][j] = getResult(x2[i],T[j],0)
            predicted.append(x)
        predictions[i] = list(predicted)
        predicted.clear()
    predictions = np.asarray(predictions)
    depth_tree = np.asarray(depth_tree)

    # COMMENT OUT ONE OF THESE METHODS AND LEAVE THE OTHER UNCOMMENTED
    predictions = useDepthMethod(predictions,depth_tree) # prefer shallow branches
    #predictions = useRandomMethod(predictions) #random classification

    matrix_shape = predictions.shape
    no_of_rows = matrix_shape[0]
    return_array = np.zeros((no_of_rows,1))

    for index,row in enumerate(predictions):
        for i in range(len(row)):
            if row[i] == 1:
                return_array[index] = i + 1
    return return_array

def useDepthMethod(predictions, depth_tree):
    """makes a predicition based on shallow branches"""
    for index,row in enumerate(predictions):
        if row.sum() == 0:
            max_num = 0
            for j in range(len(row)):
                if row[j] == 0 and depth_tree[index][j] >= max_num:
                    max_num = depth_tree[index][j]
            count_dup = 0
            temp_index_array = []
            for index_depth_tree in range(len(depth_tree[index])):
                if depth_tree[index][index_depth_tree] == max_num:
                     count_dup += 1
                     temp_index_array.append(index_depth_tree)
            if len(temp_index_array) > 1:
                random_choose = random.choice(temp_index_array)
                row[random_choose] = 1
            else:
                row[temp_index_array[0]] = 1

        elif row.sum() > 1:
            min_num = 100
            for j in range(len(row)):
                if row[j] == 1 and depth_tree[index][j] <= min_num:
                    min_num = depth_tree[index][j]
                row[j] = 0

            count_dup2 = 0
            temp_index_array2 = []
            for index_depth_tree in range(len(depth_tree[index])):
                if depth_tree[index][index_depth_tree] == min_num:
                     count_dup2 += 1
                     temp_index_array2.append(index_depth_tree)
            if len(temp_index_array2) > 1:
                random_choose = random.choice(temp_index_array2)
                row[random_choose] = 1
            else:
                row[temp_index_array2[0]] = 1

    return predictions

def useRandomMethod(predictions):
    """makes a prediction with a random choice"""
    for index,row in enumerate(predictions):

        random_choose = 0
        index_tracker = []

        if row.sum() == 0:
            random_choose = random.randint(0,5)
            row[int(random_choose)] = 1
        elif row.sum() > 1:

            for index2 in range(len(row)):
                if row[index2] == 1:
                    index_tracker.append(index2)
                row[index2] = 0
            random_choose = random.choice(index_tracker)
            row[random_choose] = 1

    return predictions


def randomClassify(predictions, classes):
    """randomly chooses one column containing a 1 from each row and returns a column vector of the indices of the column that was chosen +1"""
    return_vector = np.zeros( (len(predictions), 1), dtype=np.int16)
    for i in range(len(predictions)):
        list = []
        for j in range(len(predictions[i])):
            if predictions[i][j] == 1:
                list.append(j+1)
        if len(list) == 0:
            return_vector[i] = random.randint(1,classes)
        elif len(list) == 1:
            return_vector[i] = list[0]
        else:
            return_vector[i] = random.choice(list)
    return return_vector


def split10Fold(data, time):
    """splits data 10 fold"""
    one_fold_data = []
    nine_folds_data = []

    array = np.array(data)
    num_of_data = array.shape[0]
    one_fold = num_of_data // 10
    nine_fold = num_of_data - one_fold
    #print(one_fold)
    #print(nine_fold)
    start = (time - 1) * one_fold
    end = start + one_fold
    #print(start)
    #print(end)
    for i in range(0, num_of_data):
        if(i < end and i >= start):
            one_fold_data.append(data[i])
        else:
            nine_folds_data.append(data[i])
    return np.asarray(one_fold_data), np.asanyarray(nine_folds_data)


def confusionMatrix(T, x2, binary_targets, no_of_classes):
    """Generates and outputs a confusion matrix for the lists of trees T, data x2, correct classifications, binary_targets and the number of classes no_of_classes."""

    confusion_matrix = np.zeros((no_of_classes,no_of_classes))

    # prediction_mat = testTrees(T, x2)
    # prediction_array = matrix2array(prediction_mat)
    prediction_array = testTrees(T,x2)

    for i in range(no_of_classes):
        for j in range(no_of_classes):
            for k in range(len(binary_targets)):
                if binary_targets[k] == j+1 and prediction_array[k] == i+1:
                    confusion_matrix[i][j] += 1

    return confusion_matrix


def averageRecall(confusion_matrix, class_number):
    """returns average recall for the class"""

    total_actual = 0
    for row in confusion_matrix:
        total_actual += row[class_number - 1]

    true_positives = confusion_matrix[class_number - 1][class_number - 1]

    return float(true_positives)/total_actual


def precisionRate(confusion_matrix, class_number):
    """returns precision rate for the class"""
    # precision = True Positive/ (True Positive + False Positive)
    # '' in-row sum '''
    total_predicted = 0
    for i in confusion_matrix[class_number - 1]:
        total_predicted += i

    true_positives = confusion_matrix[class_number - 1][class_number - 1]

    return float(true_positives)/total_predicted

def f1(precision, recall):
    """calculates and returns the f1 measure using the precision and recall"""
    if precision == 0 and recall == 0:
        return 0
    return (2 * float((precision * recall))/(precision + recall))


def classificationRate(confusion_matrix, no_of_classes):
    """calculates and return the classification rate for one class."""
    total = 0
    for row in confusion_matrix:
        for cell in row:
            total += cell
    total_true = 0
    i = 0
    while i < no_of_classes:
        total_true += confusion_matrix[i][i]
        i += 1
    return float(total_true) / total

def trainTrees(number_of_trees, attribute_values, classifications, split_value):
    """returns a list of length number_of_trees trained with attribute_values and classifications split 10-fold at location split_value"""
    trees = []
    for i in range(1, number_of_trees + 1):
        example = chooseEmotion(classifications, i)
        attr_header = []
        data_merge = merge(attribute_values, example)
        (test_data, training_data) = split10Fold(attribute_values, split_value)
        (binary_test, binary_training) = split10Fold(example, split_value)
        for j in range(len(data_merge[0])):
            attr_header.append(j)
        x = decisionTree(training_data, attr_header, binary_training)
        trees.append(x)
    return trees

def crossValidationResults(data_x, data_y, no_of_classes, times):
    """prints the average classification, a confusion matrix and recall, precision and f1 measure for each class of the trees using 10 fold cross validation the specified number of times"""
    av_conf = np.zeros((no_of_classes,no_of_classes))
    av_class = np.zeros((times,1))
    for j in range(times):
        conf = np.zeros((no_of_classes,no_of_classes))
        for i in range(1,times+1):
            decision = trainTrees(no_of_classes, data_x, data_y, i)
            (test_data, training_data) = split10Fold(data_x, i)
            (binary_test, binary_training) = split10Fold(data_y, i)
            x = confusionMatrix(decision, test_data, binary_test, no_of_classes)
            conf+=x
            total =0.0
            av_class[i-1] += (classificationRate(x, no_of_classes))

        conf = conf/no_of_classes
        av_conf += conf

    av_class = av_class/times
    average_classification = sum(av_class)/len(av_class)

    print("average classification:", average_classification)

    av_conf = av_conf/times
    print("Confusion matrix:")
    print(av_conf)

    recalls = []
    precisions = []
    f1s = []
    for i in range(1,no_of_classes+1):
        recall = averageRecall(av_conf, i)
        recalls.append(recall)
        print("recall for class", i, recall)
        prec = precisionRate(av_conf, i)
        precisions.append(prec)
        print("precision rate for class", i, prec)
        f = f1(prec, recall)
        f1s.append(f)
        print("f1 for class", i, f)

    print("average recall", sum(recalls)/len(recalls))
    print("average precision", sum(precisions)/len(precisions))
    print("average f1", sum(f1s)/len(f1s))



############################# LOADING DATA ####################################
data = scipy.io.loadmat("Data/noisydata_students.mat")

array_data = np.array(data)
data_x = np.array(data['x'])
data_y = np.array(data['y'])

attribute_tracker = []
for row in range(data_x.shape[1]):
    attribute_tracker.append(row)
#

######################### TRAINING TREES ######################################
# binary = []
# for i in range(6):
#     example = chooseEmotion(data_y, i+1)
#     binary.append(example)
#
# array_tree = []
# for i in range(6):
#     array_tree.append(decisionTree(data_x, attribute_tracker, binary[i]))
# print(array_tree)
# # Output to Pickle File
# output = open('decision_tree.pkl','wb')
# pickle.dump(array_tree,output,-1)
# output.close()


################## VALIDATION RESULTS ########################################
# example_1 = chooseEmotion(data_y,1)
# data_merge1 = merge(data_x, example_1)
# attr_header = []
# for i in range(len(data_merge1[0])):
#     attr_header.append(i)
# #
# for i in range(1,11):
#     decision = trainTrees(6, data_x, data_y, i)
#     (test_data, training_data) = split10Fold(data_x, i)
#     (binary_test, binary_training) = split10Fold(data_y, i)
#     x = confusionMatrix(decision, test_data, binary_test, 6)
#     total =0.0
# print(classificationRate(x, 6))

#################### PRINTING TREES ###########################################
#graphviz required

#dot = Digraph(comment = "Decision Tree 1")
# tree.printtree(dot, tree.name) #x is the tree to be printed
# dot.render('pic/round-table.gv', view=True) #saves the image in a pdf file


####### RUNNING TESTTREES FUNCTION ###########################################
# Input, Read and Predictor Labels

pk = open('decision_tree.pkl','rb')
trees = pickle.load(pk)
predictiveLabels = testTrees(trees, data_x)
print(predictiveLabels) #print resulsts

pk.close()
