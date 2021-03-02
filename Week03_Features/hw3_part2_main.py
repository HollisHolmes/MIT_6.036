import pdb
import numpy as np
import code_for_hw3_part2 as hw3



#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features_raw = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features_better = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------
#evaluate all paramater choices
# for features in [features_raw, features_better]:
#     for T in [1, 10, 50]:
#         for learner in [hw3.perceptron, hw3.averaged_perceptron]:
#             auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
#             # print('auto data and labels shape', auto_data.shape, auto_labels.shape)
#             eval = hw3.xval_learning_alg(learner, auto_data, auto_labels, 10, T)
#             print(f'*****RESULT:  With T = {T} and using the learner: {learner} and the features {features}, \n Score is: {eval}')

#get theta from best paramater choices
# features = features_better
# data, labels = hw3.auto_data_and_labels(auto_data_all, features)
# print(hw3.averaged_perceptron(data, labels, params = {'T': 1}, hook = None))

#-------------------------------------------------------------------------------
# Review Data
# #-------------------------------------------------------------------------------
#
# # Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# # The train data has 10,000 examples
# review_data = hw3.load_review_data('reviews.tsv')
#
# # Lists texts of reviews and list of labels (1 or -1)
# review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))
#
# # The dictionary of all the words for "bag of words"
# dictionary = hw3.bag_of_words(review_texts)
#
# # The standard data arrays for the bag of whw3.auto_data_and_labels(auto_data_all, features)ords
# review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
# review_labels = hw3.rv(review_label_list)
# d, n = review_bow_data.shape
# print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)
#
# # for T in [1, 10, 50]:
# #     for learner in [hw3.perceptron, hw3.averaged_perceptron]:
# #         eval = hw3.xval_learning_alg(learner, review_bow_data, review_labels, 10, T)
# #         print(f'*****RESULT:  With T = {T} and using the learner: {learner} \n Score is: {eval}')
np.set_printoptions(threshold=np.inf)
# #get the classifier using averaged_perceptron
# th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params = {'T': 10}, hook = None)
# # print(th)
# #get the indices of the most positive or most negative influencing words. These will have the highest/lowest values in theta
# sorted_indices = np.argsort(th, axis = 0)[-10:, 0]
# print(sorted_indices)
# i = 0
# words = []
# #get the words with the indices found above
# for key in dictionary:
#     if i in sorted_indices:
#         words.append(key)
#     i += 1
# print(words)
# max_dist = 0
# max_col = None
# for col in range(n):
#     review = review_bow_data[:, col:col+1]
#     dist = hw3.signed_dist(review, th, th0)
#     if dist>max_dist and len(review_texts[col])<400:
#         max_dist = dist
#         max_col = col
# print('highest review is:', review_texts[max_col])
# print('last review is:', review_texts[-1])




#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples,m,n = x.shape
    for image_num in range(n_samples):
        image = x[image_num, :, :]
        image = np.reshape(image, (n*m, 1))
        # print('*******************************************************************************************')
        # print('image: ', image)
        try:
            flattened_images = np.concatenate((flattened_images, image), axis = 1)
        except:
            flattened_images = np.copy(image)
        # print('flattened image', flattened_images)
    return flattened_images

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    m, n = x.shape
    row_avgs = np.zeros((m, 1))
    for row in range(m):
        row_sum = 0
        for col in range(n):
            row_sum += x[row, col]
        row_avgs[row, 0] =  row_sum/n
    return row_avgs


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    x = x.T
    m, n = x.shape
    row_avgs = np.zeros((m, 1))
    for row in range(m):
        row_sum = 0
        for col in range(n):
            row_sum += x[row, col]
        row_avgs[row, 0] =  row_sum/n
    return row_avgs

def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m, n = x.shape
    mid_row = m//2
    top_sum = 0
    bot_sum = 0
    for col in range(n):
        for row in range(mid_row):
            top_sum += x[row, col]
        for row in range(mid_row, m):
            bot_sum += x[row, col]
    return np.array([[top_sum/(mid_row*n)],[bot_sum/((m-mid_row)*n)]])

def extracted_feature_set(x):
    """extend the individual feature sets on top of each other for one big combined feature set"""
    #loop through n samples of images
    n_samples,m,n = x.shape
    images_feature_set = np.empty((58, 0))
    for image_num in range(n_samples):
        image = x[image_num, :, :]
        row_feat = row_average_features(image)
        col_feat = col_average_features(image)
        top_bot_feat = top_bottom_features(image)
        image_feat = np.concatenate((row_feat, col_feat, top_bot_feat))
        images_feature_set = np.concatenate((images_feature_set, image_feat), axis = 1)
    return images_feature_set

def iterate_individual_feature(x):
    """@param x (n_samples,m,n) array with values in (0,1)
    @return (1,3) tuple array whith the feature set calculated on the set of images
    (row, col, top_bot)"""
    #loop through n samples of images
    n_samples,m,n = x.shape
    row_feature_set = np.empty((28, 0))
    col_feature_set = np.empty((28, 0))
    top_bot_feature_set = np.empty((2, 0))
    for image_num in range(n_samples):
        #access the individual image data
        image = x[image_num, :, :]
        #calculate the features for this image
        row_feat = row_average_features(image)
        col_feat = col_average_features(image)
        top_bot_feat = top_bottom_features(image)
        #add the caculated features to the data set
        row_feature_set = np.concatenate((row_feature_set, row_feat), axis = 1)
        col_feature_set = np.concatenate((col_feature_set, col_feat), axis = 1)
        top_bot_feature_set = np.concatenate((top_bot_feature_set, top_bot_feat), axis = 1)

    return row_feature_set, col_feature_set, top_bot_feature_set


flat = raw_mnist_features(data)
new_features = extracted_feature_set(data)
print(new_features.shape)
print(labels.shape)
# use this function to evaluate accuracy
# acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
acc = hw3.get_classification_accuracy(extracted_feature_set(data), labels)
print('combined feature accuracy:', acc)
#part 6.2 evaluations
#generate the feature sets
row_feature_set, col_feature_set, top_bot_feature_set = iterate_individual_feature(data)
#evaluate the accuracy of each feature set by calling get_classification_accuracy
for feat in (row_feature_set, col_feature_set, top_bot_feature_set):
    acc = hw3.get_classification_accuracy(feat, labels)
    print(f'accuracy for feature set is : {acc}')


#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data
