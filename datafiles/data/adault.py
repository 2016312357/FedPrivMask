# Functions to load and prepare data
# Init global infos

import numpy as np


inputs = (
    ("age", ("continuous",)),
    ("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")),
    ("fnlwgt", ("continuous",)),
    ("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")),
    ("education-num", ("continuous",)),
    ("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse")),
    ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")),
    ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")),
    ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
    ("sex", ("Female", "Male")),
    ("capital-gain", ("continuous",)),
    ("capital-loss", ("continuous",)),
    ("hours-per-week", ("continuous",)),
    ("native-country", ("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands")),


)

input_shape = []
for i in inputs:
    if i[0]=='sex' or i[0]=='race':
        print(i[0],sum(input_shape))
    count = len(i[1])
    input_shape.append(count)
input_dim = sum(input_shape)
print("input_shape:", input_shape)
print("input_dim:", input_dim)

outputs = (0, 1)  # (">50K", "<=50K")
output_dim = 2  # len(outputs)
print("output_dim:", output_dim)


def isFloat(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value):
                summ += float(value)
                count += 1
        means.append(summ / count)
    return means


def prepare_data(raw_data, means):
    X = raw_data[:, :-1]
    y = raw_data[:, -1:]

    # X:
    def flatten_persons_inputs_for_model(person_inputs):
        global inputs
        global input_shape
        global input_dim
        global means
        float_inputs = []

        for i in range(len(input_shape)):  # each attribute i has input_shape[i] values
            features_of_this_type = input_shape[i]
            is_feature_continuous = features_of_this_type == 1

            if is_feature_continuous:
                mean = means[i]
                if isFloat(person_inputs[i]):
                    scale_factor = 1 / (2 * mean)  # we prefer inputs mainly scaled from -1 to 1.
                    float_inputs.append(float(person_inputs[i]) * scale_factor)
                else:
                    float_inputs.append(mean)
            else:
                for j in range(features_of_this_type):
                    feature_name = inputs[i][1][j]

                    if feature_name == person_inputs[i]:
                        float_inputs.append(1.)
                    else:
                        float_inputs.append(0)
        return float_inputs

    new_X = []
    for person in range(len(X)):
        formatted_X = flatten_persons_inputs_for_model(X[person])
        new_X.append(formatted_X)
    new_X = np.array(new_X)

    # y:
    new_y = []
    for i in range(len(y)):
        if y[i] == ">50k":
            new_y.append(1)
        else:  # y[i] == "<=50k":
            new_y.append(0)
    new_y = np.array(new_y)

    return (new_X, new_y)


# Building training and test data

training_data = np.genfromtxt('adult.data.txt', delimiter=', ', dtype=str, autostrip=True)
print("Training data count:", len(training_data))
test_data = np.genfromtxt('adult.test.txt', delimiter=', ', dtype=str, autostrip=True)
print("Test data count:", len(test_data))

x_all=np.concatenate((training_data, test_data), 0)
print(x_all)
x_clean=[]
for i in x_all:
    if '?' in i:
        pass
    else:
        x_clean.append(i)
print(len(x_clean))
x_clean=np.asarray(x_clean)
means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))
print("Mean values for data types (if continuous):", means)

X_train, y_train = prepare_data(x_clean, means)
np.savez('adult.npz',data=X_train,label=y_train)
# X_test, y_test = prepare_data(test_data, means)
#
percent = sum([i for i in y_train])/len(y_train)
print("Training data percentage that is >50k:", percent*100, "%")
#
# # Explanation on data format
#
# print("Training data format example:")
# print(X_train[4])  # 4 is a random person, from cuba.
# print()
#
# print("In fact, we just crushed the data in such a way that it will optimise the neural network (model). \n\
# It is crushed according to the `input_shape` variable: \n\
#     say, if there are 41 native countries in the dataset, there will be 41 input dimensions for the \n\
#     neural network with a value of 0 for every 41 input node for a given person, except the \n\
#     node representing the real country of the person which will have a value of 1. For continuous \n\
#     values, they are normalised to a small float number.")

for i in X_train:
    if len(i) != input_dim:
        raise Exception(
            "Every person should have 105 data fields now. {} here.".format(len(i)))


