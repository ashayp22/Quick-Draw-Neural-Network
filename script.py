#imports
from mnist import MNIST
import random
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import tkinter as tk
import math
import time
import csv
import scipy as scipy
from scipy import optimize
from scipy.optimize import minimize

WINDOW_SIZE = 140


print("we starting")

# basketball_images = np.load("Images/full_numpy_bitmap_basketball.npy")
# car_images = np.load("Images/full_numpy_bitmap_car.npy")
# flower_images = np.load("Images/full_numpy_bitmap_flower.npy")
# pencil_images = np.load("Images/full_numpy_bitmap_pencil.npy")
# smiley_images = np.load("Images/full_numpy_bitmap_smiley face.npy")
#
# print("one done")

# rainbow_images = np.load("Images2/full_numpy_bitmap_rainbow.npy")
# skull_images = np.load("Images2/full_numpy_bitmap_skull.npy")
# star_images = np.load("Images2/full_numpy_bitmap_star.npy")
# triangle_images = np.load("Images2/full_numpy_bitmap_triangle.npy")
# violin_images = np.load("Images2/full_numpy_bitmap_violin.npy")
#
# print("two done")

# house_images = np.load("Images3/full_numpy_bitmap_house.npy")
# mountain_images = np.load("Images3/full_numpy_bitmap_mountain.npy")
# pants_images = np.load("Images3/full_numpy_bitmap_pants.npy")
potato_images = np.load("Images3/full_numpy_bitmap_potato.npy")
# square_images = np.load("Images3/full_numpy_bitmap_square.npy")

#
# book_images = np.load("Images4/full_numpy_bitmap_book.npy")
# calculator_images = np.load("Images4/full_numpy_bitmap_calculator.npy")
# cloud_images = np.load("Images4/full_numpy_bitmap_cloud.npy")
# eye_images = np.load("Images4/full_numpy_bitmap_eye.npy")
# hammer_images = np.load("Images4/full_numpy_bitmap_hammer.npy")
# print("four done")

# palmtree_images = np.load("Images5/full_numpy_bitmap_palm tree.npy")
# pig_images = np.load("Images5/full_numpy_bitmap_pig.npy")
# pizza_images = np.load("Images5/full_numpy_bitmap_pizza.npy")
# shark_images = np.load("Images5/full_numpy_bitmap_shark.npy")
# tornado_images = np.load("Images5/full_numpy_bitmap_tornado.npy")
# print("four done")

# knife_images = np.load("Images6/full_numpy_bitmap_knife.npy")
# sailboat_images = np.load("Images6/full_numpy_bitmap_sailboat.npy")
# snowman_images = np.load("Images6/full_numpy_bitmap_snowman.npy")
# sun_images = np.load("Images6/full_numpy_bitmap_sun.npy")
# tree_images = np.load("Images6/full_numpy_bitmap_tree.npy")
# print("six done")


#training set
# X1 = []
# y1 = []
# X2 = []
# y2 = []
# X3 = []
# y3 = []
# X4 = []
# y4 = []
# X5 = []
# y5 = []
# X6 = []
# y6 = []


examples_per_type = 3000

# for z in range(examples_per_type):
#     X1.append(basketball_images[random.randint(0, len(basketball_images) - 1)] / 255)  # normalize features
#     y1.append([0])
#     X1.append(car_images[random.randint(0, len(car_images) - 1)] / 255)
#     y1.append([1])
#     X1.append(flower_images[random.randint(0, len(flower_images) - 1)] / 255)
#     y1.append([2])
#     X1.append(pencil_images[random.randint(0, len(pencil_images) - 1)] / 255)
#     y1.append([3])
#     X1.append(smiley_images[random.randint(0, len(smiley_images) - 1)] / 255)
#     y1.append([4])

# for z in range(examples_per_type):
#     X2.append(rainbow_images[random.randint(0, len(rainbow_images) - 1)] / 255)  # normalize features
#     y2.append([0])
#     X2.append(skull_images[random.randint(0, len(skull_images) - 1)] / 255)
#     y2.append([1])
#     X2.append(star_images[random.randint(0, len(star_images) - 1)] / 255)
#     y2.append([2])
#     X2.append(triangle_images[random.randint(0, len(triangle_images) - 1)] / 255)
#     y2.append([3])
#     X2.append(violin_images[random.randint(0, len(violin_images) - 1)] / 255)
#     y2.append([4])

# for z in range(examples_per_type):
#     X3.append(house_images[random.randint(0, len(house_images) - 1)] / 255)  # normalize features
#     y3.append([0])
#     X3.append(mountain_images[random.randint(0, len(mountain_images) - 1)] / 255)
#     y3.append([1])
#     X3.append(pants_images[random.randint(0, len(pants_images) - 1)] / 255)
#     y3.append([2])
#     X3.append(potato_images[random.randint(0, len(potato_images) - 1)] / 255)
#     y3.append([3])
#     X3.append(square_images[random.randint(0, len(square_images) - 1)] / 255)
#     y3.append([4])

# for z in range(examples_per_type):
#     X4.append(book_images[random.randint(0, len(book_images) - 1)] / 255)  # normalize features
#     y4.append([0])
#     X4.append(calculator_images[random.randint(0, len(calculator_images) - 1)] / 255)
#     y4.append([1])
#     X4.append(cloud_images[random.randint(0, len(cloud_images) - 1)] / 255)
#     y4.append([2])
#     X4.append(eye_images[random.randint(0, len(eye_images) - 1)] / 255)
#     y4.append([3])
#     X4.append(hammer_images[random.randint(0, len(hammer_images) - 1)] / 255)
#     y4.append([4])

# for z in range(examples_per_type):
#     X5.append(palmtree_images[random.randint(0, len(palmtree_images) - 1)] / 255)  # normalize features
#     y5.append([0])
#     X5.append(pig_images[random.randint(0, len(pig_images) - 1)] / 255)
#     y5.append([1])
#     X5.append(pizza_images[random.randint(0, len(pizza_images) - 1)] / 255)
#     y5.append([2])
#     X5.append(shark_images[random.randint(0, len(shark_images) - 1)] / 255)
#     y5.append([3])
#     X5.append(tornado_images[random.randint(0, len(tornado_images) - 1)] / 255)
#     y5.append([4])


# for z in range(examples_per_type):
#     X6.append(knife_images[random.randint(0, len(knife_images) - 1)] / 255)  # normalize features
#     y6.append([0])
#     X6.append(sailboat_images[random.randint(0, len(sailboat_images) - 1)] / 255)
#     y6.append([1])
#     X6.append(snowman_images[random.randint(0, len(snowman_images) - 1)] / 255)
#     y6.append([2])
#     X6.append(sun_images[random.randint(0, len(sun_images) - 1)] / 255)
#     y6.append([3])
#     X6.append(tree_images[random.randint(0, len(tree_images) - 1)] / 255)
#     y6.append([4])

#
# X1 = np.array(X1) #convert into numpy array
# X2 = np.array(X2) #convert into numpy array
# X3 = np.array(X3) #convert into numpy array
# X4 = np.array(X4) #convert into numpy array
# X5 = np.array(X5) #convert into numpy array
# X6 = np.array(X6) #convert into numpy array

# print("sizes: " + str(len(X1)) + str(len(X2)) + str(len(X3)))
# print("sizes: " + str(len(X6)) + str(len(X6)))



#FOR FORMATTING AND DISPLAYING THE IMAGES

def showImage(image): #shows the image as it is
    print(len(image))
    length = math.sqrt(len(image))
    t = ""
    for i in range(len(image)):
        if(image[i] > 0):
            t += "$"
        else:
            t += "."
        if(i % length == length - 1):
            print(t)
            t = ""

print(showImage(potato_images[1]))
print(showImage(potato_images[2]))
print(showImage(potato_images[3]))


def formatImage(image): #formats the raw image (big dimensions) into a smaller size to match the dimensions of the training set
    # first, format drawing since it is too big
    features = []  # ending array

    # add empty values to features
    for i in range(28):
        t = []
        for j in range(28):
            t.append(0)
        features.append(t)

    # now, scale down image
    multiplier = int(WINDOW_SIZE / 28)
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            features[int(j / multiplier)][int(i / multiplier)] += image[i][j]

    print("picture")
    for k in features:
        t = ""
        for u in k:
            if u > 0:
                t += "$"
            else:
                t += "."
        print(t)

    features = np.array(features)  # convert the features into
    features = features.flatten()  # make 1 dimension
    print(features)
    features = np.true_divide(features, multiplier**2) # average out
    features = np.true_divide(features, 255) # normalize out
    return features

#NEURAL NETWORK PART------------------------------------------------------------------

input_layer_size = 784 #28 * 28 pixel images
hidden_layer_size = 250 #number of neurons in the hidden layer
num_labels = 5 #number of outputs, or names for the image

#ALL THE FUNCTIONS-------------------------------------------------------------------

def costFunction(theta_1, theta_2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val): #the cost function, returns cost and gradients
    # Implements the neural network cost function for a two layer neural network which performs classification
    #computes the cost and gradient of the neural network. The
    #parameters for the neural network are "unrolled" into the vector
    #nn_params and need to be converted back into the weight matrices.

    #The returned parameter grad should be a "unrolled" vector of the
    #partial derivatives of the neural network.

    #set to theta 1 and theta 2
    Theta1 = theta_1
    Theta2 = theta_2

    #set value of m (number of training examples)
    m = len(y)

    #first, feed forward

    a1 = X
    a1 = np.insert(a1, 0, 1, axis = 1) #adds bias
    a2 = sigmoid(np.dot(a1, Theta1.T)) #gets next layer
    a2 = np.insert(a2, 0, 1, axis = 1) #adds bias
    a3 = sigmoid(np.dot(a2, Theta2.T)) #hypothesis

    #transform the labels into a matrix (ex: [3] is turned into [ [0] [0] [0] [1] [0] ]

    temp_y = []

    for lab in y:
        vec = [0] * 5
        vec[lab[0]] = 1
        temp_y.append(vec)

    y = np.array(temp_y)

    #now, calculate the cost
    J = 1/m * np.sum(np.sum(-y * np.log(a3) - (1-y) * np.log(1-a3)))

    #regularize

    regTheta1 = np.delete(Theta1, 0, axis=1) #won't regularize bias weights
    regTheta2 = np.delete(Theta2, 0, axis=1)

    regularized = lambda_val / (2 * m) * (np.sum(np.sum(regTheta1**2)) + np.sum(np.sum(regTheta2**2))) #calculates the regularized part

    J += regularized #add on regularized part

    #now, calculate the gradients

    #delta values
    #create array with same dimensions as Theta1 and Theta2
    del1 = [0.0] * len(Theta1[0])
    del1 = [del1] * len(Theta1)
    del1 = np.array(del1)

    del2 = [0.0] * len(Theta2[0])
    del2 = [del2] * len(Theta2)
    del2 = np.array(del2)

    for ex in range(len(X)): #each training example
        #get the current activations and y values for the example
        currenta1 = np.array([a1[ex]]) #this to make sure it is a row vector(2D), and not just an array that is 1d
        currenta2 = np.array([a2[ex]])
        currenta3 = np.array([a3[ex]])
        currenty = np.array([y[ex]])
        #first the value for lowercase delta
        delta_3 = currenta3 - currenty
        delta_2 = (np.dot(Theta2.T, delta_3.T)) * sigmoidGradient(np.insert(np.dot(Theta1, currenta1.T), 0, [1], axis=0)) #size ends up being 26 * 1

        #now calculate uppercase delta
        del1 += np.dot(delta_2[1:], currenta1)
        del2 += np.dot(delta_3.T, currenta2)
        # np.add(del1, np.dot(delta_2[1:], currenta1), out=del1, casting="unsafe") #size 25 * 1 times size 1 * 785
        # np.add(del2, np.dot(delta_3.T, currenta2), out=del2, casting="unsafe") #size 25 * 1 times size 1 * 785

    #finally, figure out the gradients with regularization
    Theta1_gradient = 1/m * del1 + lambda_val/m * np.insert(regTheta1, 0, 0, axis = 1)
    Theta2_gradient = 1/m * del2 + lambda_val/m * np.insert(regTheta2, 0, 0, axis = 1)

    return J, Theta1_gradient, Theta2_gradient #returns the gradients

def separateCost(thetas, X, y, lambda_val):
    Theta1, Theta2 = unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

    #set value of m (number of training examples)
    m = len(y)

    #first, feed forward

    a1 = X
    a1 = np.insert(a1, 0, 1, axis = 1) #adds bias
    a2 = sigmoid(np.dot(a1, Theta1.T)) #gets next layer
    a2 = np.insert(a2, 0, 1, axis = 1) #adds bias
    a3 = sigmoid(np.dot(a2, Theta2.T)) #hypothesis

    # #transform the labels into a matrix (ex: 3 is turned into [ [0] [0] [0] [1] [0] ]
    #
    # temp_y = []
    #
    # for lab in y:
    #     vec = [0] * 5
    #     vec[lab[0]] = 1
    #     temp_y.append(vec)
    #
    # y = np.array(temp_y)

    #now, calculate the cost
    J = 1/m * np.sum(np.sum(-y * np.log(a3) - (1-y) * np.log(1-a3)))

    #regularize

    regTheta1 = np.delete(Theta1, 0, axis=1) #won't regularize bias weights
    regTheta2 = np.delete(Theta2, 0, axis=1)

    regularized = lambda_val / (2 * m) * (np.sum(np.sum(regTheta1**2)) + np.sum(np.sum(regTheta2**2))) #calculates the regularized part

    J += regularized #add on regularized part
    return J

def separateGradient(thetas, X, y, lambda_val):

    #set up + forward propagation

    Theta1, Theta2 = unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

    # set value of m (number of training examples)
    m = len(y)

    # first, feed forward

    a1 = X
    a1 = np.insert(a1, 0, 1, axis=1)  # adds bias
    a2 = sigmoid(np.dot(a1, Theta1.T))  # gets next layer
    a2 = np.insert(a2, 0, 1, axis=1)  # adds bias
    a3 = sigmoid(np.dot(a2, Theta2.T))  # hypothesis

    # # transform the labels into a matrix (ex: 3 is turned into [ [0] [0] [0] [1] [0] ]
    #
    # temp_y = []
    #
    # for lab in y:
    #     vec = [0] * 5
    #     vec[lab[0]] = 1
    #     temp_y.append(vec)
    #
    # y = np.array(temp_y)

    # regularize

    regTheta1 = np.delete(Theta1, 0, axis=1)  # won't regularize bias weights
    regTheta2 = np.delete(Theta2, 0, axis=1)


    # now, calculate the gradients

    # delta values
    # create array with same dimensions as Theta1 and Theta2
    del1 = [0.0] * len(Theta1[0])
    del1 = [del1] * len(Theta1)
    del1 = np.array(del1)

    del2 = [0.0] * len(Theta2[0])
    del2 = [del2] * len(Theta2)
    del2 = np.array(del2)

    for ex in range(len(X)):  # each training example
        # get the current activations and y values for the example
        currenta1 = np.array([a1[ex]])  # this to make sure it is a row vector(2D), and not just an array that is 1d
        currenta2 = np.array([a2[ex]])
        currenta3 = np.array([a3[ex]])
        currenty = np.array([y[ex]])
        # first the value for lowercase delta
        delta_3 = currenta3 - currenty
        delta_2 = (np.dot(Theta2.T, delta_3.T)) * sigmoidGradient(
            np.insert(np.dot(Theta1, currenta1.T), 0, [1], axis=0))  # size ends up being 26 * 1

        # now calculate uppercase delta
        del1 += np.dot(delta_2[1:], currenta1)
        del2 += np.dot(delta_3.T, currenta2)
        # np.add(del1, np.dot(delta_2[1:], currenta1), out=del1, casting="unsafe") #size 25 * 1 times size 1 * 785
        # np.add(del2, np.dot(delta_3.T, currenta2), out=del2, casting="unsafe") #size 25 * 1 times size 1 * 785

    # finally, figure out the gradients with regularization
    Theta1_gradient = 1 / m * del1 + lambda_val / m * np.insert(regTheta1, 0, 0, axis=1)
    Theta2_gradient = 1 / m * del2 + lambda_val / m * np.insert(regTheta2, 0, 0, axis=1)

    return pack_thetas(Theta1_gradient, Theta2_gradient)


def sigmoid(mat): #sigmoid function
    return 1 / (1 + np.exp(-mat))

def sigmoidGradient(z): #derivative of the sigmoid
    return sigmoid(z) * (1 - sigmoid(z))

def randInitialWeights(layer_in, layer_out): #returns initial weights for the nn, which are close to zero and break symmetry
    #Note: The first column of the weights corresponds to the parameters for the bias unit
    epsilon = 0.12
    weight = []
    for w in range(layer_out):
        row = []
        for p in range(layer_in + 1):
            w = random.random() * 2 * epsilon - epsilon
            row.append(w)
        weight.append(row)
    return np.array(weight)

def pack_thetas(t1, t2): #packs the thetas
    return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

def unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels): #unpacks the thetas into two theta
    t1_start = 0
    t1_end = hidden_layer_size * (input_layer_size + 1)
    t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
    t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
    return t1, t2

def predictImage(image, theta1, theta2):  # predicts an image
    #feed forwards
    a1 = image
    a1 = np.insert(a1, 0, 1, axis=0)  # adds bias
    a2 = sigmoid(np.dot(a1, theta1.T))  # gets next layer
    a2 = np.insert(a2, 0, 1, axis=0)  # adds bias
    a3 = sigmoid(np.dot(a2, theta2.T))  # hypothesis
    #finds max
    # print(a3)
    result = np.where(a3 == np.amax(a3))
    return result[0]

def imagePrediction(image, theta1, theta2):  # predicts an image
    #feed forwards
    a1 = image
    a1 = np.insert(a1, 0, 1, axis=0)  # adds bias
    a2 = sigmoid(np.dot(a1, theta1.T))  # gets next layer
    a2 = np.insert(a2, 0, 1, axis=0)  # adds bias
    a3 = sigmoid(np.dot(a2, theta2.T))  # hypothesis
    #finds max
    # print(a3)
    result = np.where(a3 == np.amax(a3))
    return result[0], np.amax(a3)

def evaluatePrediction(image, t11, t12, t21, t22, t31, t32, t41, t42, t51, t52, t61, t62):
    p1, v1 = imagePrediction(image, t11, t12)
    p2, v2 = imagePrediction(image, t21, t22)
    p3, v3 = imagePrediction(image, t31, t32)
    p4, v4 = imagePrediction(image, t41, t42)
    p5, v5 = imagePrediction(image, t51, t52)
    p6, v6 = imagePrediction(image, t61, t62)

    print(p2)

    p_arr = np.array([p1, p2, p3, p4, p5, p6])
    v_arr = np.array([v1, v2, v3, v4, v5, v6])

    print(p_arr)
    print(v_arr)

    max_v = np.where(v_arr == np.amax(v_arr))
    max_v = 5
    # max_v = max_v[0]
    # print(max_v[0])

    if max_v == 0:
        if p1 == 0:
            print("the drawing is a basketball")
        elif p1 == 1:
            print("the drawing is a car")
        elif p1 == 2:
            print("the drawing is a flower")
        elif p1 == 3:
            print("the drawing is a pencil")
        elif p1 == 4:
            print("the drawing is a smiley face")
    elif max_v == 1:
        if p2 == 0:
            print("the drawing is a rainbow")
        elif p2 == 1:
            print("the drawing is a skull")
        elif p2 == 2:
            print("the drawing is a star")
        elif p2 == 3:
            print("the drawing is a triangle")
        elif p2 == 4:
            print("the drawing is a violin")
    elif max_v == 2:
        if p3 == 0:
            print("the drawing is a house")
        elif p3 == 1:
            print("the drawing is a mountain")
        elif p3 == 2:
            print("the drawing is pants")
        elif p3 == 3:
            print("the drawing is a potato")
        elif p3 == 4:
            print("the drawing is a square")
    elif max_v == 3:
        if p4 == 0:
            print("the drawing is a book")
        elif p4 == 1:
            print("the drawing is a calculator")
        elif p4 == 2:
            print("the drawing is cloud")
        elif p4 == 3:
            print("the drawing is a eye")
        elif p4 == 4:
            print("the drawing is a hammer")
    elif max_v == 4:
        if p5 == 0:
            print("the drawing is a palm tree")
        elif p5 == 1:
            print("the drawing is a pig")
        elif p5 == 2:
            print("the drawing is pizza")
        elif p5 == 3:
            print("the drawing is a shark")
        elif p5 == 4:
            print("the drawing is a tornado")
    elif max_v == 5:
        if p6 == 0:
            print("the drawing is a knife")
        elif p6 == 1:
            print("the drawing is a sailboat")
        elif p6 == 2:
            print("the drawing is a snowman")
        elif p6 == 3:
            print("the drawing is the sun")
        elif p6 == 4:
            print("the drawing is a tree")



#TRAINING-------------------------------------------------------------------------------

lambda_val = 2
alpha = 0.2
iterations = 300

#LOAD EVERYTHING-----------------------------------------------------------------------------------------


#LOAD ALL THE THETA


theta1_1 = []
theta1_2 = []
theta2_1 = []
theta2_2 = []
theta3_1 = []
theta3_2 = []
theta4_1 = []
theta4_2 = []
theta5_1 = []
theta5_2 = []
theta6_1 = []
theta6_2 = []

with open('Saved/Theta1Grad1.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta1_1.append(temp_gradients)

with open('Saved/Theta2Grad1.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta1_2.append(temp_gradients)

with open('Saved/Theta1Grad2.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta2_1.append(temp_gradients)

with open('Saved/Theta2Grad2.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta2_2.append(temp_gradients)

with open('Saved/Theta1Grad3.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta3_1.append(temp_gradients)

with open('Saved/Theta2Grad3.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta3_2.append(temp_gradients)

with open('Saved/Theta1Grad4.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta4_1.append(temp_gradients)

with open('Saved/Theta2Grad4.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta4_2.append(temp_gradients)


with open('Saved/Theta1Grad5.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta5_1.append(temp_gradients)

with open('Saved/Theta2Grad5.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta5_2.append(temp_gradients)


with open('Saved/Theta1Grad6.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta6_1.append(temp_gradients)

with open('Saved/Theta2Grad6.csv', 'r') as f:
    reader = csv.reader(f)

    # read file row by row
    row_num = 0
    for row in reader:
        row_num += 1 #to make sure the skipped row doesn't get added
        if row_num % 2 == 0:
            continue
        #set to temp
        temp_gradients = []

        #make all floats
        for z in row:
            temp_gradients.append(float(z))

        #now append to alltheta
        temp_gradients = np.array(temp_gradients)
        theta6_2.append(temp_gradients)


theta1_1 = np.array(theta1_1)
theta1_2 = np.array(theta1_2)
theta2_1 = np.array(theta2_1)
theta2_2 = np.array(theta2_2)
theta3_1 = np.array(theta3_1)
theta3_2 = np.array(theta3_2)
theta4_1 = np.array(theta4_1)
theta4_2 = np.array(theta4_2)
theta5_1 = np.array(theta5_1)
theta5_2 = np.array(theta5_2)
theta6_1 = np.array(theta6_1)
theta6_2 = np.array(theta6_2)






#first----------------------------------------------
#
# all_costs = []
#
# for i in range(iterations):
#     cost, grad1, grad2 = costFunction(theta1_1, theta1_2, input_layer_size, hidden_layer_size, num_labels, X1, y1, lambda_val)
#     #update theta
#     theta1_1 -= grad1 * alpha
#     theta1_2 -= grad2 * alpha
#     #add cost for graph
#     all_costs.append(cost)
#     print("running 1: " + str(cost))
#     print(i)
#
# #SAVE THE GRADIENTS TO CSV
#
# with open("Saved/Theta1Grad1.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta1_1)
#
# with open("Saved/Theta2Grad1.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta1_2)
#
# predicted_correct = 0
#
# for example in range(len(X1)):
#     predicted = predictImage(np.array(X1[example]), theta1_1, theta1_2)
#     if y1[example] == predicted:
#         predicted_correct += 1
#
# print("percentage of predicted correct: " + str(predicted_correct / len(X1)))


#second----------------------------------------------

# all_costs = []
#
# for i in range(iterations):
#     cost, grad1, grad2 = costFunction(theta2_1, theta2_2, input_layer_size, hidden_layer_size, num_labels, X2, y2, lambda_val)
#     #update theta
#     theta2_1 -= grad1 * alpha
#     theta2_2 -= grad2 * alpha
#     #add cost for graph
#     all_costs.append(cost)
#     print("running 2: " + str(cost))
#     print(i)
#
# #SAVE THE GRADIENTS TO CSV
#
# with open("Saved/Theta1Grad2.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta2_1)
#
# with open("Saved/Theta2Grad2.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta2_2)
#
# predicted_correct = 0
#
# for example in range(len(X2)):
#     predicted = predictImage(np.array(X2[example]), theta2_1, theta2_2)
#     if y2[example] == predicted:
#         predicted_correct += 1
#
# print("percentage of predicted correct: " + str(predicted_correct / len(X2)))

#
# #third----------------------------------------------
#
# all_costs = []
#
# for i in range(iterations):
#     cost, grad1, grad2 = costFunction(theta3_1, theta3_2, input_layer_size, hidden_layer_size, num_labels, X3, y3, lambda_val)
#     #update theta
#     theta3_1 -= grad1 * alpha
#     theta3_2 -= grad2 * alpha
#     #add cost for graph
#     all_costs.append(cost)
#     print("running 3: " + str(cost))
#     print(i)
#
# #SAVE THE GRADIENTS TO CSV
#
# with open("Saved/Theta1Grad3.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta3_1)
#
# with open("Saved/Theta2Grad3.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta3_2)
#
# predicted_correct = 0
#
# for example in range(len(X3)):
#     predicted = predictImage(np.array(X3[example]), theta3_1, theta3_2)
#     if y3[example] == predicted:
#         predicted_correct += 1
#
# print("percentage of predicted correct: " + str(predicted_correct / len(X3)))

#fourth---------------------------------------------------------------------
#
# # theta4_1 = randInitialWeights(input_layer_size, hidden_layer_size)
# # theta4_2 = randInitialWeights(hidden_layer_size, num_labels)
#
# # alltheta4 = pack_thetas(theta4_1, theta4_2)
#
# # print(len(alltheta4))
# #
# #now we minimize
#
# # alltheta4 = minimize(separateCost, alltheta4, args=(X4, y4, lambda_val), method='trust-krylov', jac=separateGradient, hess=hessGrad, options={'disp': True})
#
# # alltheta4 = scipy.optimize.fmin_bfgs(separateCost, alltheta4, maxiter=400, args=(X4, y4, lambda_val), fprime=separateGradient)
#
# # theta4_1, theta4_2 = unpack_thetas(alltheta4, input_layer_size, hidden_layer_size, num_labels)
#
# all_costs = []
#
# for i in range(iterations):
#     cost, grad1, grad2 = costFunction(theta4_1, theta4_2, input_layer_size, hidden_layer_size, num_labels, X4, y4, lambda_val)
#     #update theta
#     theta4_1 -= grad1 * alpha
#     theta4_2 -= grad2 * alpha
#     #add cost for graph
#     all_costs.append(cost)
#     print("running 4: " + str(cost))
#     print(i)
#
# #SAVE THE GRADIENTS TO CSV
#
# with open("Saved/Theta1Grad4.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta4_1)
#
# with open("Saved/Theta2Grad4.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta4_2)
#
# predicted_correct = 0
#
# for example in range(len(X4)):
#     predicted = predictImage(np.array(X4[example]), theta4_1, theta4_2)
#     if y4[example] == predicted:
#         predicted_correct += 1
#
# print("percentage of predicted correct: " + str(predicted_correct / len(X4)))

#fifth-------------------------------------------------------------------------

# theta5_1 = randInitialWeights(input_layer_size, hidden_layer_size)
# theta5_2 = randInitialWeights(hidden_layer_size, num_labels)

# all_costs = []
#
# for i in range(iterations):
#     cost, grad1, grad2 = costFunction(theta5_1, theta5_2, input_layer_size, hidden_layer_size, num_labels, X5, y5, lambda_val)
#     #update theta
#     theta5_1 -= grad1 * alpha
#     theta5_2 -= grad2 * alpha
#     #add cost for graph
#     all_costs.append(cost)
#     print("running 5: " + str(cost))
#     print(i)
#
# #SAVE THE GRADIENTS TO CSV
#
# with open("Saved/Theta1Grad5.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta5_1)
#
# with open("Saved/Theta2Grad5.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta5_2)
#
# predicted_correct = 0
#
# for example in range(len(X5)):
#     predicted = predictImage(np.array(X5[example]), theta5_1, theta5_2)
#     if y5[example] == predicted:
#         predicted_correct += 1
#
# print("percentage of predicted correct: " + str(predicted_correct / len(X5)))


#sixth-------------------------------------------------------------------------
#
# theta6_1 = randInitialWeights(input_layer_size, hidden_layer_size)
# theta6_2 = randInitialWeights(hidden_layer_size, num_labels)
#
# all_costs = []
#
# for i in range(iterations):
#     cost, grad1, grad2 = costFunction(theta6_1, theta6_2, input_layer_size, hidden_layer_size, num_labels, X6, y6, lambda_val)
#     #update theta
#     theta6_1 -= grad1 * alpha
#     theta6_2 -= grad2 * alpha
#     #add cost for graph
#     all_costs.append(cost)
#     print("running 6: " + str(cost))
#     print(i)
#
# #SAVE THE GRADIENTS TO CSV
#
# with open("Saved/Theta1Grad6.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta6_1)
#
# with open("Saved/Theta2Grad6.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv,delimiter=',')
#     csvWriter.writerows(theta6_2)
#
# predicted_correct = 0
# 
# for example in range(len(X6)):
#     predicted = predictImage(np.array(X6[example]), theta6_1, theta6_2)
#     if y6[example] == predicted:
#         predicted_correct += 1
#
# print("percentage of predicted correct: " + str(predicted_correct / len(X6)))

#PREDICTING---------------------------------------------------------------------------------


#INTERFACE


#vars
mouse_pressed = False #is mouse pressed
pixels = []

def printPixels(): #prints the pixels
    for i in range(0, WINDOW_SIZE):
        text = ""
        for j in range(0, WINDOW_SIZE):
            if pixels[j][i] == 0:
                text += "."
            elif pixels[j][i] == 255:
                text += "$"
        print(text)


def createPixels(): #create matrix representing screen
    global pixels
    pixels = []
    for i in range(0, WINDOW_SIZE):
        pixels.append([])
        for j in range(0, WINDOW_SIZE):
            pixels[i].append(0)


def addPixels(x, y):
    global pixels
    for i in range(x-1, x+2):
        if i < 0 or i >= WINDOW_SIZE:
            continue
        for j in range(y-1, y+2):
            if j < 0 or j >= WINDOW_SIZE:
                continue
            pixels[i][j] = 255

createPixels()

#event handlers
def drawline(event):
    global pixels
    x, y = event.x, event.y
    if canvas.old_coords and mouse_pressed:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1)
        addPixels(x, y)
        addPixels(x1, y1)
        #pixels[x][y] = 255
        #pixels[x1][y1] = 255
        #print(str(x) + " " + str(y))
    canvas.old_coords = x, y

def keydown(e):
    printPixels()
    if e.char == "c":
        canvas.delete("all")
        createPixels()
    elif e.char == "d":
        # predictDrawing(pixels, all_theta)
        formatted = formatImage(pixels)
        print(len(formatted))
        evaluatePrediction(formatted, theta1_1, theta1_2, theta2_1, theta2_2, theta3_1, theta3_2, theta4_1, theta4_2, theta5_1, theta5_2, theta6_1, theta6_2)

def pressed(event):
    global mouse_pressed
    mouse_pressed = True

def released(event):
    global mouse_pressed
    mouse_pressed = False

#window
root = tk.Tk()

root.geometry("" + str(WINDOW_SIZE) + "x" + str(WINDOW_SIZE))

#create canvas
canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE)
canvas.pack()
canvas.old_coords = None

#binds
root.bind('<Motion>', drawline)
root.bind("<KeyPress>", keydown)
root.bind("<Button-1>", pressed)
root.bind("<ButtonRelease-1>", released)

root.mainloop() #loop, no code after gets run


#grad 1 - basketball, car, flower, pencil, smiley face
#grad 2 - rainbow, skull, star, triangle, violin
#grad 3 - house, mountain, pants, potato, square

