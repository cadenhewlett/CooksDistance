###A Slightly Convoluted way to Calculate and Produce a Cook's Distance Plot for a given set of data.
#Verifying Repository Integrity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import os
import math
#setup
os.getcwd()
os.listdir(os.getcwd())

DEBUGGING = True
data = pd.read_csv('Salary_Data.csv', skipinitialspace = False) #simple working data for salaries based on years of experience for a given companyS

#requires a sub-dataframe that is 2 by 2, not clean but whatver
#linear regression mapping x to y (1st column to 2nd)
def lreg_simple(df, y_to_x = False, info = False, keep_names = True):

    orig_names = list(df.columns.values) #original names for safe keeping

    if(not y_to_x):  #determine which way we're modelling the linear regression
        df.rename(columns = {df.columns[0]: "x", df.columns[1]: "y"}, inplace = True)
    else:
        df.rename(columns = {df.columns[0]: "y", df.columns[1]: "x"}, inplace = True)

    #get the means of our 'sample' x and y
    y_bar = df['y'].mean()
    x_bar = df['x'].mean()
    #calculate the numerator for the beta value
    beta_numerator = sum((data['x'] - x_bar)*(data['y'] - y_bar))
    #and the denominator
    beta_denominator = sum(pow((data['x'] - x_bar), 2))
    #intuitively, divide numerator by denom to get beta
    beta = beta_numerator / beta_denominator
    #now let's solve for alpha
    alpha = y_bar - (beta * x_bar)
    #find residual sum of squares
    rss = sum(pow((df['y'] - alpha - (beta*df['x'])), 2))
    #use rss to get sample variance
    var = rss / (len(data['x']) - 2)
    #use sample variance to get R^2
    r_sq = (1 - (rss / sum(pow((data['y'] - y_bar), 2))))
    #return all values as list
    values = { #make a dictionary of our regression values
        "beta" : beta,
        "alpha" : alpha,
        "variance" : var,
        "R-Squared" : r_sq
    }

    if(info): print("Linear Regression With X Variable " + orig_names[y_to_x])
    #return our columns back to their original places
    if(keep_names): df.rename(columns = {df.columns[0] : orig_names[0], df.columns[1]: orig_names[1]}, inplace = True)

    return values

#verify my LinearRegression if you'd like using the below, but it does work
    # model = LinearRegression()
    # list(data.columns.values)
    # xvals = np.array(data['YearsExperience']).reshape(-1, 1)
    # yvals = np.array(data['Salary']).reshape(-1, 1)
    # model.fit(xvals, yvals)
    # print(float(model.intercept_))
    # print(float(model.coef_))


#calculate cooks distance for a given dataframe
def cooks_distance(df, y_to_x = False): ## NOTE: Currently assumes imput of an x by y df
    #define container for our distance values
    d_values = []
    #define container for extreme d_values; SEE threshold
    extreme_values = []
    #general rule of thumb is 4 / n for extreme distance
    threshold = 4 / len(df)
    #define regression using our function and input dataframe
    original_reg =  lreg_simple(df = df, y_to_x = y_to_x, info = False, keep_names = False)
    #define our predicted values for original regression (y_hat)
    orig_predicted = original_reg['alpha'] + (original_reg['beta'] * df['x'])
    for i in range(len(df)): #repeat for entire dataframe span
            looked_at = df.loc[i] #define which point we're examining
            #create a new data frame without said data point
            df_ex = df.drop([i], axis = 0)
            #calculate the regression without that point
            ex_reg = lreg_simple(df_ex,  y_to_x = y_to_x, keep_names = False, info = False)
            #calculate the regression without the i-th point (y_hat_i)
            ex_reg_predicted = ex_reg['alpha'] + (ex_reg['beta'] * data['x'])
            #calculate the squared difference between the original regression and this new one
            d_i = (sum((orig_predicted - ex_reg_predicted)**2 / (3*original_reg['variance']))) #divided by (p + 1) * variance of first regression
            #append this measure to our distance container
            d_values.append(d_i)
            #if we identify an extreme value
            if(d_i > threshold):
                #document index of extreme value and corresponding values
                extreme_values.append(looked_at)

    #place our outputs into a dictionary with nice keys
    outputs = {
        "distance" : d_values, #distances
        "extremes" : extreme_values #extremes NOTE not necessarily outliers!
    }
    return outputs

cook = cooks_distance(data)
cook
