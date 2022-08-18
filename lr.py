#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
# %pylab
import numpy as np
import pandas as pd

def get_features(file_path):
    df = pd.read_csv(file_path)
    df['owner'] = df['owner'].str.replace('First Owner','1')
    df['owner'] = df['owner'].str.replace('Second Owner','0.8')
    df['owner'] = df['owner'].str.replace('Third Owner','0.7')
    df['owner'] = df['owner'].str.replace('Test Drive Car','.6')
    df['owner'] = df['owner'].str.replace('Fourth & Above Owner','.55')
    df['owner'] = df['owner'].astype(float)
    extr = df['engine'].str.extract(r'(\d+)', expand=False)
    df['engine'] = pd.to_numeric(extr)
    extr = df['mileage'].str.extract(r'(\d+\.\d+)', expand=False)
    df['mileage'] = pd.to_numeric(extr)
    
    """Giving weights according to my intuition :) """

    df['transmission'] = df['transmission'].str.replace('Manual','0')
    df['transmission'] = df['transmission'].str.replace('Automatic','.2')
    df['transmission'] = pd.to_numeric(df['transmission'])
    df['transmission'] = df['transmission'].astype(float)
    df['seller_type'] = df['seller_type'].str.replace('Individual','.3')
    df['seller_type'] = df['seller_type'].str.replace('Dealer','.1')
    df['seller_type'] = df['seller_type'].str.replace('Trustmark Dealer','.3')
    df['seller_type'] = df['seller_type'].str.replace('Trustmark .1','.3')
    df['fuel'] = df['fuel'].str.replace('Diesel','0')
    df['fuel'] = df['fuel'].str.replace('Petrol','0')
    df['fuel'] = df['fuel'].str.replace('LPG','.1')
    df['fuel'] = df['fuel'].str.replace('CNG','.1')
    df['fuel'] = df['fuel'].astype(float)
    df['seller_type'] = df['seller_type'].astype(float)
    extr = df['torque'].str.extract(r'(\d+)', expand=False)
    df['torque'] = pd.to_numeric(extr)
    df = df.drop('Index', axis=1)        #dropping the index column

    df['max_power'].fillna(value=df['max_power'].mean(), inplace=True)   #filling the empty/holes in data set with avg values
    df.engine.fillna(value=df['engine'].mean(), inplace=True)
    df.torque.fillna(value=df.torque.mean(), inplace=True)
    df.seats.fillna(value=df.seats.mean(), inplace=True)
    df.mileage.fillna(value=df.mileage.mean(), inplace=True)
    df["year"]=(df.year - df.year.min()) / (df.year.max() - df.year.min())
    df["engine"]=(df.engine - df.engine.min()) / (df.engine.max() - df.engine.min())
    df["km_driven"]=(df.km_driven - df.km_driven.min()) / (df.km_driven.max() - df.km_driven.min())
    df["mileage"]=(df.mileage - df.mileage.min()) / (df.mileage.max() - df.mileage.min())
    df["max_power"]=(df.max_power - df.max_power.min()) / (df.max_power.max() - df.max_power.min())
    df["torque"]=(df.torque - df.torque.min()) / (df.torque.max() - df.torque.min())
    df["seats"]=(df.seats - df.seats.min()) / (df.seats.max() - df.seats.min())
    df["bias"]=1
    df['max_power'].fillna(value=df['max_power'].mean(), inplace=True)
    df.engine.fillna(value=df['engine'].mean(), inplace=True)
    df.torque.fillna(value=df.torque.mean(), inplace=True)
    df.seats.fillna(value=df.seats.mean(), inplace=True)
    df.mileage.fillna(value=df.mileage.mean(), inplace=True)    
    phi = df.drop('selling_price', axis=1)
    
    return phi

def get_features_for_test(file_path):
    """this is to handle test.csv file as it does have selling price column"""
    df = pd.read_csv(file_path)
    
    """Giving weights according to my intuition :) """
    
    df['owner'] = df['owner'].str.replace('First Owner','1')
    df['owner'] = df['owner'].str.replace('Second Owner','0.8')
    df['owner'] = df['owner'].str.replace('Third Owner','0.7')
    df['owner'] = df['owner'].str.replace('Test Drive Car','.6')
    df['owner'] = df['owner'].str.replace('Fourth & Above Owner','.55')
    df['owner'] = df['owner'].astype(float)
    extr = df['engine'].str.extract(r'(\d+)', expand=False)
    df['engine'] = pd.to_numeric(extr)
    extr = df['mileage'].str.extract(r'(\d+\.\d+)', expand=False)
    df['mileage'] = pd.to_numeric(extr)
    df['transmission'] = df['transmission'].str.replace('Manual','0')
    df['transmission'] = df['transmission'].str.replace('Automatic','.2')
    df['transmission'] = pd.to_numeric(df['transmission'])
    df['transmission'] = df['transmission'].astype(float)
    df['seller_type'] = df['seller_type'].str.replace('Individual','.3')
    df['seller_type'] = df['seller_type'].str.replace('Dealer','.1')
    df['seller_type'] = df['seller_type'].str.replace('Trustmark Dealer','.3')
    df['seller_type'] = df['seller_type'].str.replace('Trustmark .1','.3')
    df['fuel'] = df['fuel'].str.replace('Diesel','0')
    df['fuel'] = df['fuel'].str.replace('Petrol','0')
    df['fuel'] = df['fuel'].str.replace('LPG','.1')
    df['fuel'] = df['fuel'].str.replace('CNG','.1')
    df['fuel'] = df['fuel'].astype(float)
    df['seller_type'] = df['seller_type'].astype(float)
    extr = df['torque'].str.extract(r'(\d+)', expand=False)
    df['torque'] = pd.to_numeric(extr)
    df = df.drop('Index', axis=1)

    df['max_power'].fillna(value=df['max_power'].mean(), inplace=True)
    df.engine.fillna(value=df['engine'].mean(), inplace=True)
    df.torque.fillna(value=df.torque.mean(), inplace=True)
    df.seats.fillna(value=df.seats.mean(), inplace=True)
    df.mileage.fillna(value=df.mileage.mean(), inplace=True)  
    df["year"]=(df.year - df.year.min()) / (df.year.max() - df.year.min())
    df["engine"]=(df.engine - df.engine.min()) / (df.engine.max() - df.engine.min())
    df["km_driven"]=(df.km_driven - df.km_driven.min()) / (df.km_driven.max() - df.km_driven.min())
    df["mileage"]=(df.mileage - df.mileage.min()) / (df.mileage.max() - df.mileage.min())
    df["max_power"]=(df.max_power - df.max_power.min()) / (df.max_power.max() - df.max_power.min())
    df["torque"]=(df.torque - df.torque.min()) / (df.torque.max() - df.torque.min())
    df["seats"]=(df.seats - df.seats.min()) / (df.seats.max() - df.seats.min())
    df["bias"]=1
    df['max_power'].fillna(value=df['max_power'].mean(), inplace=True)
    df.engine.fillna(value=df['engine'].mean(), inplace=True)
    df.torque.fillna(value=df.torque.mean(), inplace=True)
    df.seats.fillna(value=df.seats.mean(), inplace=True)
    df.mileage.fillna(value=df.mileage.mean(), inplace=True)
    phi=df  
    
    return phi

def generate_output(phi_test,w):
    df = pd.read_csv('test.csv')
    y_predicted=np.dot(phi_test,w)    # The predicted value of Y
    df1 = pd.DataFrame(abs(y_predicted), columns = ['Expected']) 
    df1.to_csv("output.csv")  #saving the generated output file as csv

def compute_RMSE(phi, w , y) :
    y_pred =np.dot(phi,w)          # The current predicted value of Y
    error= sum((y-y_pred)**2)/len(y)   # mean squared error
    return np.sqrt(error)

def closed_soln(phi, y):
    w=np.linalg.pinv(phi).dot(y).astype('float')  
    return w
def gradient_descent(phi, y,phi_dev,y_dev) :
    
    w= np.zeros((len(phi[0]),1))   # initial weight vector
    tolerance=0.1   # minimum movement allowed in gd algorithm
    L =0.1  # The learning Rate
    epochs=100000    # The number of iterations to perform gradient descent
    n = float(len(y)) # Number of elements in X
    # Performing Gradient Descent     
    for i in range(epochs): 
        y_predict =np.dot(phi,w) # The current predicted value of Y
        b = y-y_predict
        d=np.dot(phi.transpose(),b)
        
        D_m = (-2/n)*d    # Derivative wrt w
        c = -L*D_m
        
        if np.all(np.abs(c) <= tolerance):  # Stopping parameter
#             print("fff")
            break

          # Update w
        w = w - np.dot(L, D_m)
    return w


def one_hot_encoder(phi,phi_dev,phi_test):
    df= pd.concat([phi,phi_dev,phi_test],sort=False)
    df['test'] = df['name'].apply(lambda x: ' '.join(x.split()[:4]))  
    
    """Taking first four words of 'name'of cars 
    but we can take shorter names to get faster results as I have done in error plotting of part4 of Question doc given"""
    df_cars = pd.get_dummies(df.test, prefix='car', drop_first=True)
    df = pd.concat([df, df_cars], axis=1)
    df = df.drop('name', axis=1)
    df=df.drop('test',axis=1)
    df.to_csv("jnjd.csv",mode='w',index=False)
    
    return df[:4500],df[4500:5500],df[5500:]

def closed_soln_for_regularizer(phi, y):
    """to find the closed form soln form L2 regularizer"""
    l=(phi.transpose().dot(phi)+ 1000*np.identity(len(phi[0]), dtype =int))
    k=np.linalg.pinv(l).dot(phi.transpose())
    w=k.dot(y).astype('float')
    return w
def pnorm(phi, y, phi_dev, y_dev,t) :
    # Building the model
    w= np.zeros((len(phi[0]),1))   # initial weight vector
    tolerance=0.01   # minimum movement allowed in gd algorithm
    L =0.00000001  # The learning Rate
    epochs =100000  # The number of iterations to perform gradient descent

    # Performing Gradient Descent 
    for i in range(epochs): 
        y_predict =np.dot(phi,w) # The current predicted value of Y
        b = y-y_predict
        d=np.dot(phi.transpose(),b)
        D_m = ((-2*d) + t*1000*(w**(t-1)))    # Derivative wrt w
        
        v=np.abs(-L*D_m)
        if np.all(np.abs(v) <= tolerance):  # Stopping parameter
#             print("fff")
            break
        w = w - np.dot(L, D_m)        # Update w
    return w

def sgd(phi, y, phi_dev, y_dev) :
    epochs=100000
    tolerance=0.1    # minimum movement allowed in sgd algorithm
    w= np.zeros((len(phi[0]),1))   # initial weight vector
    L= 0.1     # The learning Rate
    
    # Performing Gradient Descent 
    for i in range(epochs):
        z=np.random.randint(0,4363)
        p=phi[z]
        q=y[z]
        p=p.reshape((len(phi[0])),1)
        y_predict =np.dot(p.transpose(),w)     # The current predicted value of Y
        b = q-y_predict
        b=b.reshape(1,1)
        d=np.dot(p,b)
        D_m = (-2*d)    # Derivative wrt w
        
        v=np.abs(-L*D_m)
        if np.all(np.abs(v) <= tolerance):   # Stopping parameter
#             print("fff")
            break
        w = w - np.dot(L, D_m)        # Update w
    return w

def get_features_basis(file_path):
    df = pd.read_csv(file_path)
    
    """Giving weights according to my intuition :) """
    
    df['owner'] = df['owner'].str.replace('First Owner','1')
    df['owner'] = df['owner'].str.replace('Second Owner','0.8')
    df['owner'] = df['owner'].str.replace('Third Owner','0.7')
    df['owner'] = df['owner'].str.replace('Test Drive Car','.6')
    df['owner'] = df['owner'].str.replace('Fourth & Above Owner','.55')
    df['owner'] = df['owner'].astype(float)
    extr = df['engine'].str.extract(r'(\d+)', expand=False)
    df['engine'] = pd.to_numeric(extr)
    extr = df['mileage'].str.extract(r'(\d+\.\d+)', expand=False)
    df['mileage'] = pd.to_numeric(extr)
    df['transmission'] = df['transmission'].str.replace('Manual','.1')
    df['transmission'] = df['transmission'].str.replace('Automatic','.2')
    df['transmission'] = pd.to_numeric(df['transmission'])
    df['transmission'] = df['transmission'].astype(float)
    df['seller_type'] = df['seller_type'].str.replace('Individual','.3')
    df['seller_type'] = df['seller_type'].str.replace('Dealer','.1')
    df['seller_type'] = df['seller_type'].str.replace('Trustmark Dealer','.3')
    df['seller_type'] = df['seller_type'].str.replace('Trustmark .1','.3')
    df['fuel'] = df['fuel'].str.replace('Diesel','.1')
    df['fuel'] = df['fuel'].str.replace('Petrol','.1')
    df['fuel'] = df['fuel'].str.replace('LPG','0')
    df['fuel'] = df['fuel'].str.replace('CNG','0')
    df['fuel'] = df['fuel'].astype(float)
    df['seller_type'] = df['seller_type'].astype(float)
    extr = df['torque'].str.extract(r'(\d+)', expand=False)
    df['torque'] = pd.to_numeric(extr)
    df = df.drop('Index', axis=1)

    df['max_power'].fillna(value=df['max_power'].mean(), inplace=True)
    df.engine.fillna(value=df['engine'].mean(), inplace=True)
    df.torque.fillna(value=df.torque.mean(), inplace=True)
    df.seats.fillna(value=df.seats.mean(), inplace=True)
    df.mileage.fillna(value=df.mileage.mean(), inplace=True)
    
    df["year"]=(df.year - df.year.min()) / (df.year.max() - df.year.min())
    df["engine"]=(df.engine - df.engine.min()) / (df.engine.max() - df.engine.min())
    df["km_driven"]=(df.km_driven - df.km_driven.min()) / (df.km_driven.max() - df.km_driven.min())
    df["mileage"]=(df.mileage - df.mileage.min()) / (df.mileage.max() - df.mileage.min())
    df["max_power"]=(df.max_power - df.max_power.min()) / (df.max_power.max() - df.max_power.min())
    df["torque"]=(df.torque - df.torque.min()) / (df.torque.max() - df.torque.min())
    df["seats"]=(df.seats - df.seats.min()) / (df.seats.max() - df.seats.min())
    df["bias"]=1
 
    df['max_power'].fillna(value=df['max_power'].mean(), inplace=True)
    df.engine.fillna(value=df['engine'].mean(), inplace=True)
    df.torque.fillna(value=df.torque.mean(), inplace=True)
    df.seats.fillna(value=df.seats.mean(), inplace=True)
    df.mileage.fillna(value=df.mileage.mean(), inplace=True)
    """trying to find a good basis by hit and trial"""
    df.engine= df.engine**(.75)
    df.torque=df.torque**(.18)
    df.seats= df.seats**(1.5)
    df.mileage=df.mileage**(2.2)
    df.year=df.year**(5)
    df.max_power=df.max_power**(.85)
    df.torque=df.torque**(3)
    df.seller_type=df.seller_type**(1.5)
    df.fuel=df.fuel**8
    df.owner=df.owner**(.9)
    
    phi = df.drop('selling_price', axis=1)
    return phi    

def plot_with_samples():
    phase = "train"
    phi = get_features('train.csv')

    phase = "eval"
    phi_dev = get_features('dev.csv')
    phi_test = get_features_for_test('test.csv')
    phi,phi_dev,phi_test=one_hot_encoder(phi,phi_dev,phi_test)

    """ I have changed slightly code of above funstion to take only first two word of 'name' of cars
    so that we get results in reasonable amount of time"""

    df=pd.read_csv('train.csv')
    y=df['selling_price']
    y = y.to_numpy()
    phi = phi.to_numpy()
    y=y.reshape(len(y),1)

    df=pd.read_csv('dev.csv')
    y_dev=df['selling_price']
    y_dev = y_dev.to_numpy()
    phi_dev = phi_dev.to_numpy()
    y_dev=y_dev.reshape(len(y_dev),1)


    rmse=[]
    for n in [2000,2500,3000,4500]:
        phi_sample=phi[:n]
        y_sample=y[:n]
        w1 = gradient_descent(phi_sample, y_sample,phi_dev,y_dev)
        rmse=rmse+[compute_RMSE(phi_dev, w1, y_dev)]

    plt.plot([2000,2500,3000,4500],rmse,'.')    
    plt.xlabel("size of samples")
    plt.ylabel("Root mean sqare error")
    plt.legend(['error'])

def main():
    
#     phase = "train"
#     phi = get_features('train.csv')
    
#     phase = "eval"
#     phi_dev = get_features('dev.csv')
#     phi_test = get_features_for_test('test.csv')
#     phi,phi_dev,phi_test=one_hot_encoder(phi,phi_dev,phi_test)
    
#     df=pd.read_csv('train.csv')
#     y=df['selling_price']
#     phi = phi.to_numpy()
#     y = y.to_numpy()
#     y=y.reshape(len(y),1)
    
#     df=pd.read_csv('dev.csv')
#     y_dev=df['selling_price']
#     phi_dev = phi_dev.to_numpy()
#     y_dev = y_dev.to_numpy()
#     y_dev=y_dev.reshape(len(y_dev),1)
    
#     phi_test = phi_test.to_numpy() 
    '''Here I have used another intuition for setting optimum learning rate and epochs 
    by checking the obtained weights by their closed form solutions'''
#     w1 = closed_soln(phi, y)
#     generate_output(phi_test,w1)
#     w2=gradient_descent(phi, y,phi_dev,y_dev)
#     r1 = compute_RMSE(phi_dev, w1, y_dev)
#     print(r1)
#     r2 = compute_RMSE(phi_dev, w2, y_dev)
#     print('1a: ')
#     print(abs(r1-r2))
#     w3 = sgd(phi, y, phi_dev, y_dev)
#     r3 = compute_RMSE(phi_dev, w3, y_dev)
#     print(w3)
#     print(abs(r2-r3)
    
#     w1=closed_solnt(phi,y)  
    '''gives closed form soln for with regularizer (lambda=1000)'''   
#     print(w1)
#     w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
#     print(w_p2)
#     w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)
#     w1 = closed_soln_for_regulizer(phi, y)
#     print(w_p4)
#     r= compute_RMSE(phi_dev, w1, y_dev)
#     r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
#     print(r)
#     print(r_p2)
#     r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
#     print(r_p4)

#     phase = "train"
#     phi = get_features_basis('train.csv')
    
#     phase = "eval"
#     phi_dev = get_features_basis('dev.csv')
#     phi_test = get_features_for_test('test.csv')
#     phi_basis,phi_dev,phi_test=one_hot_encoder(phi,phi_dev,phi_test)
#     df=pd.read_csv('train.csv')
#     y=df['selling_price']
#     phi = phi.to_numpy()
#     y = y.to_numpy()
#     y=y.reshape(len(y),1)  
#     phase = "eval"
#     phi_dev, y_dev = get_features_basis('dev.csv')
#     w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
#     w1 = closed_soln(phi_basis, y)
#     rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
#     r1 = compute_RMSE(phi_dev, w1, y_dev)
#     print('1a: ')
#     print((r1))
#     print('Task 3: basis')
#     print(rmse_basis)


      ####### Real Program #########
    phase = "train"
    phi = get_features('train.csv')
    
    phase = "eval"
    phi_dev = get_features('dev.csv')
    phi_test = get_features_for_test('test.csv')
    phi,phi_dev,phi_test=one_hot_encoder(phi,phi_dev,phi_test)
    
    df=pd.read_csv('train.csv')
    y=df['selling_price']
    phi = phi.to_numpy()
    y = y.to_numpy()
    y=y.reshape(len(y),1)
    
    df=pd.read_csv('dev.csv')
    y_dev=df['selling_price']
    phi_dev = phi_dev.to_numpy()
    y_dev = y_dev.to_numpy()
    y_dev=y_dev.reshape(len(y_dev),1)
    
    phi_test = phi_test.to_numpy() 
    
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))
    
    ######## Task 2 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    ######## Task 3 #########

    phase = "train"
    phi = get_features_basis('train.csv')
    phase = "eval"
    phi_dev = get_features_basis('dev.csv')
    phi_test = get_features_for_test('test.csv')
    phi_basis,phi_dev,phi_test=one_hot_encoder(phi,phi_dev,phi_test)
    
    df=pd.read_csv('train.csv')
    y=df['selling_price']
    phi_basis = phi_basis.to_numpy()
    y = y.to_numpy()
    y=y.reshape(len(y),1)
    
    df=pd.read_csv('dev.csv')
    y_dev=df['selling_price']
    phi_dev = phi_dev.to_numpy()
    y_dev = y_dev.to_numpy()
    y_dev=y_dev.reshape(len(y_dev),1)
    
    w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
    rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
    print('Task 3: basis')
    print(rmse_basis)
    
    generate_output(phi_test,w1)     # to find the predicted selling prices on test set
    plot_with_samples()             # to find the plot of error in dev set vs sample size in training set

main()

"""Here I tried to find the covariance of different features with selling price and reasoned out which two
properties are least useful """
# df.selling_price.corr(df.engine)
# df.selling_price.corr(df.year)
# df.selling_price.corr(df.fuel)
# df.selling_price.corr(df.transmission)
# df.selling_price.corr(df.max_power)
# df.selling_price.corr(df.torque)
# df.selling_price.corr(df.km_driven)
# df.selling_price.corr(df.seats)
# df.selling_price.corr(df.seller_type)
# df.selling_price.corr(df.owner)
# df.selling_price.corr(df.mileage)
# df.selling_price.corr(df.owner)
"""Here I tried to find the graph of different features with selling price and reasoned out what polynomial 
will make it more linear"""
# df.plot.line(x='seats', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='engine', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='transmission', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='max_power', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='torque', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='seller_type', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='mileage', y='selling_price', linestyle='', marker='o')
# df.plot.line(x='km_driven', y='selling_price', linestyle='', marker='o')


# In[ ]:




