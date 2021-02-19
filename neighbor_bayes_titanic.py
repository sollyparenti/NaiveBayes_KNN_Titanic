import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt

#Performing K-Nearest Neighbors algorithm using most common label of K nearest neighbors
#trainingData = np array of the training data
#trainingLabels = np array of labels of the training data
#newX = np array of size (n,6) consisting of n samples to be predicted
#K = How many of the nearest neighbors we will look at
#dist = how we will measure distance
#This method returns an np array of predictions of size (n,)
def nearestNeighbors(trainingData,trainingLabels,newX,K=1,dist='euclidean'):
    #Reshape newX into (n,1,6) to vectorize computations
    #This helps us treat it as n (1,6) vectors
    x = (newX).reshape((newX.shape[0],1,newX.shape[1]))
    if (dist == 'euclidean'):
        #Make an array of distances between newX and the training data
        distances = np.sqrt(np.sum((trainingData - x)**2,axis=2))
    #Linearly squash all features into [0,1], then use euclidean distance
    if (dist == 'scaledeuclidean'):
        #First, scale all the features to be in [0,1]
        min_feat = np.min(trainingData,axis = 0)
        max_feat = np.max(trainingData,axis = 0)
        scaled_dist = (trainingData - min_feat) / (max_feat - min_feat)
        scaled_x = (x-min_feat) / (max_feat - min_feat)
        distances = np.sqrt(np.sum((scaled_dist - scaled_x)**2,axis=2))
    #Linearly squash all features into [0,1], then use manhattan distance
    if (dist == 'scaledmanhattan'):
        min_feat = np.min(trainingData,axis = 0)
        max_feat = np.max(trainingData,axis = 0)
        scaled_dist = (trainingData - min_feat) / (max_feat - min_feat)
        scaled_x = (x-min_feat) / (max_feat - min_feat)
        distances = np.sum(np.abs(scaled_dist - scaled_x),axis=2)
    y = trainingLabels.reshape((1,trainingLabels.shape[0]))
    sorted_dist = np.take_along_axis(y,np.argsort(distances,axis=1),axis=1)
    k_nearest_labels = sorted_dist[:,:K]
    #Finally, we return the mode of the k nearest labels
    return stats.mode(k_nearest_labels,axis=1)[0].reshape(newX.shape[0])

#data is a pandas dataframe containing the data and the labels
#newX is an np array of size (n,6) containing n samples to be predicted
#This method returns an np array of predictions of size (n,)
def naiveBayesPredictor(data,newX):
    survived = data.loc[data.Survived == 1].copy()
    dead = data.loc[data.Survived == 0].copy()
    survProb = len(survived) / len(data)
    deadProb = len(dead)/len(data)

    pclassSurv = categoricalProb(survived['Pclass'])
    pclassDead = categoricalProb(dead['Pclass'])
    genderSurv = categoricalProb(survived['Sex'])
    genderDead = categoricalProb(dead['Sex'])
    ageSurv = gaussian(survived['Age'])
    ageDead = gaussian(dead['Age'])
    sibSpSurv = categoricalProb(survived['Siblings/Spouses Aboard'])
    sibSpDead = categoricalProb(dead['Siblings/Spouses Aboard'])
    parKidSurv = categoricalProb(survived['Parents/Children Aboard'])
    parKidDead = categoricalProb(dead['Parents/Children Aboard'])
    fareSurv = gaussian(survived['Fare'])
    fareDead = gaussian(dead['Fare'])

    surviveList = [pclassSurv,genderSurv,ageSurv,sibSpSurv,parKidSurv,fareSurv]
    deadList = [pclassDead,genderDead,ageDead,sibSpDead,parKidDead,fareDead]

    #Prob(y) * Product of Prob(x | y) that we use in Naive Bayes
    surviveNaiveBayes = survProb * naiveProb(surviveList,newX)
    deadNaiveBayes = deadProb * naiveProb(deadList,newX)

    #Given the dead and survival "naive probabilities", we make predictions
    return ((surviveNaiveBayes-deadNaiveBayes) >= 0).astype(int)

#data is a pandas dataframe, so this needs to be a little different
def naive_cross_val(data,folds=10):
    predictions = np.zeros(data.shape[0])
    for n in range(folds):
        if (n == 0):
            newData = data.iloc[data.shape[0]//folds:,:]
            dataToPredict = data.iloc[:data.shape[0]//folds,:].drop(columns=['Survived'],
            inplace=False).to_numpy()
            predictions[:data.shape[0]//folds] = naiveBayesPredictor(newData,
            dataToPredict)
        elif (n == (folds - 1)):
            newData = data.iloc[:((folds-1)*data.shape[0])//folds,:]
            dataToPredict = data.iloc[((folds-1)*data.shape[0])//folds:,:].drop(columns=['Survived'],
            inplace=False).to_numpy()
            predictions[((folds-1)*data.shape[0])//folds:] = naiveBayesPredictor(newData,
            dataToPredict)
        else:
            end = (n *data.shape[0]) // folds
            start = ((n+1) * data.shape[0]) // folds
            newData = data.iloc[:end,:].append(data.iloc[start:,:],ignore_index=True)
            dataToPredict = data.iloc[end:start,:].drop(columns=['Survived'],inplace=False).to_numpy()
            predictions[end:start] = naiveBayesPredictor(newData,
            dataToPredict)
    return predictions

#Given a list of dicts of attributes and a new samples X,
#this method computes Prod Prob(x_j | y)
#For now, our attributes use either categorical distribution or gaussian
#newX is an np array of size (n,6)
#This method returns an np array of size (n,)
def naiveProb(listOfAttributes,newX):
    probs = np.zeros(newX.shape)
    for n in range(len(listOfAttributes)):
        #Checking for Gaussian
        if ('gaussmean' in listOfAttributes[n].keys()):
            probs[:,n] = norm.pdf(newX[:,n],
            loc = listOfAttributes[n]['gaussmean'],scale = listOfAttributes[n]['gaussvar']**.5)
        else:
            templist = listOfAttributes[n].copy()
            #Doing some laplace smoothing
            for key in newX[:,n]:
                if (key not in templist.keys()):
                    templist[key] = 1/len(listOfAttributes[n].keys())
            probs[:,n] = np.vectorize(templist.get)(newX[:,n])
    return probs.prod(axis = 1)

#data is a pandas series
#Computes the sample mean and sample (unbiased) variance
#Returns a dict of the form {'gaussmean': mean, 'gaussvar':var}
def gaussian(data):
    info = {}
    info['gaussmean'] = data.mean()
    info['gaussvar'] = data.var()
    return info

#data is one column of a pandas dataframe, i.e. a series
#returns a dictionary. The keys are possible values of the series, the values
#are the associated probabilities with Laplace smoothing.
def categoricalProb(data):
    table = {}
    for val in data.unique():
        valProb = (len(data.loc[data == val]) + 1) / (len(data) + len(data.unique()))
        table[val] = valProb
    return table

#We perform N fold cross validation for k nearest neighbors.
#We return a numpy array of the same shape as labels. Each entry is the prediction
#for that sample using cross validation.
#So for example, the first tenth of the returned array are the predictions on the 
#first tenth of the samples using the data 
def neigh_cross_val(data,labels,K=1,dist='scaledeuclidean',folds=10):
    predictions = np.zeros(labels.shape)
    for n in range(folds):
        if (n == 0):
            newData = data[data.shape[0]//folds:,:]
            newLabels = labels[data.shape[0]//folds:]
            predictions[:data.shape[0]//folds] = nearestNeighbors(newData,newLabels,
            data[:data.shape[0]//folds,:],K=K,dist=dist)
        elif (n == (folds - 1)):
            newData = data[:((folds-1)*data.shape[0])//folds,:]
            newLabels = labels[:((folds-1)*data.shape[0])//folds]
            predictions[((folds-1)*data.shape[0])//folds:] = nearestNeighbors(newData,
            newLabels,data[((folds-1)*data.shape[0])//folds:,:],K=K,dist=dist)
        else:
            end = (n *data.shape[0]) // folds
            start = ((n+1) * data.shape[0]) // folds
            newData = np.append(data[:end,:],data[start:,:],axis = 0)
            newLabels = np.append(labels[:end],labels[start:])
            predictions[end:start] = nearestNeighbors(newData,newLabels,
            data[end:start,:],K=K,dist=dist)
    return predictions
        
def main():
    data = pd.read_csv('titanic_data.csv')
    y = data['Survived'].to_numpy().T
    X = data.drop(columns=['Survived'],inplace=False).to_numpy()
    print(np.sum(np.abs(y - neigh_cross_val(X,y,K=25,dist='scaledeuclidean'))))
    print(np.sum(np.abs(y - naive_cross_val(data,folds=10))))
    x = np.array([[1,0,28,6,2,72.73]])
    print(naiveBayesPredictor(data,x))
    aliveordead = []
    for k in range(1,50):
        aliveordead.append(nearestNeighbors(X,y,x,K=k,dist='scaledeuclidean')[0])
        if (aliveordead[k-1] == 1):
            print(k)
    plt.plot(aliveordead,'ro')
    plt.ylabel('Dead or Alive')
    plt.yticks([0,1])
    plt.xlabel('K')
    plt.show()
    #plt.savefig('nearestneighbors.png')
    
if __name__ == "__main__":
    main()
