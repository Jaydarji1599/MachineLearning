import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def get_data():
    readed_data=loadmat('NumberRecognition.mat')

    input_train_eight = np.reshape(np.array(readed_data["imageArrayTraining8"]),[-1,np.array(readed_data["imageArrayTraining8"]).shape[-1]]).transpose()
    input_train_nine = np.reshape(np.array(readed_data["imageArrayTraining9"]),[-1,np.array(readed_data["imageArrayTraining9"]).shape[-1]]).transpose()
    input_test_eights = np.reshape(np.array(readed_data["imageArrayTesting8"]),[-1,np.array(readed_data["imageArrayTesting8"]).shape[-1]]).transpose()
    input_test_nine = np.reshape(np.array(readed_data["imageArrayTesting9"]),[-1,np.array(readed_data["imageArrayTesting9"]).shape[-1]]).transpose()
    #https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    #https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    #take out training and testing dataset
    #reshaping the dataset from 3d to 2d
    #transpose the 2d data set. eg (from[a,b] to [b,a])
    #store it in variables
    label_train_eight = np.ones(input_train_eight.shape[0])
    label_train_nine = np.zeros(input_train_nine.shape[0])
    label_test_eight = np.ones(input_test_eights.shape[0])
    label_test_nine = np.zeros(input_test_nine.shape[0])
    #labeling the dataset

    eightnine__conc = np.concatenate((input_train_eight,input_train_nine), axis=0)
    eigthnine_label__conc = np.concatenate((label_train_eight,label_train_nine), axis=0)
    eightnine__test_conc = np.concatenate((input_test_eights,input_test_nine), axis=0)
    eigthnine_label__test_conc = np.concatenate((label_test_eight,label_test_nine), axis=0)
    #https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    #concatenating our training array of 8 and 9 along column
    #concatenating our testing array of 8 and 9 along column
    return eightnine__conc,eigthnine_label__conc,eightnine__test_conc,eigthnine_label__test_conc


def plotGraph(erros, qType):
    plt.plot(range(1,21), erros)
    plt.xlabel('Value of K from 1...20 KNN')
    plt.ylabel('Error Rate in %')
    if (qType ==1):
      plt.savefig("knn_q1")
    else:
      plt.savefig("knn_q3")
    #https://stackoverflow.com/questions/56153726/plot-k-nearest-neighbor-graph-with-8-features
    #ploting the error rates and values of k from 0 to 20
    #saving the error rate to check how well our model has performed  

def train(input_data,input_label, input_test_data,input_test_label):
  knn_loop = range(1, 21)
  scores = []
  error = []
  for k in knn_loop:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(input_data, input_label)
    output = knn.predict(input_test_data)
    scores.append(metrics.accuracy_score(input_test_label, output))
    print (scores)
    error.append(np.mean(output != input_test_label)*100)
    #https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
    #To Running knn traning and testing
    #creating a loop from k=1 to 20
    #creating a array and storing the values in the array for error rates
  
  
  return scores,error

def question1():
    eightnine__conc,eigthnine_label__conc,eightnine__test_conc,eigthnine_label__test_conc = get_data()
    socres,error = train(eightnine__conc,eigthnine_label__conc,eightnine__test_conc,eigthnine_label__test_conc)
    # plotGraph(error,1)
    #save(error)


def save (errors) -> None:
   import numpy as np
   from pathlib import Path

   arr = np.array(errors)
   if len( arr.shape) > 2 or ( len(arr.shape) == 2 and 1 not in arr.shape):
     raise ValueError(
       "Invalid output shape. Output should be an array "
       "that can be unambiguously raveled/squeezed."
     )
   if arr. dtype not in [np. float64, np.float32, np.float16]:
     raise ValueError( "Your error rates must be stored as float values.")
   arr = arr.ravel()
   if len( arr) != 20 or (arr[ 0] >= arr[ -1]):
     raise ValueError(
       "There should be 20 error values, with the first value "
       "corresponding to k=1, and the last to k=20."
     )
   if arr[ -1] >= 2.0:
     raise ValueError(
       "Final array value too large. You have done something "
       "very wrong (probably relating to standardizing)."
     )
   if arr[ -1] < 0.8:
     raise ValueError(
       "You probably have not converted your error rates to percent values."
     )
   outfile = Path(__file__). resolve().parent / "errors.npy"
   np.save( outfile, arr, allow_pickle=False)
   print(f"Error rates succesfully saved to {outfile }")


#dataset for question 2
#https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+Behavior+Risk
def question2():
      data = pd.read_csv('sobar-72.csv')
      auc_all_scores = []
      auc_col_names = []
      #creating 2 arrays so store the information from data
      for i in data.columns:
            auc_all_scores.append(roc_auc_score(data.iloc[:,-1], data[i]))
            #https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
            #to go through all the collumns in the data      
      auc_col_names.append(data.columns)
      sorted_data = np.column_stack((np.array(auc_col_names).flatten(),np.array(auc_all_scores)))

      #https://www.w3resource.com/numpy/manipulation/ndarray-flatten.php
      #https://www.geeksforgeeks.org/numpy-column_stack-in-python/#:~:text=column_stack()%20in%20Python,-Last%20Updated%20%3A%2006&text=06%20Jan%2C%202019-,numpy.,just%20like%20with%20hstack%20function.
      #changing the array from row to collum so that we can concatinate 2 collumns behavior and auc values, side wise and get the needed data
      data = pd.DataFrame(sorted_data)
      sorted_array = data.sort_values(1, ascending=False)
      #https://www.kite.com/python/answers/how-to-sort-a-pandas-dataframe-in-descending-order-in-python#:~:text=Call%20pd.,on%20the%20values%20in%20column%20.
      #sorting the data in desending order
      print(sorted_array)


def question3():
      data = pd.read_csv('sobar-72.csv')
      inputs = np.array(data.iloc[:,0:19])
      #creating the array for all the values from column 1 to 19
      labels= np.array(data.iloc[:,-1])
      #in the data set the las collum is the lables so taking that last column and putting it in the array

      sc = StandardScaler()
      training_data, testing_data, trainig_label, testing_label = train_test_split(inputs, labels, test_size = 0.1)
      #splitting the data into two parts for testing and training purposed. the data is split into 10% testing data and 90% training data 
      training_data = sc.fit_transform(training_data)
      testing_data = sc.transform(testing_data)
      #standardising the collected data

      scores,error= train(training_data, trainig_label, testing_data, testing_label )


      plotGraph(error,3)


if __name__ == "__main__":
    question1()
    #question2()
    #question3()







