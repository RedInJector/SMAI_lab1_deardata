import numpy as np

#Examples of using NumPy (Numerical Python) library.
#NumPy has methods for working with arrays, linear algebra,
#fourier transform and matrices.
#NumPy ndarray works much faster than standard Python lists.
#Most of the parts requiring fast computation are written in C or C++

myArray = np.array([1, 2, 3])   #basic numpy array example.
#print(type(myArray), myArray)

#ARRAY DIMENTIONS
array0D = np.array(33)                          #array with single value
array1D = np.array([1, 2, 3])                   #1D array  
array2D = np.array([[1, 2, 3], [4, 5, 6]])      #2D array

#print(array0D.ndim, array1D.ndim, array2D.ndim)

#ACCESSIG ARRAY ELEMENTS: To access an element you specify them starting with the
#print(array2D[0,1] + array2D[1,2])  #most external one: row and column in this case.
#print(array2D[0,-1])   #Array elements can also be acessed from the end.
#print(array2D[0,0:2])  #Ranges can be sliced - this works the same as a for loop.
#print(array2D[0:2,1])  #We can also print specific elements from all subsets.
#print(array2D[:,0])    #Printing fist elements from all rows
#print(array2D[0:2,0:2]) #Creating array subsets - this will return an array 

#DATA TYPES - those matter in NumPy.
#print(array2D.dtype)                                #we can check the data type
newArray1D = np.array([1.1,2.2,3.3,4.4], dtype='f') #and define it for a new array
#print(newArray1D.dtype)                              

#COPY AND VIEW
arrayCopy = newArray1D.copy()
arrayView = newArray1D.view()
newArray1D[0] = 5.5
#print(arrayCopy, arrayView)
#print(arrayCopy.base, arrayView.base)   #Checking if the array owns its data

#ARRAY SHAPE
#print(array2D.shape)                #Checking shape of an array
newArray2D = newArray1D.reshape(2,2)#Reshaping an existing array
#print(newArray2D)
newArray1D = newArray2D.reshape(-1) #We can also flatten the multidimentional array
#print(newArray1D)

#ITERATING THROUGH ARRAY ELEMENTS
newArrayForIteration3D = np.array([[[0,1,2,3],[4,5,6,7]],[[8,9,10,11],[12,13,14,15]]])
for set in newArrayForIteration3D:
    for row in set:                     
        for element in row:
            element+=1
            #print(element)
#The above might be difficult for highly-dimentional arrays
for x in np.nditer(newArrayForIteration3D): #Alternative method to iterate through all elements
    x=x+1
    #print(x)
#Sometimes we would also require elements index:
for idx, x in np.ndenumerate(newArrayForIteration3D):
    x=x+1
    #print(idx, x)

#JOINING ARRAYS
firstArray = np.array([1, 2, 3])
secondArray = np.array([4, 5, 6])
newArrayConcatenated = np.concatenate((firstArray, secondArray))
#print(newArrayConcatenated) 
newArrayStacked = np.stack((firstArray, secondArray), axis=1)#If axis is not passed it is taken as 0
#print(newArrayStacked) 
arrayStackedColumns = np.vstack((firstArray, secondArray))
arrayStackedRows = np.hstack((firstArray, secondArray))
#print(arrayStackedColumns, arrayStackedRows)
arrayDepthStack = np.dstack((firstArray, secondArray)) #Stacking along depth/height
arrayHeightStack = np.hstack((firstArray, secondArray))
#print(arrayDepthStack)
#print(arrayHeightStack)

#SPLITTING ARRAYS
arrayToSplit = np.array([1,2,3,4,5,6,7,8,9])
newArraySet = np.array_split(arrayToSplit, 3)    #splits the orignal array into 3 new ones.
#print(newArraySet)
array2DToSplit = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], 
                           [16, 17, 18]])
new2DArraySet = np.array_split(array2DToSplit, 3)
#print(new2DArraySet)
array2DSplitAlongRows = np.hsplit(array2DToSplit, 3)
#print(array2DSplitAlongRows)

#SEARCHING ELEMENTS
arrayWithElements = np.array([3, 4, 4, 5, 6, 1, 2, 4, 7, 8, 9])
arrayWithSortedValues = np.array([1,2,3,4,5,6,8,9])
elemntLocation = np.where(arrayWithElements==4) #Finding indices of elements qual to 4
#print(elemntLocation)
#print(np.searchsorted(arrayWithSortedValues,7)) #finding where the value needs to be inserted to maitain sorted lits

#SORTING
unsortedArrayNumbers = np.array([5,4,2,1,3,0])
unsorted2DArrayNumbers = np.array([[3,2,1],[5,6,4]])
unsortedArrayStrings = np.array(['something', 'else', 'or', 'different'])
#print(np.sort(unsortedArrayNumbers),np.sort(unsorted2DArrayNumbers),np.sort(unsortedArrayStrings))

#FILTERING
arrayOriginal = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
elementsForNewArray = [True, False, True, False, False, False, True, False, False, True]
filteredArray= arrayOriginal[elementsForNewArray]
#print(filteredArray)
#Creating filters according to requirements
filterToUse = []
for element in arrayOriginal:
    if(element%2==0):
        filterToUse.append(True)
    else:
        filterToUse.append(False)
arrayWithEvenValues = arrayOriginal[filterToUse]
#print(arrayWithEvenValues)
filterFromArrayFasterVersion = arrayOriginal>45
#print(arrayOriginal[filterFromArrayFasterVersion])

#RANDOM
#Requires importing the random module, or using np.random
#np.random.seed(123)
exampleRandomNumberInt = np.random.randint(100)     #random number in 0-100 range
exampleRandomNumberFloat = np.random.rand()         #random float in 0-1 range
exampleRandomNumverFloatInRange = np.random.uniform(0,100)  #random float in 0-100 range
#print(exampleRandomNumberInt, exampleRandomNumberFloat,exampleRandomNumverFloatInRange)
randomArrayWithInts = np.random.randint(100, size=(2,3))
#print(randomArrayWithInts)
choosingRandomNumberFromSet = np.random.choice([1,3,5,7], size=(4,4))
#print(choosingRandomNumberFromSet)

#Example: creating array with ordered int elements:
intArray2D = np.array([[(x+(y*10)) for x in range(0,10)]for y in range(0,10)])
#print(intArray2D)
#Example: creating a 100 element array with random float values:
floatRandomArray2D = np.random.rand(10,10)
#print(floatRandomArray2D)