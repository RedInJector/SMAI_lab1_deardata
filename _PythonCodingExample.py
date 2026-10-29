#imports
import numpy as np

#Variable types
stringVariable = "string"   #Comment
intVariable = 23
floatVariable = 3.9

#Assigning multipele variables
variable1STR, var2float, var3int = "test", 4.4, 12

#print(variable1STR, var2float, var3int)

#2D Array
arraySizeHeight, arraySizeWidth = 10, 10
arrayExample2D = [[0 for x in range(0, arraySizeWidth)]for y in range(0, arraySizeHeight)]

#Method example
def array2DPrint(arrayForPrint):
    """
    Description of the method inside those marks
    This method is meant for printing 2D table/array
     - Input: 2D standard array
    """
    for row in arrayForPrint:
        print(row)

arrayExample2D = [[x+(y*arraySizeWidth) for x in range(0, arraySizeWidth)]for y in range(0, arraySizeHeight)]
#array2DPrint(arrayExample2D)

#String operations
print("The first variable (float) is: " + str(var2float) 
      + ", the second variable (int) is: " + str(var3int))
print("The first variable is: {1}, the second variable is: {0}.".format(var2float, var3int))

#For, while
for i in range(0, 20, 2):
    #print(i)
    pass

iterator = 0
while iterator < 20:
    #print(iterator)
    iterator += 2

stringExample = "A very short string that we will be working with."
#String indexing
print(stringExample[0:5])
print(stringExample[:5])
print(stringExample[5:])
print(stringExample[:-5])
print(stringExample[-5:])

#Operations on array elements
print(arrayExample2D[0])
print(arrayExample2D[1][3])

numpyArrayExample = np.zeros((10,10))
print(numpyArrayExample)
print(numpyArrayExample[0,0])
print(numpyArrayExample[0:3,0:2])

#User input
userVariable = input("Please input int variable or e/E for end: ")
while userVariable not in ['e', 'E']:
    intUserVariable = int(userVariable)
    for varible in range(0, intUserVariable):
        print("test variable {}".format(varible))
    userVariable = input("Please input int variable or e/E for end: ")