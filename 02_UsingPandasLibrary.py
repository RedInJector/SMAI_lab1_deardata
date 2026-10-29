import numpy as np
import pandas as pd


mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}   #example dataset

myvar = pd.DataFrame(mydataset) #creating a data frame

#Pandas series is a one-dimentional array holding data of any type
#Defining series:
a = [1, 7, 2]
myvar = pd.Series(a)
#print(myvar)
myvar = pd.Series(a, index = ["x", "y", "z"]) #adding indexes
#print(myvar)
#print(myvar["y"])               #returing a value from label

calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)     #creating a series from a dictionary
#print(myvar)

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
#df = pd.DataFrame(data)      #creating a Data Frame from two Series
#print(df)

#print(df.loc[0])                #refer to the row index
#print(df.loc[[0, 1]])           #quse a list of indexes
#df = pd.DataFrame(data, index = ["day1", "day2", "day3"])   #adding names/labels
#print(df)
#print(df.loc["day2"])           #refering to the index/label

df = pd.read_csv('data.csv')    #read csv file and load it into Data Frame
#print(df.to_string()) 
#print(df)                      #without the conversion, but has restrictions
#print(pd.options.display.max_rows)  #checking systems max rows to return
#If the data frame is larger than maximum, the system will return just 
#the first and last 5 rows.
#pd.options.display.max_rows = 300  #changing the max display rows value
#print(df)

#df = pd.read_json('data.json')      #reading json file
#print(df.to_string())

#print(df.head(10))          #printing the first 10 rows
#print(df.tail())            #printing last 5 rows

#DATA CLEANING
#new_df = df.dropna()        #return new Data Frame with no empty cells
#df.dropna(inplace = True)   #working on the original DF
#df.fillna(130, inplace = True)  #filling NULL values
#df["Calories"].fillna(130, inplace = True) #doing the same for specific column

#Different approaches
x = df["Calories"].mean()                   #Calculating mean value  
#df["Calories"].fillna(x, inplace = True)    #replacing missing values with mean
x = df["Calories"].median() 
#df["Calories"].fillna(x, inplace = True)     # same for median
x = df["Calories"].mode()[0]
df["Calories"].fillna(x, inplace = True) #and mode (value that appears most frequently)

#Dealing with wrong data
#print(df.to_string())
#Example: in some rows we have significantly higher values
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120     #Changing the value to the chosen one


"""for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)      #alternatively, removing the row"""

#print(df.to_string())

#Dealing with duplicates
#print(df.duplicated().to_string())  #True for each row with a duplicate
df.drop_duplicates(inplace = True) #Dropping the duplicated rows

#print(df.corr())       #Show correlations between columns