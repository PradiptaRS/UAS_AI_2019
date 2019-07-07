from sklearn import tree
import pandas as pd

#filepath = "C:/Users/user/Downloads/lung_cancer_examples.csv"
#sep = ","
#delimiter = None

df = pd.read_csv("C:/Users/user/Downloads/lung_cancer_examples.csv")

a = df[["Age","Smokes","AreaQ","Alkhol"]]
b = df["Result"]

clf = tree.DecisionTreeClassifier()
clf.fit(a,b)
w = input("Please enter age : ")
x = input("Please enter smokes : ")
y = input("Please enter AreaQ : ")
z = input("Please enter Alkohol : ")
print ("\n============================\n")
print("Prediction Result : ")
if clf.predict([[w,x,y,z]]) == 1:
    print("possibility of lung cancer")
elif clf.predict([[w,x,y,z]]) == 0:
    print ("possibility of not getting lung cancer")