# Importing Liraries that are required for the Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# Loading DataSet in Mall_data
Mall_data = pd.read_csv("Mall_Customers.csv")


Mall_data.head()



Mall_data.describe()


# Checking the dataset has Null Values or not
Mall_data.isnull().sum()


# Checking type
Mall_data.dtypes


# Droping Not useable Columns just like ID
Mall_data1 = Mall_data.copy()
Mall_data1.drop(["CustomerID"],axis = 1, inplace = True)



Mall_data1.head()



# Ploting 
Columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
n = 0
plt.figure(1,figsize=(20,6))
for i in Columns:
    n+=1
    plt.subplot(1,3,n)
    sb.distplot(Mall_data1[i],bins = 20,)
    plt.title("DistPlot of {}".format(i))
              
plt.show()




sb.countplot(Mall_data['Gender'])
plt.title("Gender Count")
plt.show()




df_18_25 = Mall_data1[(Mall_data1.Age>=18) & (Mall_data1.Age<=25)]  
df_26_35 = Mall_data1[(Mall_data1.Age>=26) & (Mall_data1.Age<=35)]  
df_36_45 = Mall_data1[(Mall_data1.Age>=36) & (Mall_data1.Age<=45)]  
df_46_55 = Mall_data1[(Mall_data1.Age>=46) & (Mall_data1.Age<=55)]  
df_56_above = Mall_data1[Mall_data1.Age>=56]

  
X_part = ["18-25","26-35","36-45","46-55","55+"]
Y_part = [len(df_18_25), len(df_26_35),len(df_36_45),len(df_46_55),len(df_56_above)]

sb.barplot(x = X_part,y = Y_part)
plt.xlabel("Age Group")
plt.ylabel("Number of Customer")
plt.title("Number of Customer and Age")
plt.show()



ai_30 = Mall_data1[Mall_data1["Annual Income (k$)"]<=30]
ai_31_60 = Mall_data1[(Mall_data1["Annual Income (k$)"]>=31) & (Mall_data1["Annual Income (k$)"]<=60)]
ai_61_90 = Mall_data1[(Mall_data1["Annual Income (k$)"]>=61) & (Mall_data1["Annual Income (k$)"]<=90)]
ai_91_120 = Mall_data1[(Mall_data1["Annual Income (k$)"]>=91) & (Mall_data1["Annual Income (k$)"]<=120)]
ai_121_above = Mall_data1[Mall_data1["Annual Income (k$)"]>=121]




x_ai = ["0-30", "31-60", "61-90", "91-120","121-above"]
y_ai = [len(ai_30), len(ai_31_60), len(ai_61_90), len(ai_91_120), len(ai_121_above)]



sb.barplot(x = x_ai,y = y_ai)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customer")
plt.title("Number of Customer and Annual Income")
plt.show()



x = Mall_data1.loc[:,["Age","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k,init = "k-means++")
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.grid()
plt.plot(range(1,11), wcss, marker = "8",linewidth = 2)
plt.xlabel('K Values')
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters = 4)
label = kmeans.fit_predict(x)
print(label)
print(kmeans.cluster_centers_)



plt.scatter(x[:,0],x[:,1],c = kmeans.labels_,cmap = 'rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300,marker="*",color = 'black')
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.title("Age vs Spending Score (1-100)")
plt.show()



x1 = Mall_data1.loc[:,["Annual Income (k$)","Spending Score (1-100)"]].values

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k,init = "k-means++")
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)
plt.grid()
plt.plot(range(1,11), wcss, marker = "8",linewidth = 2)
plt.xlabel('K Values')
plt.ylabel("WCSS")
plt.show()



kmeans1 = KMeans(n_clusters = 5)
label1 = kmeans1.fit_predict(x1)
print(label1)
print(kmeans1.cluster_centers_)




plt.scatter(x1[:,0],x1[:,1],c = kmeans1.labels_,cmap = 'rainbow')
plt.scatter(kmeans1.cluster_centers_[:,0],kmeans1.cluster_centers_[:,1],s = 300,marker="*",color = 'brown')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Age vs Spending Score (1-100)")
plt.show()



res = Mall_data1.copy()

res['Group'] = label1


res.head()


