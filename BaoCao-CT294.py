
import pandas as pd
import numpy as np
import array as arr
import matplotlib.pyplot as plt
from cProfile import label
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Doc du lieu 
df = pd.read_csv("healthcare-dataset-stroke-data.csv",index_col=0)              
df.columns = ["Gender", "Age", "Hypertension", "Heart Disease", "Ever Married", "Work Type", "Residence Type", "Avg. Glucose Level", "BMI", "Smoking Status", "Stroke"]

# xoa cac đối tương chứa NAN tai cot BMI
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

#chuyen cac thuoc tinh thanh du lieu so voi dummies 
df["Hypertension"].replace([0,1], ["No","Yes"], inplace=True)
df["Heart Disease"].replace([0,1], ["No","Yes"], inplace=True)

# tao 1 mang df2 de luu cac thuoc tinh không phai so sang kieu du lieu so
df2 = df[["Gender","Age","Hypertension","Heart Disease","Ever Married","Work Type","Residence Type","Avg. Glucose Level","BMI", "Smoking Status","Stroke"]]
gender = pd.get_dummies(df2["Gender"], drop_first=True)
hypertension = pd.get_dummies(df2["Hypertension"], drop_first=True, prefix="HT")
heartdisease = pd.get_dummies(df2["Heart Disease"], drop_first=True, prefix="HD")
evermarried = pd.get_dummies(df2["Ever Married"], drop_first=True, prefix="EM")
worktype = pd.get_dummies(df2["Work Type"], drop_first=True)
residence = pd.get_dummies(df2["Residence Type"],drop_first=True)
smoking = pd.get_dummies(df2["Smoking Status"], drop_first=True)

# tao 1 mang df3 (de chua các thuộc tính đa được chuyển đổi) tu mang df2 
df3 = pd.concat([df2,gender,hypertension,heartdisease,evermarried,worktype,residence,smoking], axis=1, join='outer', ignore_index=False)
df3.drop(["Gender","Hypertension","Heart Disease","Ever Married","Work Type", "Residence Type","Smoking Status"], axis=1, inplace=True)

# gán nhãn cho các thuộc tính mới vừa chuyển đổi
df4 = df3.reindex(labels=["Age","Male","HT_Yes","HD_Yes","EM_Yes","Never_worked","Private","Self-employed","children","BMI","Urban","Avg. Glucose Level","formerly smoked", "never smoked", "smokes","Stroke"], axis=1)

# khai bao thuộc tính và nhẵn
X = df4[["Age","Male","HT_Yes","HD_Yes","EM_Yes","Never_worked","Private","Self-employed","children","BMI","Avg. Glucose Level","formerly smoked", "never smoked", "smokes"]]
y = df4["Stroke"]

kf = model_selection.KFold(n_splits = 10, random_state=42, shuffle= True)
i = 1
arrNB = []
arrDCT = []
arrKNN = []
# chuẩn hóa số liệu của thuộc tính
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.transform(X)

# huấn luyện mô hinh với K-fold
for train_id, test_id in kf.split(X):
    print("===============================================")
    print("Lap lan thu: ",i)
    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
    y_train, y_test = y.iloc[train_id], y.iloc[test_id]

    # KNN 
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=124)
    Mohinh_KNN.fit(X_train, y_train)
    y_pred_KNN = Mohinh_KNN.predict(X_test)
    KNN = accuracy_score(y_test, y_pred_KNN) * 100
    
    # Xay dung mo hinh Gaussian naive bayes model
    clf = GaussianNB()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    Bayes = accuracy_score(y_test, y_pred) * 100
    
    
    # số cây để huấn luyện mô hình
    seed = 42 
    num_trees = 100
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=10, max_depth=10, min_samples_leaf=10)
    clf_gini.fit(X_train, y_train)
    y_pred_DCT= clf_gini.predict(X_test)
    DCT = accuracy_score(y_test, y_pred_DCT) * 100
    
    
    # in kết quả dự đoán của các thuật toán
    arrNB.append(Bayes)
    arrDCT.append(DCT)
    arrKNN.append(KNN)
    print("Accuracy Score Gaussian = ", Bayes)
    print("Accuracy Score KNN = ", KNN)
    print("Accuracy Score DecisionTreeClassifier = ", DCT)
    i += 1
k_range = range(1,11)
plt.plot(k_range,arrKNN)
plt.plot(k_range,arrDCT)
plt.plot(k_range,arrNB)
plt.ylabel('Testing mean accuracy')
plt.xlabel('Number of iterations : K')
plt.legend(labels=['KNN','Desicion tree','Naive Bayes'])
plt.show()