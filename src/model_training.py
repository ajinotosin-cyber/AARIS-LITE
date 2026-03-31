import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier, IsolationForest

import pickle



df = pd.read_excel("Grade_CS_Students.xlsx")



X = df[["Score", "CourseCoount"]]

y = df["GPA"]



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



reg_model = LinearRegression()

reg_model.fit(X_train,y_train)



df["Goodstanding"] = (df["GPA"] >= 2.0).astype(int)



X_cls = df[["Score"]]

y_cls = df["Goodstanding"]



rf = RandomForestClassifier()

rf.fit(X_cls,y_cls)



iso = IsolationForest(contamination=0.05)

iso.fit(X)



pickle.dump(reg_model,open("../app/regression_model.pkl","wb"))

pickle.dump(rf,open("../app/classifier_model.pkl","wb"))

pickle.dump(iso,open("../app/anomaly_model.pkl","wb"))

