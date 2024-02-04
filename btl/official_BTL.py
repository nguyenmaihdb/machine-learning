import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold


df = pd.read_csv('Result.csv')     
data1 = np.array(df[['gender', 'point_process', 'test_score', 'interest', 'attitude', 'result']].values) #lấy ra các cột có tên đc nêu


#hàm chuẩn hóa dữ liệu(đưa dữ liệu dạng chuỗi về số thì máy mới học đc)
def data_encoder(data):
    for i, j in enumerate(data):
        for k in range(0, 6):
            if (j[k] == "Female"):
                j[k] = 0
            elif (j[k] == "Male"):
                j[k] = 1
            elif (j[k] == "low"):
                j[k] = 2
            elif (j[k] == "medium"):
                j[k] = 3
            elif (j[k] == "high"):
                j[k] = 4
            elif (j[k] == "focus"):
                j[k] = 5
            elif (j[k] == "lazy"):
                j[k] = 6               
    return data

def data_encoder_y(data):
    for k in range(0, len(data)):
        if (data[k] == "Faile"):
            data[k] = 7
        elif (data[k] == "Pass"):
            data[k] = 8               
    return data

#chuẩn hóa dữ liệu trong data1
data = data_encoder(data1)

#chia dữ liệu sau khi chuẩn hóa thành 2 phần: train 70%, test = 30%
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = False)


#chia dữ liệu thành 5 phần(k=5), random_state=None: không làm gì cả
kf = KFold(n_splits=5,random_state=None)

max=0
stt = 1 #biến đếm ban đầu được khởi tạo = 1
#chia dữ liệu dt_Train thành 5 phần
#duyệt trên từng mô hình chia được để tìm ra mô hình tốt(sự sai lệch nhỏ nhất)
for train_index,test_index in kf.split(dt_Train):
    X_train,X_test = dt_Train[train_index,:5],dt_Train[test_index,:5] 
    y_train,y_test = dt_Train[train_index,5:6],dt_Train[test_index,5:6]


    id3 = tree.DecisionTreeClassifier(criterion='entropy')
    id3.fit(X_train,y_train)         
    Y_pred_train = id3.predict(X_train)
    Y_pred_test = id3.predict(X_test)

    print("\n\n- train accuracy:", metrics.accuracy_score(y_train,Y_pred_train))
    print("\n\n validation accuracy:", metrics.accuracy_score(y_test,Y_pred_test))
    #tổng tỉ lệ dự đoán đúng trên 2 mô hình càng lớn càng tốt 
    sum = metrics.accuracy_score(y_train,Y_pred_train) + metrics.accuracy_score(y_test,Y_pred_test)
    
    if(sum>max):    #nếu mô hình có sum>max --> là mô hình tốt
        stt_model_best = stt
        max = sum   #lưu lại tổng tỉ lệ dự đoán cao nhất của mô hình hiện tại
        modelmax_id3 = id3.fit(X_train,y_train)  #huấn luyện mô hình tốt nhất hiện tìm được
        data_train_best = np.concatenate ((X_train, y_train), axis = 1) #lưu lại tập train của mô hình tốt nhất
        data_test_best = np.concatenate ((X_test, y_test), axis = 1) #lưu lại tập test của mô hình tốt nhất
    stt = stt + 1 #sau đó tăng stt lên 1 để duyệt mô hình tiếp theo 

print ("\n - Mô hình tốt nhất là mô hình thứ: ", stt_model_best, "\n- Tỉ lệ đúng của mô hình là: ", max)
#print ("\nTập train:\n", data_train_best)
#print ("\nTập test:\n", data_test_best)
#thực hiện tính giá trị dự đoán trên dữ liệu dt_Test(chiếm 30% data đã chia từ đầu)
y_predict = modelmax_id3.predict(dt_Test[:,:5])

#giá trị thực tế của dữ liệu dt_Test(chiếm 30% data đã chia từ đầu)
y = dt_Test[:,5]

#đánh giá mô hình dự trên các độ đo
print("precision = ",metrics.precision_score(y, y_predict, average='macro'))  #độ chính xác
print("recall = ",metrics.recall_score(y, y_predict, average='micro')) #độ thu hồi
print("f1 = ",metrics.f1_score(y, y_predict, average='weighted'))  #giá trị trung bình giữa độ chính xác và độ thu hồi
print("accuracy = ",metrics.accuracy_score(y, y_predict))#*100)+"%"+'\n'






