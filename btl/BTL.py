import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tkinter.ttk import *
from tkinter import *
from tkinter import messagebox

#Hàm chuẩn hóa dữ liệu
def DataEncoder(dataStr):
	dataConvert = dataStr
	for i, j in enumerate(dataConvert):
		for k in range(0,len(dataConvert[0])):
			if(j[k]=="Female"):j[k] = 1
			elif(j[k]=="Male"):j[k] = 2
			elif(j[k]=="low"):j[k] = 3
			elif(j[k]=="medium"):j[k] = 4
			elif(j[k]=="high"):j[k] = 5
			elif(j[k]=="lazy"):j[k] = 6
			elif(j[k]=="focus"):j[k] = 7
			
	return dataConvert

#Hàm tính tỉ lệ dự đoán đúng
def RateRating(Y_pred):
	countPredictTrue = 0
	for i in range(len(Y_pred)):
		if(Y_pred[i] == Y_test[i]):
			countPredictTrue = countPredictTrue + 1
		rate = countPredictTrue / len(Y_pred)
	return rate

#Đọc dữ liệu từ file
data = pd.read_csv('Result.csv')

#Chia dữ liệu thành 2 phần: X là các thuộc tính, Y là nhãn của dữ liệu
X = DataEncoder(np.array(data[['gender', 'point_process', 'test_score', 'interest', 'attitude']].values))
Y = np.array(data['result'].values)

# #Thuật toán ID3
# #Tiến hành chia dữ liệu thành các phần train, test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = False)
# #Khai báo phương thức tạo cây (Decision Tree) với tiêu chí entropy (ID3), đồng thời tiến hành dựng cây quyết định
TreeID3 = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, Y_train)
# #Tiến hành dự đoán trên tập test
Y_pred_ID3 = TreeID3.predict(X_test)

#Bắt đầu thuật toán PCA
maxRateID3 = 0 #Tỉ lệ dự đoán đúng lớn nhất
for i in range(1,6):
	#Khai báo PCA và số thành phần cần giữ lại
	pca = PCA(n_components = i)
	#Tìm một hệ cơ sở trực chuẩn và loại bỏ những thuộc tính ít quan trọng nhất
	X_Decreased = pca.fit_transform(X)
	#Sau khi dữ liệu đã được giảm kích thước thì tiến hành chia dữ liệu thành các phần train, test
	X_train, X_test, Y_train, Y_test = train_test_split(X_Decreased, Y, test_size = 0.3, shuffle = False)
	#Khai báo phương thức tạo cây (Decision Tree) với tiêu chí entropy (ID3), đồng thời tiến hành dựng cây quyết định
	TreeID3 = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, Y_train)

	#Tiến hành dự đoán trên tập test
	Y_pred_ID3 = TreeID3.predict(X_test)
	# #Tính tỉ lệ dự đoán đúng bằng thư viện
	trained_score = accuracy_score(Y_test, Y_pred_ID3)
	#Điều kiện so sánh để lưu lại các thông tin của PCA và mô hình tốt nhất khi được kết hợp với PCA
	if(RateRating(Y_pred_ID3) > maxRateID3):
		maxRateID3 = RateRating(Y_pred_ID3) #Cập nhật tỉ lệ dự đoán đúng lớn nhất
		numComponentsID3 = i #Cập nhật số thuộc tính tốt nhất
		bestPcaID3 = pca #Cập nhật mô hình PCA tốt nhất
		modelMaxID3 = TreeID3 #Cập nhật mô hình có tỉ lệ dự đoán đúng lớn nhất
		bestPredID3 = Y_pred_ID3
	

#print("Độ chính xác trung bình của 2 lớp: precision = ", precision_score(Y_test, bestPredID3, average='macro'))
# print("Độ thu hồi trung bình của 2 lớp: recall = ", recall_score(Y_test, bestPredID3, average='macro'))
# print("Gía trị trung bình giữa độ chính xác và thu hồi: f1= ", f1_score(Y_test, bestPredID3, average='macro'))
#Hàm dự đoán sử dụng ID3
def PredictWithID3():
	try:
		newData = DataEncoder(np.array([[cbbgender.get(), poinprocess.get(), interest.get(), interest.get(), attitude.get()]])).reshape(1, -1)
		newData_Decreased = bestPcaID3.transform(newData)
		Result = modelMaxID3.predict(newData_Decreased)
		lbPredictId3.configure(text= Result[0])
	except:
		messagebox.showinfo("Cảnh báo", "Vui lòng chọn thông tin để dự đoán")
#Phần thiết kế form
#Tạo form
FORM = Tk()

#Tắt chức năng thay đổi kích thước của form
FORM.resizable(False, False)

#Đặt kích thước cho form
FORM.geometry('470x400')

#Đặt tên cho form
FORM.title("Dự đoán kết quả cuối kì môn học máy")

#Các đối tượng được dùng trong form: Label, Combobox, Button, LabelFrame (Group) 

lbSpace = Label(FORM, text="Dự đoán kết quả:", font=("Arial", 10)).grid(row=0, column=0, pady=5, sticky="e")

lbgender = Label(FORM, text="Giới tính:",).grid(row=1, column=0, pady=5)
cbbgender = Combobox(FORM, state="readonly", values=('Female','Male'))
cbbgender.grid(row=1, column=1, pady=5)

lbpoin = Label(FORM, text="Điểm quá trình:").grid(row=2, column=0, pady=5)
#poinprocess = Entry(lbSpace)
poinprocess = Combobox(FORM, state="readonly", values=('1','2', '3','4','5','6','7','8','9','10'))
poinprocess.grid(row=2, column=1, pady=5)

lbtestscore = Label(FORM, text="Điểm kiểm tra:").grid(row=3, column=0, pady=5)
#testscore = Entry(lbSpace)
testscore = Combobox(FORM, state="readonly", values=('1','2', '3','4','5','6','7','8','9','10'))
testscore.grid(row=3, column=1, pady=5)

lbỉnerest = Label(FORM, text="Sự thích thú:").grid(row=4, column=0, pady=5)
interest = Combobox(FORM, state="readonly", values=('low','medium','high'))
interest.grid(row=4, column=1, pady=5)

lbattitude = Label(FORM, text="Thái độ:").grid(row=5, column=0, pady=5)
attitude = Combobox(FORM, state="readonly", values=('lazy','focus'))
attitude.grid(row=5, column=1, pady=5)

btnPredictId3 = Button(FORM, text="Dự đoán", fg = "white", bg = "black", command=PredictWithID3).grid(row=6, column=0, pady=5)
lbID3 = Label(FORM, text="Kết quả:\n\n\n(Faile/Pass)").grid(row=6, column=1)
lbPredictId3 = Label(FORM, text="---------")
lbPredictId3.grid(row=6, column=1, pady=5)

lb_id3 = Label(FORM, text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
						   +"Accuracy_score: "+str(accuracy_score(Y_test, bestPredID3)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(Y_test, bestPredID3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(Y_test, bestPredID3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(Y_test, bestPredID3, average='macro')*100)+"%").grid(row=8, column=0, pady=5)


#Gọi vòng lặp sự kiện chính để các hành động có thể diễn ra trên màn hình máy tính
FORM.mainloop()
