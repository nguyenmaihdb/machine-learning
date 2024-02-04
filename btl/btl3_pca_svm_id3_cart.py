from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score

df =  pd.read_csv('iBuy12.csv')

X = np.array(df[['GENDER','FAMILY_INFLUENCE','FRIENDS_INFLUENCE','APPLE_ECOSYSTEM'
                 ,'EMPLOYMENT','AGE','THIRD_PARTY']].values)    
y = np.array(df['IPHONE_12_PURCHASE'])

max_svc = 0
max_id3 = 0
max_cart = 0
for j in range(1,8):
    pca = decomposition.PCA(n_components = j)
    #print (pca)
    pca.fit(X)

    Xbar = pca.transform(X)

    X_Train, X_Test,y_Train,y_Test = train_test_split(Xbar,y, test_size=0.3 , shuffle = True)
    #print(X_Train)
#SVM
    svc = SVC(kernel = 'linear')
    svc = svc.fit(X_Train,y_Train)
    y_svc = svc.predict(X_Test)
    dem = 0
    for i in range(0,len(y_svc)):
        if(y_svc[i] == y_Test[i]):
            dem = dem +1
    rate_svc = dem/len(y_svc) #tỷ lệ % dự đoán đúng
    #print ("Tỷ lệ dự đoán đúng là :",rate_svc,"với n_component:", j)
    if(rate_svc > max_svc ):
        num_pca_svc = j
        pca_svc_best = pca
        max_svc = rate_svc
        model_svc_max = svc
    #print ("Tỷ lệ dự đoán đúng là :",max_svc)
#print("Tỷ lệ dự đoán max",max_svc,"với n_component:",num_pca)

#ID3
    id3 = DecisionTreeClassifier(criterion = 'entropy')
    id3 = id3.fit(X_Train,y_Train)
    y_id3 = id3.predict(X_Test)
    dem  = 0
    for i in range (0,len(y_id3)):
        if (y_id3[i] == y_Test[i]):
            dem = dem + 1
    rate_id3 = dem/len(y_id3) #tỷ lệ % dự đoán đúng
    #print ("Tỷ lệ dự đoán đúng là :",rate_svc,"với n_component:", j)
    if(rate_id3 > max_id3):
        num_pca_id3 = j
        pca_id3_best = pca
        max_id3 = rate_id3
        model_id3_max = id3
    #print("max",max_id3,"d",num_pca)
#print("Tỷ lệ dự đoán max",max_id3,"với n_component:",num_pca_id3)
#CART
    cart = DecisionTreeClassifier(criterion = 'gini')
    cart = cart.fit(X_Train,y_Train)
    y_cart = cart.predict(X_Test)   
    dem  = 0
    for i in range (0,len(y_cart)):
        if (y_cart[i] == y_Test[i]):
            dem = dem + 1
    rate_cart = dem/len(y_id3)#tỷ lệ % dự đoán đúng
    #print ("Tỷ lệ dự đoán đúng là :",rate_svc,"với n_component:", j)
    if(rate_cart > max_cart):
        num_pca_cart = j
        pca_cart_best = pca
        max_cart = rate_cart
        model_cart_max = cart
    #print("max",max_id3,"d",num_pca)
#print("Tỷ lệ dự đoán max",max_id3,"với n_component:",num_pca_cart))

#form
form = Tk()
form.title("Dự đoán quyết định mua Iphone:")
form.geometry("1500x1000")

lable_ten = Label(form, text = "Nhập các thông tin liên quan:", font=("Arial Bold", 10), fg="Green")
lable_ten.grid(row = 1, column = 2, padx = 20, pady = 10)

lable_gender = Label(form, text = " Giới Tính :")
lable_gender.grid(row = 2, column = 2, pady = 10)
textbox_gender = Entry(form)
textbox_gender.grid(row = 2, column = 3)

lable_familyifl = Label(form, text = "Tác động từ gia đình:")
lable_familyifl.grid(row = 3, column = 2, pady = 10)
textbox_familyifl = Entry(form)
textbox_familyifl.grid(row = 3, column = 3)

lable_friendifl = Label(form, text = "Tác động từ bạn bè:")
lable_friendifl.grid(row = 4, column = 2,pady = 10)
textbox_friendifl = Entry(form)
textbox_friendifl.grid(row = 4, column = 3)

lable_apecosytem = Label(form, text = "Tác động từ sự đa dạng của Apple:")
lable_apecosytem.grid(row = 5, column = 2, pady = 10)
textbox_apecosytem = Entry(form)
textbox_apecosytem.grid(row = 5, column = 3)

lable_employee = Label(form, text = "Tác động từ nhân viên bán hàng:")
lable_employee.grid(row = 6, column = 2, pady = 10 )
textbox_employee = Entry(form)
textbox_employee.grid(row = 6, column = 3)

lable_age = Label(form, text = "Độ tuổi:") 
lable_age.grid(row = 7, column = 2, pady = 10 )
textbox_age = Entry(form)
textbox_age.grid(row = 7, column = 3)

lable_thirdparty = Label(form, text = "Qua trung gian :")
lable_thirdparty.grid(row = 8, column = 2, pady = 10 )
textbox_thirdparty = Entry(form)
textbox_thirdparty.grid(row = 8, column = 3)

#CART 
lbl1 = Label(form)
lbl1.grid(column=1, row=9)
lbl1.configure(text="Tỉ lệ dự đoán đúng của CART: "+'\n'
                           +"Precision: "+str(precision_score(y_Test, y_cart, average='macro')*100,)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_Test, y_cart, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_Test, y_cart, average='macro')*100)+"%"+'\n')
def dudoancart():
    gender = textbox_gender.get()
    familyifl = textbox_familyifl.get()
    friendifl = textbox_friendifl.get()
    apecosytem = textbox_apecosytem.get()
    employee =textbox_employee.get()
    age =textbox_age.get()
    thirdparty =textbox_thirdparty.get()
    if((gender == '') or (familyifl == '') or (friendifl == '') or (apecosytem == '')
       or (employee == '')or (age == '')or (thirdparty == '')):
        messagebox.showinfo("Thông báo nhập", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([gender,familyifl,friendifl,apecosytem,employee,age,thirdparty]).reshape(1, -1)
        X_dudoan = pca_cart_best.transform(X_dudoan)
        y_kqua = model_cart_max.predict(X_dudoan)
        lbl.configure(text= y_kqua)
button_cart = Button(form, text = 'Kết quả dự đoán theo CART', command = dudoancart,fg ="blue")
button_cart.grid(row = 10, column = 1, pady = 20)
lbl = Label(form, text="...")
lbl.grid(column=2, row=10)
def khanangcart():
    count = (max_cart)*100
    lbl1.configure(text= str(count)+"%")
button_cart1 = Button(form, text = 'Tỷ lệ dự đoán đúng với '+ str(num_pca_cart)+' thành phần chính', command = khanangcart,fg ="blue")
button_cart1.grid(row = 11, column = 1, pady = 20)
lbl1 = Label(form, text="...")
lbl1.grid(column=2, row=11,padx = 50)

#ID3
lbl3 = Label(form)
lbl3.grid(column=3, row=9)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_Test, y_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_Test, y_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_Test, y_id3, average='macro')*100)+"%"+'\n')
def dudoanid3():
    gender = textbox_gender.get()
    familyifl = textbox_familyifl.get()
    friendifl = textbox_friendifl.get()
    apecosytem = textbox_apecosytem.get()
    employee =textbox_employee.get()
    age =textbox_age.get()
    thirdparty =textbox_thirdparty.get()
    if((gender == '') or (familyifl == '') or (friendifl == '') or (apecosytem == '')
       or (employee == '')or (age == '')or (thirdparty == '')):
        messagebox.showinfo("Thông báo nhập", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([gender,familyifl,friendifl,apecosytem,employee,age,thirdparty]).reshape(1, -1)
        X_dudoan = pca_id3_best.transform(X_dudoan)
        y_kqua = model_id3_max.predict(X_dudoan)
        lbl2.configure(text= y_kqua)
    
button_id3 = Button(form, text = 'Kết quả dự đoán theo ID3', command = dudoanid3,fg ="blue")
button_id3.grid(row = 10, column = 3, pady = 20)
lbl2 = Label(form, text="...")
lbl2.grid(column=4, row=10,padx = 50)

def khanangid3():
    count = (max_id3)*100
    lbl3.configure(text= str(count)+"%")
button_id31 = Button(form,text = 'Tỷ lệ dự đoán đúng với '+ str(num_pca_id3)+' thành phần chính', command = khanangid3,fg ="blue")
button_id31.grid(row = 11, column = 3, padx = 20)
lbl3 = Label(form, text="...")
lbl3.grid(column=4, row=11,padx = 50)


#SVM
lbl5 = Label(form)
lbl5.grid(column = 5, row = 9)
lbl5.configure(text="Tỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_Test, y_svc, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_Test, y_svc, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_Test, y_svc, average='macro')*100)+"%"+'\n')
def dudoansvm():
    gender = textbox_gender.get()
    familyifl = textbox_familyifl.get()
    friendifl = textbox_friendifl.get()
    apecosytem = textbox_apecosytem.get()
    employee =textbox_employee.get()
    age =textbox_age.get()
    thirdparty =textbox_thirdparty.get()
    if((gender == '') or (familyifl == '') or (friendifl == '') or (apecosytem == '')
       or (employee == '')or (age == '')or (thirdparty == '')):
        messagebox.showinfo("Thông báo nhập", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([gender,familyifl,friendifl,apecosytem,employee,age,thirdparty]).reshape(1, -1)
        X_dudoan = pca_svc_best.transform(X_dudoan)
        y_kqua = model_svc_max.predict(X_dudoan)
        lbl4.configure(text= y_kqua)
    
button_svm = Button(form, text = 'Kết quả dự đoán theo SVM',command = dudoansvm,fg ="blue")
button_svm.grid(row = 10, column = 5, pady = 20)
lbl4 = Label(form, text="...")
lbl4.grid(column=6, row=10,padx = 50)

def khanangsvm():
    count = (max_svc)*100
    count = (dem/len(y_svc))*100
    lbl5.configure(text= str(count)+"%")
button_svml = Button(form, text = 'Tỷ lệ dự đoán đúng với '+ str(num_pca_svc)+' thành phần chính',command = khanangsvm,fg ="blue")
button_svml.grid(row = 11, column = 5, padx = 20)
lbl5 = Label(form, text="...")
lbl5.grid(column=6, row=11,padx = 50)
#nhập mẫu [1,0,1,0,0,44,1]
form.mainloop()
