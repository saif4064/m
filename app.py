from flask import Flask,request,jsonify,render_template,url_for
from werkzeug.utils import secure_filename
import os
import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array





app=Flask(__name__)
location=r"static/"
app.config['Upload_Folder']=location





@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','Post'])
def predict():

    f=request.files['file1']
    print(f)
    f.save(os.path.join(app.config['Upload_Folder'],secure_filename(f.filename)))
    c=r'static/'
    print(f.filename)
    d=c+f.filename
    print(d)
    img=(d)
    img1=load_img(img, target_size=(100, 100))
    img2 = img_to_array(img1)
    print(img2)
    img3 = img2.reshape(1, 100, 100, 3)
    model=pickle.load(open("wb.pkl",'rb'))
    result =model.predict_classes(img3)
    s=result[0]
    print(s)
    if(s==1):
        return render_template('index.html',prediction_text="Patient is Covid Postive")
    #print(img)
    if(s==9):
        return render_template('index.html',prediction_text="Patient is Covid  Negative.Patient  is SARS Positive")
    if(s==8):
        return render_template('index.html',prediction_text="Patient is Covid  Negative.Patient  is Pneumonia Postive")
    if(s==0):
        return render_template('index.html',prediction_text="Patient is Covid  Negative.Patient  is ARDS Postive")   
    if(s==10):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")  
    if(s==10):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")  
    
    if(s==5):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    if(s==6):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    if(s==7):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    if(s==4):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    if(s==3):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    if(s==2):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    if(s==1):
        return render_template('index.html',prediction_text="Patient is Covid  Negative")
    
    else:
        return render_template('index.html')
        
    
    
    
   
    return render_template('index.html',prediction_text="OPPS Data Not found")
if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080)













