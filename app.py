from flask import Flask,render_template , request
import pickle
# cv is count tokenizer
#clf is classifier
tokenizer = pickle.load(open("models/cv.pkl","rb")) #rb = read binary because it is a binary file
model = pickle.load(open("models/clf.pkl","rb"))
app = Flask(__name__)
@app.route("/")

def home():

    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    if request.method =="POST":
        email = request.form.get("email-content")
    # so this tokenized_email is X
    tokenized_email =tokenizer.transform([email])#coming from sklearn
    predictions = model.predict(tokenized_email)
    predictions =1 if predictions==1 else -1

    return render_template("index.html",predictions=predictions,email=email)
if __name__ =="__main__":
    app.run(debug= True) # if we need to changes accordingly 
                         #we need to put debug=true