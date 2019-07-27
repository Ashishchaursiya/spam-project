from sklearn.externals import joblib
model = joblib.load('best.pkl')
wtv= joblib.load('best1.pkl') 

def find(msg="you entered nothing"):
    text=list(msg)
    data= wtv.transform(text)
    status = model.predict(data)
    if status[0]==1:
        return "SPAM"
    else:
        return "NOT SPAM"