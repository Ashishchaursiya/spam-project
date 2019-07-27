from flask import Flask, request, render_template, url_for
from result_pro import find
app = Flask(__name__)
@app.route("/main")
def home():
    return render_template("home.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form["msg"]
    status = find(form_data)
    return render_template("page2.html",status=status)

if __name__ == "__main__":
    app.run(debug=True)
