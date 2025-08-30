from flask import Flask # importing the Flask class from the flask module

app = Flask(__name__) # creating an instance of the Flask class

@app.route("/")
def home():
    return "Hello Welcome all to the MLOPS classes"

if __name__ == "__main__":
    app.run(debug=True)   # Starts the web server
