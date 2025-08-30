from flask import Flask # importing the Flask class from the flask module

app = Flask(__name__) # creating an instance of the Flask class

@app.route("/")
def home():
    return "Hello Welcome all to the MLOPS classes"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
   # Starts the web server
