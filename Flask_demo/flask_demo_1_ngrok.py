from flask import Flask # importing the Flask class from the flask module
from pyngrok import ngrok, conf

# Add your auth token here
conf.get_default().auth_token = "2zmUB4Oe98vFPwQ0OddUl9LZ53E_4ZktfeZ499J6hVooyvJgj"

# Now you can open a tunnel
#public_url = ngrok.connect(8080)
#print("ngrok tunnel:", public_url)

app = Flask(__name__) # creating an instance of the Flask class

@app.route("/")
def home():
    return "Hello Welcome all to the MLOPS-3 classes"

if __name__ == "__main__":
    public_url = ngrok.connect(8080)
    print(" * ngrok tunnel available at:", public_url)
    app.run(debug=True, host="0.0.0.0", port=8080)
   # Starts the web server
