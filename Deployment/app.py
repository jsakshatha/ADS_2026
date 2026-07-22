# Import the Flask class to create the web application.
# render_template is used to display HTML pages.
# request is used to get data submitted by the user through forms.
from flask import Flask, render_template, request

# Import pickle to load the trained machine learning model.
import pickle

# Import NumPy for handling arrays, which are required by the ML model.
import numpy as np


# Load the trained machine learning model from the file 'model.pkl'.
# 'rb' means "read in binary mode".
model = pickle.load(open('model.pkl', 'rb'))

# Create a Flask application object.
# __name__ tells Flask where to find the application.
app = Flask(__name__)


# ---------------------- Home Route ----------------------

# The '@app.route("/")' decorator maps the URL '/' (home page)
# to the function immediately below it.
@app.route('/')

# This function is executed whenever the user visits the home page.
def index():

    # render_template() loads the HTML file named 'index.html'
    # from the templates folder and displays it in the browser.
    return render_template('index.html')


# ---------------------- Prediction Route ----------------------

# This route is called when the user submits the form.
# methods=['POST'] means this route accepts only POST requests.
@app.route('/predict', methods=['POST'])

# Function that performs prediction using the ML model.
def predict_placement():

    # Read the CGPA value entered in the HTML form.
    # request.form.get('cgpa') returns the value as a string.
    # float() converts it into a decimal number.
    cgpa = float(request.form.get('cgpa'))

    # Read IQ from the form and convert it to an integer.
    iq = int(request.form.get('iq'))

    # ❌ This line is unnecessary because it overwrites the float value.
    # After this line, cgpa becomes a string again.
    # It should be removed.
    cgpa = request.form.get('cgpa')

    # Read Profile Score from the form and convert it to an integer.
    profile_score = int(request.form.get('profile_score'))


    # ---------------- Prediction ----------------

    # Create a NumPy array with the three input values.
    # reshape(1,3) converts it into one row and three columns.
    # Example:
    # [[8.2, 120, 85]]
    #
    # The model expects a 2D array as input.
    result = model.predict(
        np.array([cgpa, iq, profile_score]).reshape(1, 3)
    )

    # model.predict() returns an array like:
    # [1]  -> Placed
    # [0]  -> Not Placed

    if result[0] == 1:
        result = 'congrats'
    else:
        result = 'not placed'

    # Return the same HTML page and pass the prediction result
    # so that it can be displayed on the webpage.
    return render_template('index.html', result=result)


# ---------------------- Run Flask Server ----------------------

# This block runs only when this file is executed directly.
# It will not run if this file is imported into another Python file.
if __name__ == '__main__':

    # Start the Flask development server.
    # host='0.0.0.0' makes it accessible from other devices on the network.
    # port=8088 means the application runs on port 8088.
    app.run(host='0.0.0.0', port=8088)