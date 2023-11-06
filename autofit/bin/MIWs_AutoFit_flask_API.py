"""For running MIW's AutoFit on a server"""

# default libraries
import re as regex
import pickle

# external libraries
from flask import Flask, request

# internal classes
from autofit.src.data_handler import DataHandler


########################################################################################################################

app = Flask(__name__)

header = ""

def sanitize(data_str : bytes) -> str :
    """Cleans the user-input raw text to escape nefarious actions"""
    cleaned_str = ""

    word_str = data_str.decode()
    first_issue = r'\\n'
    second_issue = "\\\\\""
    word_str = regex.sub(first_issue,' <br> ',word_str)
    word_str = regex.sub(second_issue,'&lsquo;',word_str)

    for paragraph in word_str.split('\n') :
        for word in paragraph.split() :
            skip_next = False
            shifted_word = word + ' '
            for char, next_char in zip(word,shifted_word[1:]) :
                if skip_next :
                    skip_next = False
                    continue
                if char == '<' :
                    cleaned_str += ' &lt; '
                elif char == '>' :
                    cleaned_str += ' &gt; '
                elif char == '/':
                    cleaned_str += ' &sol; '
                elif char == '\\' :
                    if next_char == 'n' :
                        cleaned_str += " <br> "
                        skip_next = True
                        continue
                    elif next_char == '"' :
                        cleaned_str += "&ldquo;"
                        skip_next = True
                        continue
                    cleaned_str += ' &bsol; '
                elif char in ['\n','\r']:
                    cleaned_str += ' <br> '
                elif char == '"' :
                    cleaned_str += '&lsquo;'
                else :
                    cleaned_str += char
            cleaned_str += ' '
        cleaned_str += "<br>"

    return cleaned_str

def load_frontend():
    """To be used on boot; loads header from html"""
    global header
    with open('MIWs_AutoFit.html', 'r') as f_html:
        header = f_html.read()

def load_model():
    """Load the stored inference model and vocabulary"""
    global vocab
    global model

    with open('vocabulary_hashed.pkl', 'rb') as f_v:
        vocab = pickle.load(f_v)
    with open('sentiment_inference_model_hashed.pkl', 'rb') as f_m:
        model = pickle.load(f_m)


@app.route('/')
def home_endpoint():
    """Locally: what to show when visiting localhost:80"""
    return header

# ssh -i C:\Users\Matt\Documents\AWS\AWS_DEPLOYED_MODELS.pem ec2-user@18.216.26.152
# scp -i C:\Users\Matt\Documents\AWS\AWS_DEPLOYED_MODELS.pem files ec2-user@18.216.26.152:/home/ec2-user
@app.route('/fit_data', methods=['POST'])
def get_prediction():
    """Locally: what to show when receiving a post request at localhost:80/predict"""
    # Usage:
    # >  curl.exe -X POST localhost:80/predict -H 'Content-Type: application/json' -d 'This is a review'

    # Works only for a single sample
    if request.method == 'POST':
        data = sanitize(request.get_data())  # Get data posted as a string
        data_handler = DataHandler(data)
        transformed_data = vocab.transform([data])  # Transform the input string with the HasheVectorizer
        sentiment = model.predict_proba(transformed_data)[0,1]  # Runs globally-loaded model on the data
        return prob_to_html(sentiment) + reasoning_html_from_string(data)
    else :
        raise RuntimeError(f"Can't handle >{request.method}< request method")


if __name__ == '__main__':
    load_frontend()  # load html for the user-facing site
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)

