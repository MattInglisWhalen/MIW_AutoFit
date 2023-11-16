"""For running MIW's AutoFit on a server"""

# default libraries
import re as regex
import pickle

# external libraries
from flask import Flask, request
from pandas import DataFrame
# internal classes
#from autofit.src.optimizer import Optimizer
from autofit.src.datum1D import Datum1D
from numpy import histogram

########################################################################################################################

app = Flask(__name__)

header = ""

#function_list = []  # list[Datum1D]
optimizer = None    # Optimizer


def make_histogram_data_from_vals(vals):

    data = []

    # bin the values with number of bins = sqrt(count)
    minval, maxval, count = min(vals), max(vals), len(vals)

    if minval - minval // 1 < 2 / count:
        # if it looks like the min and max vals are bolted to an integer, use the integers as a bin boundary
        minval = minval // 1
    if -(-maxval//1) - maxval < 2 / count:
        maxval = maxval // 1

    num_bins = int(count**0.5 // 1)

    bin_edges = [minval + i * (maxval - minval) / num_bins for i in range(num_bins + 1)]
    hist_counts, hist_bounds = histogram(vals, bins=bin_edges)
    for idx, count in enumerate(hist_counts):
        data.append(Datum1D(pos=(hist_bounds[idx + 1] + hist_bounds[idx]) / 2,
                                  val=count,
                                  sigma_pos=(hist_bounds[idx + 1] - hist_bounds[idx]) / 2,
                                  sigma_val=max(count**0.5, 1)
                                  )
                          )
    return data

def parse_string_into_data_list(input_string):

    hist_vals = []
    data = []

    lines = input_string.split('\n')
    for ldx, line in enumerate( input_string.split('\n') ) :
        str_entries = line.split(',')
        print(str_entries)

        while len(str_entries) > 0 and str_entries[-1] == "" :
            str_entries = str_entries[:-1]
        if len(str_entries) == 0 :
            continue

        try :
            entries = [float(str_entry) for str_entry in str_entries]
        except ValueError :
            print("Can't convert string to float")
            return []
        if len(lines) == 1 and len(entries) > 1 :  # 1-line histogram
            hist_vals = entries
        elif len(lines) > 1 and len(entries) == 1:  # 1-column histogram
            hist_vals += float(entries[0])
        elif len(entries) == 2 :
            x, y = entries
            data.append( Datum1D(pos=x,val=y) )
        elif len(entries) == 3 :
            x, y, sigmay = entries
            data.append(Datum1D(pos=x, val=y, sigma_val=sigmay))
        elif len(entries) == 4 :
            x, sigmax, y, sigmay = entries
            data.append(Datum1D(pos=x, sigma_pos=sigmax, val=y, sigma_val=sigmay))
        else :
            return []

    if len(hist_vals) > 0 :
        data = make_histogram_data_from_vals(hist_vals)

    return data

def sanitize(data_str : bytes) -> str :
    """Cleans the user-input raw text to escape nefarious actions"""
    word_str = data_str.decode()
    first_issue = r'\\n'
    word_str = regex.sub(first_issue,'\n',word_str)
    return word_str

def result_string(fit_optimizer):

    best = fit_optimizer.top_model
    best_args = fit_optimizer.top_args
    best_uncs = fit_optimizer.top_uncs
    best_rchisqr = fit_optimizer.top_rchisqr

    if best_rchisqr < 0.001 :
        best_uncs = [0 for _ in best_uncs]

    print_string = ""

    print_string += f"\nBest model is y = {best.name}"
    print_string += f"(x) w/ {best.dof} degrees of freedom and where\n"
    for idx, (par, unc) in enumerate(zip(best_args, best_uncs)):
        print_string += f"  c{idx} =  {par:+.2E}  \u00B1  {unc:.2E}\n"

    print_string += f"\nThis has reduced chi-squared = "
    print_string += f"{best_rchisqr:.2F},"
    print_string += f" and as a tree, this is \n"
    print_string += best.tree_as_string_with_args() + "\n"

    return print_string

def load_frontend():
    """To be used on boot; loads header from html"""
    global header
    with open('MIWs_AutoFit.html', 'r') as f_html:
        header = f_html.read()

def load_model():
    """Load the stored inference model and vocabulary"""
    global optimizer
    # global function_list
    #
    # use_functions_dict = {'cos': True, 'sin': True, 'exp': True, 'log': True, '1/x': True}
    # optimizer = Optimizer(data=None,
    #                       use_functions_dict=use_functions_dict,
    #                       max_functions=4,
    #                       criterion='rchisqr')
    #
    # with open('function_list.pkl', 'rb') as f_fl:
    #     function_list = pickle.load(f_fl)
    with open('optimizer_pickle.pkl', 'rb') as f_m:
        optimizer = pickle.load(f_m)


@app.route('/')
def home_endpoint():
    """Locally: what to show when visiting localhost:80"""
    return header

# ssh -i C:\Users\Matt\Documents\AWS\AWS_DEPLOYED_MODELS.pem ec2-user@18.216.26.152
# scp -i C:\Users\Matt\Documents\AWS\AWS_DEPLOYED_MODELS.pem files ec2-user@18.216.26.152:/home/ec2-user
@app.route('/fit_data', methods=['POST'])
def get_prediction():
    global optimizer
    """Locally: what to show when receiving a post request at localhost:80/fit_data"""
    # Usage:
    # >  curl.exe -X POST localhost:80/fit_data -H 'Content-Type: application/json' -d 'This is a review'

    # Works only for a single sample
    if request.method == 'POST':
        csv_data = sanitize(request.get_data())  # Get data posted as a string
        data = parse_string_into_data_list(csv_data)
        if len(data) == 0 :
            return { 'res_str' : "Incorrect input -- you can't include any letters in your data",
                     'base64_str' : "Not Available" }
        optimizer.set_data_to(data)
        optimizer.top5_rchisqrs = [1e5 for _ in optimizer.top5_rchisqrs]
        optimizer.find_best_model_for_dataset()
        print("\n\n\n\n\nSuccessfully optimized the dataset! Will respond with\n")
        res_str = result_string(optimizer)
        # res_str = "<br>".join(result_string(optimizer).split('\n'))
        print(res_str)
        base64_img = optimizer.make_fit_image(optimizer.top_model) # "This is an image"
        return { 'res_str' : res_str, 'base64_img' : base64_img }
        # transformed_data = vocab.transform([data])  # Transform the input string with the HasheVectorizer
        # sentiment = model.predict_proba(transformed_data)[0,1]  # Runs globally-loaded model on the data
        # return prob_to_html(sentiment) + reasoning_html_from_string(data)
    else :
        raise RuntimeError(f"Can't handle >{request.method}< request method")


if __name__ == '__main__':
    load_frontend()  # load html for the user-facing site
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)

