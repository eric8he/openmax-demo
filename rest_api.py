# standard library imports
import os
import operator
import pickle

# third party imports
from flask import Flask, flash, redirect, request, Response
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import json

# define constants
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FLASK_PORT = 443

CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASSES = len(CIFAR10_LABELS)
TOPK_TO_GET = 5

INPUT_SIZE = 32
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]
TRANSFORM_NORM = transforms.Compose([transforms.ToTensor(), transforms.Resize(INPUT_SIZE),transforms.CenterCrop(INPUT_SIZE), transforms.Normalize(MEAN, STD)])

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
SOFTMAX = nn.Softmax(0)

# load MAVs and models from file
with open('cifar10_dists.pickle', 'rb') as handle:
    dists = pickle.load(handle)

# create and initialize Flask instance
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

# function to prepare image for processing
def preprocess_image(path):
    # open image and ensure proper format
    img = Image.open(path).convert("RGB")

    # get normalized image
    img_normalized = TRANSFORM_NORM(img).float()
    img_normalized = img_normalized.unsqueeze_(0)

    return img_normalized

def eval_image(img_normalized, model:nn.Module):
    # run processed image through NN
    with torch.no_grad():
        # evaluate model on image
        model.eval()
        t = model(img_normalized)[0]
        return t

# processing fucntion to get beautified data for SoftMax
def get_results_naive(t, label_arr, classes, top_K):
    # scale input into probabilities
    output = SOFTMAX(t).tolist()

    # create tuple mapping nums to labels
    d = [(output[i], label_arr[i]) for i in range(classes)]

    # sort, reverse, clip, and return
    return [(x[1], x[0] * 100) for x in sorted(d, reverse=True)[:top_K]]

# processing fucntion to get beautified data for OpenMax
# this closely follows Algorithm 2 as outlined in the paper
def get_results_openmax(t, label_arr, classes, top_K):
    # load in MAVs
    means = torch.Tensor(dists[0])

    # add min value to make AV nonnegative
    t = torch.add(t, torch.min(t))

    # get indices for classes, to use in computation later

    # I also spent 4 HOURS trying to debug without realizing
    # that argsort sorts in ascending order by default, FML
    s = torch.argsort(t, dim=0, descending=True).tolist() # line 1 in paper

    # compute multipliers for the top K classes 
    # (w_s(i) in paper)
    # if not in top K, multiplier is 1
    mult = [1] * classes # line 1 in paper

    for i in range(top_K):
        # what the fuck (line 3 in paper)
        mult[s[i]] = 1 - (top_K-i-1)/top_K * (1-dists[1][s[i]].cdf(torch.norm(t - means[s[i]])))

    # elementwise multiply the AV and multipliers
    openmax_res = [mult[i] * t[i] for i in range(classes)] # line 5 in paper

    # calculate virtual activation level for the unknown class
    unknown_prob = sum([t[i] * (1 - mult[i]) for i in range(classes)]) # line 6 in paper

    # combine the activation vector, and softmax to turn them into probabilities
    openmax_res = [unknown_prob] + openmax_res
    new_res = SOFTMAX(torch.Tensor(openmax_res)).tolist() # line 7 in paper

    # add class of unknowns to label array
    new_arr = ["NONE"] + label_arr

    # pair labels with revised confidence scores
    res = [[new_arr[i], new_res[i]*100] for i in range(len(new_res))]

    # sort list by second element, select top K elements
    return sorted(res, key=operator.itemgetter(1), reverse=True)[:top_K+1]


# helper for enforcing proper image paths
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # make sure file is attached and is valid
        # check if request is valid
        if 'file' not in request.files:
            flash('No file part')
            print('No file part')
            return redirect(request.url)
        
        # check if file is attached
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(request.url)
        
        # check if file is of an allowed filetype 
        if file and not allowed_file(file.filename):
            flash('Unreadable file type (this only supports PNG, JPG/JPEG, and GIF formats)')
            print('Unreadable file type (this only supports PNG, JPG/JPEG, and GIF formats)')
            return redirect(request.url)
        
        # get file and save in folder
        if file and allowed_file(file.filename):
            pth = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(pth)

        # preprocess and then process image
        img = preprocess_image(pth)
        t = eval_image(img, model)

        # get naive and OpenMax classification results
        naive = get_results_naive(t, CIFAR10_LABELS, CLASSES, TOPK_TO_GET)
        opened = get_results_openmax(t, CIFAR10_LABELS, CLASSES, TOPK_TO_GET)
        
        # only attach OpenMax if the slider switch is flipped on
        processed = ([] if request.form.get('open') == None else [opened]) + [naive]
        print(processed)

        resp = Response(json.dumps({"ok": True, "vals": processed, "img": pth}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    
    resp = Response("success")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/', methods=['OPTIONS'])
def options():
    resp = Response("success")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=FLASK_PORT, debug = True, ssl_context='adhoc')
