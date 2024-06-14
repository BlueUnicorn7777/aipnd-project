import argparse

import tensorflow
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
import os
import json
from PIL import Image
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser(description='Predict the class for flowers dataset using pretrained model' +
                                     ' Baisc Usage python predict.py /path/to/image checkpoint')
      
    parser.add_argument('image_path', type=str, default="",help='path to the input image file')
    parser.add_argument('checkpoint', type=str, default="", help='path to saved checkpoint')
    parser.add_argument('--category_names', default='cat_to_name.json',type=str, 
                        help='specify the class names file')
    parser.add_argument('--top_k', default=5, type=int, help='number of top k classes to print')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='use gpu if available')

    return parser.parse_args()

def validate_args(arg):
    
    if (arg.gpu and torch.cuda.is_available()):
        device = torch.device("cuda")
        print(" *** Model in GPU Mode ***")
    else:
        device = torch.device("cpu")
        print(" *** Model in CPU Mode ***")

    return device , True

def load_checkpoint(arg):

    checkpoint = torch.load(arg.checkpoint)
    model = models.__dict__[checkpoint['model']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
            
    return model

def main():
    arg = arg_parser()
    device , error = validate_args(arg)
    if( not error):
       print("Erros Reported , Please check the input arguments and rerun")
       return
    model = load_checkpoint(arg)
    model.to(device)
    model
    
    with open(arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    classes_lookup = {}
    for key, value in model.class_to_idx.items():
        classes_lookup[str(value)] = key

    predict(arg, model, device,cat_to_name,classes_lookup)



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    original_width, original_height = image.size
    shortest_side = 256
    crop = 224
    if original_width < original_height:
        new_width = shortest_side
        new_height = int(original_height * (shortest_side / original_width))
    else:
        new_height = shortest_side
        new_width = int(original_width * (shortest_side / original_height))

    # Resize the image
    resized_image = image.resize((new_width, new_height))
 
    # Calculate coordinates for the center crop
    left = (new_width - crop) // 2
    top = (new_height - crop) // 2
    right = (new_width + crop) // 2
    bottom = (new_height + crop) // 2

    # Crop the center portion
    cropped_image = resized_image.crop((left, top, right, bottom))

    # convert cropped image to np array  
    np_img = np.array(cropped_image)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean)/std      
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img  

def predict(arg, model, device,cat_to_name,classes_lookup):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
            
    image = Image.open(arg.image_path)
    image = process_image(image)
    #print ( image.shape)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    #print(image.shape)  
    image = torch.FloatTensor(image)
    model, image = model.to(device), image.to(device)
      
    # Perform inference
    with torch.no_grad():
        model.eval()
        predictions = model.forward(image)
        ps = torch.exp(predictions)
        top_p, top_class = ps.topk(arg.top_k, dim=1)  
        
    probs = top_p.cpu().numpy()
    classes = top_class.cpu().numpy()
    names = []
    for row in classes:
        for value in row:
            #print(value)  
            names.append(cat_to_name[classes_lookup[str(value)]])      
    
    print('Top 5 Probabilities : ',  probs)
    print('Top 5 Classes : ', classes)
    print('Class Names :' , names)
           
       
    return probs , classes , names

# Call to main function to run the program
if __name__ == "__main__":
    main()