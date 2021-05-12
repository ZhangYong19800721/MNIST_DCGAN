import argparse
import pickle
import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
import random
import DATASET_MNIST
import tools

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device
    ##########################################################################
    ## load the AI model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_G_File", type=str, help="the path for G model")

    args = parser.parse_args()

    modelG_file = open(args.model_G_File, "rb")  # open the model file
    modelG = pickle.load(modelG_file)  # load the model file
    if isinstance(modelG, nn.DataParallel):
        modelG = modelG.module
    modelG.to('cpu')  # push model to GPU device
    modelG.eval()  # set the model to evaluation mode, (the dropout layer need this)
    modelG_file.close()  # close the model file

    ## set the data set
    dataset = DATASET_MNIST.TRAINSET("./data/mnist.mat")
    dataLoader = DATASET_MNIST.DATASET_LOADER(dataset, minibatch_size=100)
    minibatch_count = len(dataLoader)

    # show some data samples
    print("Show some images ...., press ENTER to continue. ")
    n = random.randint(0, len(dataLoader))
    minibatch = dataLoader[n]
    tools.showNineGrid_3x3(minibatch['image'][0], minibatch['image'][1], minibatch['image'][2],
                           minibatch['image'][3], minibatch['image'][4], minibatch['image'][5],
                           minibatch['image'][6], minibatch['image'][7], minibatch['image'][8])

    with torch.no_grad():
        noise = torch.randn((9, 100, 1, 1))
        images = modelG(noise)

    tools.showNineGrid_3x3(images[0], images[1], images[2],
                           images[3], images[4], images[5],
                           images[6], images[7], images[8])

