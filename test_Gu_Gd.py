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
    parser.add_argument("--model_Gu_File", type=str, help="the path for Gu model")
    parser.add_argument("--model_Gd_File", type=str, help="the path for Gd model")

    args = parser.parse_args()

    modelGu_file = open(args.model_Gu_File, "rb")  # open the model file
    modelGu = pickle.load(modelGu_file)  # load the model file
    if isinstance(modelGu, nn.DataParallel):
        modelGu = modelGu.module
    modelGu.to('cpu')  # push model to GPU device
    modelGu.eval()  # set the model to evaluation mode, (the dropout layer need this)
    modelGu_file.close()  # close the model file

    modelGd_file = open(args.model_Gd_File, "rb")  # open the model file
    modelGd = pickle.load(modelGd_file)  # load the model file
    if isinstance(modelGd, nn.DataParallel):
        modelGd = modelGd.module
    modelGd.to('cpu')  # push model to GPU device
    modelGd.eval()  # set the model to evaluation mode, (the dropout layer need this)
    modelGd_file.close()  # close the model file

    ## set the data set
    dataset = DATASET_MNIST.TRAINSET("./data/mnist.mat")
    dataLoader = DATASET_MNIST.DATASET_LOADER(dataset, minibatch_size=100)
    minibatch_count = len(dataLoader)

    # show some data samples
    print("Show some images ...., press ENTER to continue. ")
    n = random.randint(0, len(dataLoader)-1)
    minibatch = dataLoader[n]
    tools.showNineGrid_3x3(minibatch['image'][0], minibatch['image'][1], minibatch['image'][2],
                           minibatch['image'][3], minibatch['image'][4], minibatch['image'][5],
                           minibatch['image'][6], minibatch['image'][7], minibatch['image'][8])

    with torch.no_grad():
        images = modelGu(minibatch['image'])
        recons = modelGd(images)

    tools.showNineGrid_3x3(images[0], images[1], images[2],
                           images[3], images[4], images[5],
                           images[6], images[7], images[8])

    tools.showNineGrid_3x3(recons[0], recons[1], recons[2],
                           recons[3], recons[4], recons[5],
                           recons[6], recons[7], recons[8])

