import argparse
import pickle
import torch
import torch.nn as nn
import tools
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf
import torchvision.datasets as dset
import random
import DATASET_MNIST

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device
    ##########################################################################
    ## load the AI model
    parser = argparse.ArgumentParser()
    parser.add_argument("--ModelGuFile", type=str, help="None or the path for Gu model")
    parser.add_argument("--sampleID", type=int, help="specify the sample ID")
    args = parser.parse_args()

    modelGu_file = open(args.ModelGuFile, "rb")  # open the model file
    modelGu = pickle.load(modelGu_file)  # load the model file
    if isinstance(modelGu, nn.DataParallel):
        modelGu = modelGu.module
    modelGu.to('cpu')  # push model to GPU device
    modelGu.eval()  # set the model to evaluation mode, (the dropout layer need this)
    modelGu_file.close()  # close the model file

    image_H, image_W = 28, 28
    ## set the data set
    dataset = DATASET_MNIST.TRAINSET("./mnist/mnist.mat")
    dataLoader = DATASET_MNIST.DATASET_LOADER(dataset, minibatch_size=1)
    n = args.sampleID if args.sampleID else random.randint(0, len(dataLoader))
    original_image = dataLoader[n]['image']
    generate_image = modelGu(original_image)
    targeted_image = 1 - original_image
    show_image = torch.cat((original_image[0][0], generate_image[0][0], targeted_image[0][0]), dim=1)
    show_image = ttf.to_pil_image(show_image)
    print("show the final image")
    show_image.show()