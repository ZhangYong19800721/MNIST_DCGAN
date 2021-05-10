# coding=utf-8

########################################################################################################################
# step04_Train_SPWGAN_MMSE.py
# train the model, include parameters initializing
########################################################################################################################
import argparse
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
import Model
import Data
import DATASET_MNIST
import tools
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The manual random seed")
    parser.add_argument("--MaxMinibatchID", type=int, help="the Max Minibatch ID, use this to cut the trainset")
    parser.add_argument("--learn_rate", type=float, help="The learn rate")
    parser.add_argument("--minibatch_size", type=int, help="The minibatch size")
    parser.add_argument("--NGPU", type=int, help="specify the number of GPUs to use")
    parser.add_argument("--B_EPOCHS", type=int, help="The start epoch id")
    parser.add_argument("--N_EPOCHS", type=int, help="The end epoch id")
    parser.add_argument("--isLoadPretrainedGu", type=str, help="None or the path for pretrained Gu model")
    parser.add_argument("--isLoadPretrainedD", type=str, help="None or the path for pretrained D model")
    parser.add_argument("--logdir", type=str, help="The log dir")
    args = parser.parse_args()

    writer = SummaryWriter(args.logdir + "/Train_Log_" + time.strftime("%Y%m%d[%H:%M:%S]", time.localtime()))

    ## set the hyper parameters
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    DEBUG = True
    N_GPU = args.NGPU  # we have 2 GPUs
    B_EPOCHS, N_EPOCHS = args.B_EPOCHS, args.N_EPOCHS  # train the model for n epochs
    learn_rate = args.learn_rate  # set the learning rate
    minibatch_size = args.minibatch_size  # set the minibatch size
    isLoadPretrainedGu, isLoadPretrainedD = args.isLoadPretrainedGu, args.isLoadPretrainedD
    MAX_MINIBATCH_NUM = args.MaxMinibatchID if args.MaxMinibatchID != None else 1e100

    ## set the data set
    dataset = DATASET_MNIST.TRAINSET("./mnist/mnist.mat")
    dataLoader = DATASET_MNIST.DATASET_LOADER(dataset, minibatch_size=minibatch_size)
    minibatch_count = min(MAX_MINIBATCH_NUM, len(dataLoader))

    ## specify the computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() and N_GPU > 0 else "cpu")

    # show some data samples
    if DEBUG:
        print("Show some images ...., press ENTER to continue. ")
        n = random.randint(0, len(dataLoader))
        minibatch = dataLoader[n]
        tools.showNineGrid_1x2(minibatch['image'][0], minibatch['image'][1])

    if isLoadPretrainedGu:
        ##########################################################################
        ## load the pretrained G model
        modelGu_file = open(isLoadPretrainedGu, "rb")  # open the model file
        Gu = pickle.load(modelGu_file)  # load the model file
        if isinstance(Gu, nn.DataParallel):
            Gu = Gu.module
        Gu.to(device)  # push model to GPU device
        modelGu_file.close()  # close the model file
    else:
        Gu = Model.GeneratorMNIST(inChannel=1, interChannel=64, outChannel=1)  # create a generator
        Gu.apply(tools.weights_init)  # initialize weights for generator

    if isLoadPretrainedD:
        ##########################################################################
        ## load the pretrained Db model
        modelD_file = open(isLoadPretrainedD, "rb")  # open the model file
        D = pickle.load(modelD_file)  # load the model file
        if isinstance(D, nn.DataParallel):
            D = D.module
        D.to(device)  # push model to GPU device
        modelD_file.close()  # close the model file
    else:
        D = Model.DiscriminatorMNIST(inChannel=1)  # create a discriminator
        D.apply(tools.weights_init)  # initialize weights for discriminator

    # Initialize BCE and MSE function
    MSE = nn.MSELoss(reduction='mean')

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(D.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optimizerGu = optim.Adam(Gu.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    ## push models to GPUs
    Gu = Gu.to(device)
    D = D.to(device)
    if device.type == 'cuda' and N_GPU > 1:
        Gu = nn.DataParallel(Gu, list(range(N_GPU)))
        D = nn.DataParallel(D, list(range(N_GPU)))

    print("Start to train .... ")
    alpha = 0.01
    AVE_DIFF = tools.EXPMA(alpha)
    AVE_MMSE = tools.EXPMA(alpha)

    # leakyRELU = nn.LeakyReLU(0.0)
    for epoch in range(B_EPOCHS, N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update D network:
            # train with all-real batch
            minibatch = dataLoader[minibatch_id]
            images = minibatch['image']
            images = images.to(device)
            negative_images = 1 - images

            ## Update D network: for WGAN maximize D(x) - D(G(z))
            D.zero_grad()  # set discriminator gradient to zero
            images_fake = Gu(images)
            output_real_D = D(negative_images)
            output_fake_D = D(images_fake)
            diff = (output_real_D - output_fake_D).mean()
            loss = -diff
            loss.backward()
            optimizerD.step()

            Gu.zero_grad()  # set the generator gradient to zero
            images_fake = Gu(images)
            output_fake_G_D = D(images_fake)
            loss_optim_mmse = MSE(images_fake, negative_images)
            loss_G_D = -output_fake_G_D.mean() # + 1e-3 * loss_optim_mmse
            loss_G_D.backward()
            optimizerGu.step()  # Update Gu parameters

            V_AVE_DIFF = AVE_DIFF.expma(abs(diff.item()))
            V_AVE_MMSE = AVE_MMSE.expma(loss_optim_mmse.mean().item())

            message = "Epoch:%5d/%5d, MinibatchID:%5d/%5d, DIFF:% 6.12f, MMSE: % 6.12f" % (epoch, N_EPOCHS-1, minibatch_id, minibatch_count, V_AVE_DIFF, V_AVE_MMSE)
            print(message)

            istep = minibatch_count * (epoch - B_EPOCHS) + minibatch_id
            writer.add_scalar("AVE_DIFF", V_AVE_DIFF, istep)
            writer.add_scalar("AVE_MMSE", V_AVE_MMSE, istep)

            if istep % 500 == 0:
                # save model every 1000 iteration
                model_Gu_file = open(r"./model/model_Gu_CPU_%05d.pkl" % epoch, "wb")
                model_D_file = open(r"./model/model_D_SP_CPU_%05d.pkl" % epoch, "wb")
                pickle.dump(Gu.to("cpu"), model_Gu_file)
                pickle.dump(D.to("cpu"), model_D_file)
                Gu.to(device)
                D.to(device)
                model_Gu_file.close()
                model_D_file.close()

        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
