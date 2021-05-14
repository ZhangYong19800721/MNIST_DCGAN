# coding=utf-8

########################################################################################################################
# train_DCWGAN.py
# train the model, include parameters initializing
########################################################################################################################
import os
import argparse
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import Model
import DATASET_MNIST
import tools
from torch.utils.tensorboard import SummaryWriter

"""
--learn_rate=0.0005
--optimizer=ADAM
--minibatch_size=3000
--NGPU=2
--B_EPOCHS=0
--N_EPOCHS=9999
--outputDir=./output
--logDir=./log
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The manual random seed")
    parser.add_argument("--learn_rate", type=float, help="The learn rate")
    parser.add_argument("--optimizer", type=str, help="The optimizer, SGD or ADAM or RMSProp")
    parser.add_argument("--minibatch_size", type=int, help="The minibatch size")
    parser.add_argument("--NGPU", type=int, help="specify the number of GPUs to use")
    parser.add_argument("--B_EPOCHS", type=int, help="The start epoch id")
    parser.add_argument("--N_EPOCHS", type=int, help="The end epoch id")
    parser.add_argument("--outputDir", type=str, help="the output directory")
    parser.add_argument("--isLoadG", type=str, help="None or the path for pretrained G model")
    parser.add_argument("--isLoadD", type=str, help="None or the path for pretrained D model")
    parser.add_argument("--logDir", type=str, help="The log directory")
    args = parser.parse_args()

    nz = 60

    open_time_str = time.strftime("%Y%m%d[%H:%M:%S]", time.localtime())
    os.mkdir(args.outputDir + "/" + open_time_str)
    writer = SummaryWriter(args.logDir + "/" + open_time_str)

    ## set the hyper parameters
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    ## set the data set
    dataset = DATASET_MNIST.TRAINSET("./data/mnist.mat")
    dataLoader = DATASET_MNIST.DATASET_LOADER(dataset, minibatch_size=args.minibatch_size)
    minibatch_count = len(dataLoader)

    ## specify the computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.NGPU > 0 else "cpu")

    # show some data samples
    print("Show some images ...., press ENTER to continue. ")
    n = random.randint(0, len(dataLoader))
    minibatch = dataLoader[n]
    tools.showNineGrid_3x3(minibatch['image'][0], minibatch['image'][1], minibatch['image'][2],
                           minibatch['image'][3], minibatch['image'][4], minibatch['image'][5],
                           minibatch['image'][6], minibatch['image'][7], minibatch['image'][8])

    if args.isLoadG:
        ##########################################################################
        ## load the pretrained G model
        modelG_file = open(args.isLoadG, "rb")  # open the model file
        Gu = pickle.load(modelG_file)  # load the model file
        if isinstance(Gu, nn.DataParallel):
            Gu = Gu.module
        Gu.to(device)  # push model to GPU device
        modelG_file.close()  # close the model file
    else:
        G = Model.Generator(nz=nz)  # create a generator
        G.apply(tools.weights_init)  # initialize weights for generator

    if args.isLoadD:
        ##########################################################################
        ## load the pretrained Db model
        modelD_file = open(args.isLoadD, "rb")  # open the model file
        D = pickle.load(modelD_file)  # load the model file
        if isinstance(D, nn.DataParallel):
            D = D.module
        D.to(device)  # push model to GPU device
        modelD_file.close()  # close the model file
    else:
        D = Model.Discriminator_GP(inChannel=1)  # create a discriminator
        D.apply(tools.weights_init)  # initialize weights for discriminator

    # Setup optimizers for both G and D
    if args.optimizer == 'SGD':
        optimizerD = optim.SGD(D.parameters(), lr=args.learn_rate, momentum=0.9)
        optimizerG = optim.SGD(G.parameters(), lr=args.learn_rate, momentum=0.9)
    elif args.optimizer == 'ADAM':
        optimizerD = optim.Adam(D.parameters(), lr=args.learn_rate, betas=(0.9, 0.999))
        optimizerG = optim.Adam(G.parameters(), lr=args.learn_rate, betas=(0.9, 0.999))
    elif args.optimizer == 'RMSProp':
        optimizerD = optim.RMSprop(D.parameters(), lr=args.learn_rate)
        optimizerG = optim.RMSprop(G.parameters(), lr=args.learn_rate)

    ## push models to GPUs
    G = G.to(device)
    D = D.to(device)
    if device.type == 'cuda' and args.NGPU > 1:
        G = nn.DataParallel(G, list(range(args.NGPU)))
        D = nn.DataParallel(D, list(range(args.NGPU)))

    print("Start to train .... ")
    alpha = 0.01
    AVE_DIFF = tools.EXPMA(alpha)
    AVE_GRDP = tools.EXPMA(alpha)

    for epoch in range(args.B_EPOCHS, args.N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update D network:
            # train with all-real batch
            minibatch = dataLoader[minibatch_id]
            real_images = minibatch['image']
            real_images = real_images.to(device)
            noise = torch.randn((args.minibatch_size, nz, 1, 1))

            ## Update D network: for WGAN maximize D(x) - D(G(z))
            D.zero_grad()  # set discriminator gradient to zero
            fake_images = G(noise).detach()
            output_real_D = D(real_images)
            output_fake_D = D(fake_images)
            diff = (output_real_D - output_fake_D).mean()
            gradient_penalty = tools.cal_gradient_penalty(D, device, real_images, fake_images)
            loss = gradient_penalty - 0.1 * diff
            loss.backward()
            optimizerD.step()

            G.zero_grad()  # set the generator gradient to zero
            fake_images = G(noise)
            output_fake_G_D = D(fake_images)
            loss_G_D = -0.1 * output_fake_G_D.mean()
            loss_G_D.backward()
            optimizerG.step()  # Update G parameters

            V_AVE_DIFF = AVE_DIFF.expma(abs(diff.item()))
            V_AVE_GRDP = AVE_GRDP.expma(gradient_penalty.item())

            message = "Epoch:%5d/%5d, MinibatchID:%5d/%5d, DIFF:% 6.12f, GRADIENT_PENALTY:% 6.12f" % \
                      (epoch, args.N_EPOCHS, minibatch_id, minibatch_count, V_AVE_DIFF, V_AVE_GRDP)
            print(message)

            istep = minibatch_count * (epoch - args.B_EPOCHS) + minibatch_id
            writer.add_scalar("AVE_DIFF", V_AVE_DIFF, istep)
            writer.add_scalar("AVE_GRDP", V_AVE_GRDP, istep)

            if istep % 300 == 0:
                # save model every 1000 iteration
                model_G_file = open(args.outputDir + "/" + open_time_str + "/model_G_CPU.pkl", "wb")
                model_D_file = open(args.outputDir + "/" + open_time_str + "/model_D_CPU.pkl", "wb")
                pickle.dump(G.to("cpu"), model_G_file)
                pickle.dump(D.to("cpu"), model_D_file)
                G.to(device)
                D.to(device)
                model_G_file.close()
                model_D_file.close()

        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
