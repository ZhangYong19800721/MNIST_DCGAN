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
--N_EPOCHS=9000
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
    parser.add_argument("--logDir", type=str, help="The log directory")
    args = parser.parse_args()

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

    Gu = Model.GeneratorUx1(1, 32, 1)  # create a generator
    Gu.apply(tools.weights_init)  # initialize weights for generator

    # Setup optimizers for both G and D
    if args.optimizer == 'SGD':
        optimizerGu = optim.SGD(Gu.parameters(), lr=args.learn_rate)
    elif args.optimizer == 'ADAM':
        optimizerGu = optim.Adam(Gu.parameters(), lr=args.learn_rate, betas=(0.9, 0.999))
    elif args.optimizer == 'RMSProp':
        optimizerGu = optim.RMSprop(Gu.parameters(), lr=args.learn_rate)


    ## push models to GPUs
    Gu = Gu.to(device)
    if device.type == 'cuda' and args.NGPU > 1:
        Gu = nn.DataParallel(Gu, list(range(args.NGPU)))

    print("Start to train .... ")
    alpha = 0.01
    AVE_MMSE = tools.EXPMA(alpha)

    MSE = nn.MSELoss()

    for epoch in range(args.B_EPOCHS, args.N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update D network:
            # train with all-real batch
            minibatch = dataLoader[minibatch_id]
            real_images = minibatch['image']
            real_images = real_images.to(device)
            fine_images = 1 - real_images

            Gu.zero_grad()  # set the generator gradient to zero
            output_Gu = Gu(real_images)
            loss_mmse = MSE(output_Gu, fine_images)
            loss_mmse.backward()
            optimizerGu.step()  # Update Gu parameters

            V_AVE_MMSE = AVE_MMSE.expma(loss_mmse.item())

            message = "Epoch:%5d/%5d, MinibatchID:%5d/%5d, MMSE:% 6.12f" % \
                      (epoch, args.N_EPOCHS, minibatch_id, minibatch_count, V_AVE_MMSE)
            print(message)

            istep = minibatch_count * (epoch - args.B_EPOCHS) + minibatch_id
            writer.add_scalar("MMSE", V_AVE_MMSE, istep)

            if istep % 300 == 0:
                # save model every 1000 iteration
                model_Gu_file = open(args.outputDir + "/" + open_time_str + "/pretrained_Gu_CPU.pkl", "wb")
                pickle.dump(Gu.to("cpu"), model_Gu_file)
                Gu.to(device)
                model_Gu_file.close()

        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
