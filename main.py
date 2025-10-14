import os
import sys
import logging
import argparse
from shutil import copyfile
import torch
from Codes import utils
from Codes.network import UNet
from Codes.training import *
from Codes.Losses.losses import MSELoss, SnakeFastLoss, SnakeSimpleLoss
from Codes.dataset import DriveDataset, collate_fn
from Codes import utils
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def main(config_file="main.config", resume_from=None, start_epoch=0):

    torch.set_num_threads(1)

    __c__ = utils.yaml_read(config_file)
    
    output_path = __c__["output_path"]
    batch_size = __c__["batch_size"]
    ours = __c__["ours"]
    
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "main.log"))

    copyfile(config_file, os.path.join(output_path, "main.config"))
    copyfile(__file__, os.path.join(output_path, "main.py"))

    logger.info("Command line: {}".format(' '.join(sys.argv)))

    logger.info("Loading training dataset")
    dataset_training = DriveDataset(train=True, cropSize=tuple(__c__["crop_size"]), th=__c__["threshold"])
    dataloader_training= DataLoader(dataset_training, batch_size=batch_size, num_workers=4, \
                                    shuffle=True, collate_fn=collate_fn)
    
    logger.info("Done. {} datapoints loaded.".format(len(dataset_training)))

    logger.info("Loading validation dataset")
    dataset_validation = DriveDataset(train=False, cropSize=tuple(__c__["crop_size"]), th=__c__["threshold"])
    dataloader_validation = DataLoader(dataset_validation, batch_size=1, num_workers=1, \
                                        shuffle=False, collate_fn=collate_fn)
    
    logger.info("Done. {} datapoints loaded.".format(len(dataset_validation)))
    
    training_step = TrainingEpoch(dataloader_training,
                                  ours,
                                  __c__["ours_start"])
    
    validation = Validation(tuple(__c__["crop_size_test"]),
                            tuple(__c__["margin_size"]),
                            dataloader_validation,
                            __c__["num_classes"],
                            output_path)

    logger.info("Creating model...")
    network = UNet(in_channels=__c__["in_channels"],
                   m_channels=__c__["m_channels"],
                   out_channels=__c__["num_classes"],
                   n_convs=__c__["n_convs"],
                   n_levels=__c__["n_levels"],
                   dropout=__c__["dropout"],
                   batch_norm=__c__["batch_norm"],
                   upsampling=__c__["upsampling"],
                   pooling=__c__["pooling"],
                   three_dimensional=__c__["three_dimensional"]).cuda()
        
    network.train(True)
    optimizer = torch.optim.Adam(network.parameters(), lr=__c__["lr"],
                                 weight_decay=__c__["weight_decay"])
    
    # Load checkpoint if specified
    if resume_from:
        logger.info(f"Loading checkpoint from: {resume_from}")
        checkpoint = torch.load(resume_from)
        network.load_state_dict(checkpoint)
        logger.info(f"✓ Loaded checkpoint successfully")

    if __c__["lr_decay"]:
        lr_lambda = lambda it: 1/(1+it*__c__["lr_decay_factor"])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = None
    
    base_loss = MSELoss().cuda()
    
    if ours:
        stepsz = __c__["stepsz"]
        alpha = __c__["alpha"]
        beta = __c__["beta"]
        fltrstdev = __c__["fltrstdev"]
        extparam = __c__["extparam"]
        nsteps = __c__["nsteps"]
        ndims = __c__["ndims"]
        cropsz = __c__["cropsz"]
        dmax = __c__["dmax"]
        maxedgelength = __c__["maxedgelength"]
        extgradfac = __c__["extgradfac"]
        
        our_loss = SnakeSimpleLoss(stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                                               cropsz,dmax,maxedgelength,extgradfac).cuda()
    else:
        our_loss = None

    logger.info("Running...")

    trainer = Trainer(training_step=lambda iter: training_step(iter, network, optimizer,
                                                                lr_scheduler, base_loss, our_loss),
                         validation   =lambda iter: validation(iter, network, base_loss),
                         valid_every=__c__["valid_every"],
                         print_every=__c__["print_every"],
                         save_every=__c__["save_every"],
                         save_path=output_path,
                         save_objects={"network":network},
                         save_callback=None,
                         ours=__c__["ours"],
                         ours_start=__c__["ours_start"],
                         starting_iteration=start_epoch)

    trainer.run_for(__c__["num_iters"])

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="main.config")
    parser.add_argument("--resume_from", "-r", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--start_epoch", "-s", type=int, default=0,
                        help="Starting epoch number when resuming (e.g., 400)")

    args = parser.parse_args()

    main(**vars(args))
