import argparse
import torch
from model.GAN import Discriminator
from model.GAN import Generator
from model.GAN import weights_init
from data.data_utils import get_dl
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a fully-connected GAN on MNIST")
    parser.add_argument("--epochs",type=int,default=5,help="number of training epochs")
    parser.add_argument("--batch-size",type=int,default=128,help="batch size")
    parser.add_argument("--lr",type=float,default=0.0002,help="learning rate for both optimizers")
    parser.add_argument("--input-size",type=int,default=100,help="dimensionality of the generator's noise input")
    parser.add_argument("--sample-dir",default="samples",help="where generated image grids are saved")
    parser.add_argument("--checkpoint-dir",default="checkpoints",help="where model checkpoints are saved")
    parser.add_argument("--seed",type=int,default=4,help="random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader,test_loader = get_dl(args.batch_size)
    dl = {}
    dl['train'] = train_loader
    dl['valid'] = test_loader

    disc = Discriminator()
    gen = Generator(args.input_size)
    disc.apply(weights_init)
    gen.apply(weights_init)
    # betas=(0.5,0.999) is the standard DCGAN setting; the default (0.9,0.999) tends to destabilize training
    optimD = torch.optim.Adam(disc.parameters(),lr=args.lr,betas=(0.5,0.999))
    optimG = torch.optim.Adam(gen.parameters(),lr=args.lr,betas=(0.5,0.999))
    loss_fn = torch.nn.BCELoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    disc.to(device)
    gen.to(device)

    train_model(
        args.epochs,disc,gen,optimD,optimG,dl,loss_fn,args.input_size,args.batch_size,
        sample_dir=args.sample_dir,checkpoint_dir=args.checkpoint_dir,
    )

if __name__ == "__main__":
    main()
