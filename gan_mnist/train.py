import os
import torch
from torchvision.utils import save_image

#function to train the GAN model
def train_model(no_of_epochs,disc,gen,optimD,optimG,dataloaders,loss_fn,input_size,batch_size,sample_dir="samples",checkpoint_dir="checkpoints"):
    """
    disc: Discriminator model
    gen: Generator model
    optimD: Optimizer for Discriminator
    optimG: Optimizer for Generator
    sample_dir: where generated image grids are saved each epoch
    checkpoint_dir: where generator/discriminator weights are saved each epoch
    """

    # setting the device as cuda or cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    reall = 1  # real label
    fakel = 0  # fake label
    os.makedirs(sample_dir,exist_ok=True)
    os.makedirs(checkpoint_dir,exist_ok=True)
    # fixed noise so the same 128 digits are regenerated each epoch, to see the model improve
    fixed_noise = torch.randn(batch_size,input_size,device=device)
    #running each epoch
    for epoch in range(no_of_epochs):
        print('Epoch {}/{}'.format(epoch+1,no_of_epochs))
        running_loss_D = 0
        running_loss_G = 0
        for phase in ["train"]:
            #getting input and label from dataloader
            for inputs, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                #converting labels into torch with proper size as per the batch size
                real_label = torch.full((batch_size,),reall,dtype=inputs.dtype,device=device)
                fake_label = torch.full((batch_size,),fakel,dtype=inputs.dtype,device=device)

                optimD.zero_grad()
                #output from discriminator
                output =disc(inputs)
                #Discriminator real loss
                # Compairing output label with real label which is loss
                D_real_loss = loss_fn(output,real_label)
                D_real_loss.backward()
                
                #random torch tensor as a noise data
                noise = torch.randn(batch_size,input_size,device=device)
                #passing noise throgh generator to get fake image
                fake = gen(noise)
                #passing fake image through discriminator with detaching(not passing gradient)
                output = disc(fake.detach())
                
                #Discriminator fake loss
                D_fake_loss = loss_fn(output,fake_label)
                #back propogation
                D_fake_loss.backward()

                # total loss for Discriminator
                Disc_loss = D_real_loss+D_fake_loss
                running_loss_D = running_loss_D+Disc_loss
                optimD.step()

                optimG.zero_grad()
                #passing fake image obtained from generator to discriminator
                output = disc(fake)
                #getting generator loss by giving fake image as input but giving real label
                Gen_loss = loss_fn(output,real_label)
                running_loss_G = running_loss_G + Gen_loss
                #backpropogation
                Gen_loss.backward()
                optimG.step()
        print("Discriminator Loss : {}".format(running_loss_D))
        print("Generator Loss : {}".format(running_loss_G))

        # saving a grid of generated digits for this epoch, rescaled from tanh's [-1,1] to [0,1]
        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_noise)
        gen.train()
        save_image(samples,os.path.join(sample_dir,"epoch_{:03d}.png".format(epoch+1)),normalize=True)

        # saving model weights so training can be resumed or the generator reused later
        torch.save({
            "epoch": epoch+1,
            "generator": gen.state_dict(),
            "discriminator": disc.state_dict(),
            "optimG": optimG.state_dict(),
            "optimD": optimD.state_dict(),
        },os.path.join(checkpoint_dir,"checkpoint_epoch_{:03d}.pth".format(epoch+1)))


