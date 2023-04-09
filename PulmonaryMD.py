import logging
import os
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_configuration_file, accuracy, save_check

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class PulmonaryMD(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.summary_writer = SummaryWriter()
        logging.basicConfig(filename = os.path.join(self.summary_writer.log_dir, 'training.log'), level = logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
    
    '''def nt_xent_loss(self, feats):
        out = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim = 0)
        n_samples = len(out)

        # Similarity Matrix
        cov = torch.mm(out, out.t())
        sim = torch.exp(cov / self.args.temperature)

        # Negative Similarity
        mask = ~torch.eye(n_samples, device = sim.device).bool()
        negatives = sim.masked_select(mask).view(n_samples, -1).sum(dim = -1)

        # Positive Similarity
        pos = torch.exp(torch.sum(out, dim = -1))
        pos = torch.cat([pos, pos], dim = 0)

        loss = -torch.log(pos / negatives).mean()
        return loss'''
    '''def info_nce_loss(self, feats, mode='train'):
        # Encode all images
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll'''
    def info_nce_loss(self, feats):
        lbls = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim = 0)
        lbls = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).float()
        lbls = lbls.to(self.args.device)

        similarities = torch.matmul(feats, feats.T)
        #similarities = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        #similarities = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)

        posits = similarities[lbls.bool()].view(lbls.shape[0], -1)
        negats = similarities[~lbls.bool()].view(similarities.shape[0], -1)
        logits = torch.cat([posits, negats], dim = 1)
        lbls = torch.zeros(logits.shape[0], dtype = torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, lbls
    
    def train(self, train_loader):
        scaler = GradScaler(enabled = self.args.fp16_precision)
        save_configuration_file(self.summary_writer.log_dir, self.args)

        n_iterations = 0
        logging.info(f"PulmonaryMD training beginning for {self.args.epochs} epochs")
        logging.info(f"PulmonaryMD Training with gpu: {self.args.disable_cuda}")
        train_loss = []
        for counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat((images, _), dim = 0)
                

                with autocast(enabled = self.args.fp16_precision):
                    images = images.to(self.args.device)
                    feats = self.model(images)
                    logits, lbls = self.info_nce_loss(feats)
                    #logits, lbls = self.nt_xent_loss(feats)
                    loss = self.criterion(logits, lbls)
                    train_loss.append(loss)
                    
                
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iterations % self.args.log_every_n_steps == 0:
                    topOne, topFive = accuracy(logits, lbls, topk = (1, 5))
                    self.summary_writer.add_scalar("loss", loss, global_step = n_iterations)
                    self.summary_writer.add_scalar("acc/topOne", topOne[0], global_step = n_iterations)
                    self.summary_writer.add_scalar("acc/topFive", topFive[0], global_step = n_iterations)
                    self.summary_writer.add_scalar("learning_rate", self.scheduler.get_lr()[0], global_step = n_iterations)
                
                n_iterations += 1
            
            '''self.model.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    images = torch.cat((images, _), dim = 0)
                    images = images.to(self.args.device)
                    feats = self.model(images)
                    logits, lbls = self.info_nce_loss(feats)
                    loss = self.criterion(logits, lbls)
                    val_loss.append(loss)

                    if n_iterations % self.args.log_every_n_steps == 0:
                        valTopOne, valTopFive = accuracy(logits, lbls, topk = (1, 5))
                        self.summary_writer.add_scalar("val_loss", loss, global_step = n_iterations)
                        self.summary_writer.add_scalar("val_acc/topOne", valTopOne[0], global_step = n_iterations)
                        self.summary_writer.add_scalar("val_acc/topFive", valTopFive[0], global_step = n_iterations)
                        #self.summary_writer.add_scalar("learning_rate", self.scheduler.get_lr()[0], global_step = n_iterations)
                
                n_iterations += 1'''
       
            if counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {counter + 1}\tLoss: {loss}\t TopOne accuracy: {topOne[0]} \t TopFive accuracy: {topFive[0]}")
        
        logging.info("Training has finished.")
        checkpoint_name = 'mobilenet_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_check({
            'epoch': self.args.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, is_best = False, filename = os.path.join(self.summary_writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.summary_writer.log_dir}")

