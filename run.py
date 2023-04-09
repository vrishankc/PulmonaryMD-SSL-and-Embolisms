import argparse
import torch
import torch.backends.cudnn as cudnn

from torch.optim.adam import Adam

from torch.utils.data import DataLoader
from dataLoad import trainDataset, valDataset
#from resnet50_encoder import ResnetPulmonaryMD
from PulmonaryMD import PulmonaryMD
from mobilenet_encoder import MobileNetEncoder
from swin_transformer_encoder import SwinEncoder
#from vit_encoder import Encoder

parser = argparse.ArgumentParser(description = "PulmonaryMD")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument("--epochs", default = 75, type = int, metavar = "N", help = "total number of epochs")
parser.add_argument("-b", "--batch-size", default = 8, type = int, metavar = "N", help = "mini-batch-size")
parser.add_argument("--lr", "--learning-rate", default = 1e-2, type = float, metavar = "W", help = "initial learning rate", dest = "lr")
parser.add_argument("--wdecay", "--weight-decay", default = 1e-3, type = float, metavar = "W", help = "weight decay", dest = "weight_decay")
parser.add_argument("--seed", default = 42, type = int, help = "seed for initializing training")
parser.add_argument('--disable-cuda', action='store_false', help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument("--out_dim", default = 250, type = int, help = "feature dim (default : 128)")
parser.add_argument("--log-every-n-steps", default = 100, type = int, help = "log every n steps")
parser.add_argument("--temperature", default = 0.5, type = float, help = "temperature")
parser.add_argument("--n-views", default = 2, type = int, metavar = "N", help = "number of views")
parser.add_argument("--gpu-index", default = 0, type = int, help = "GPU index")

def main():
    arguments = parser.parse_args()
    assert arguments.n_views == 2, "Only two views may be created."
    if torch.cuda.is_available():
        device = "cuda"
        arguments.device = torch.device("cuda")
        cudnn.benchmark = True
        cudnn.deterministic = True
    else: 
        arguments.device = torch.device("cpu")
        arguments.gpu_index = -1
    
    training_dataset = trainDataset
    train_loader = DataLoader(
        training_dataset, batch_size = arguments.batch_size, num_workers = arguments.workers, pin_memory = True, drop_last = True, shuffle = True
    )
    '''val_loader = DataLoader(
        valDataset, batch_size=arguments.batch_size, num_workers = arguments.workers, pin_memory = True, drop_last = True, shuffle = True
    )'''
    #checkpoint = torch.load('runs/FIFTH/checkpoint_0075.pth.tar')
    #model = ResnetPulmonaryMD(out_dimensions = 4 * arguments.out_dim).to(arguments.device)
    model = SwinEncoder().to(arguments.device)
    #model.load_state_dict(checkpoint['state_dict'])

    optimizer = Adam(params = model.parameters(), lr = arguments.lr, weight_decay = arguments.weight_decay)
    #optimizer = optim.SGD(params = model.parameters(), lr = arguments.lr, momentum = 0.09, weight_decay= arguments.weight_decay)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader), eta_min = 0, last_epoch = -1)

    with torch.cuda.device(arguments.gpu_index):
        pulmonary_md = PulmonaryMD(model = model, optimizer = optimizer, scheduler = scheduler, args = arguments)
        pulmonary_md = pulmonary_md.train(train_loader)
        
        

if __name__ == '__main__':
    main()
    #torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    print("Trained!!!")
