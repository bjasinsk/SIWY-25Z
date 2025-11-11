import torch
import torchvision
from torchvision.datasets import ImageFolder
import os
from torch.optim import SGD
from pathlib import Path
import numpy as np
import tqdm
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
import warnings
warnings.filterwarnings('ignore')
DATA_DIR = Path("D:\SIWY\SIWY-25Z-Jarczewski-Rozej-Jasinski\data")
from TracInPyTorch.src.tracin import vectorized_calculate_tracin_score

### MODEL
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)


def construct_rn9(num_classes=10):
    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
                torch.nn.BatchNorm2d(channels_out),
                torch.nn.ReLU(inplace=True)
        )
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model


from torchvision.datasets import ImageFolder
from torchvision import transforms

global train_dataset, val_dataset

### DATALOADER
def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True):
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
                          torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                                  torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])

    is_train = (split == 'train')
    # dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
    #                                        download=True,
    #                                        train=is_train,
    #                                        transform=transforms)
    if is_train:
      dataset = ImageFolder(root=Path(f'{DATA_DIR}/task1/task1/easy/train'), transform=transforms)
    else:
      dataset = ImageFolder(root=Path(f'{DATA_DIR}/task1/task1/easy/val'), transform=transforms)


    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    return loader

def train(model, loader, lr=0.4, epochs=24, momentum=0.9,
          weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0, model_id=0):

    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for ep in range(epochs):
        print(ep)
        for it, (ims, labs) in enumerate(loader):
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        if ep in [12, 15, 18, 21, 23]:
            torch.save(model.state_dict(), f'{DATA_DIR}/checkpoints-2cls/sd_{model_id}_epoch_{ep}.pt')

    return model

def create_criterion(model, lr, momentum=0.9, weight_decay=5e-4):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

if __name__ == "__main__":
# TODO: fix num classes to 

    print(torch.cuda.is_available())
    # train_dataset = ImageFolder(root=f'{DATA_DIR}/task1/task1/easy/train', transform=transform)
    # val_dataset = ImageFolder(root=f'{DATA_DIR}/task1/task1/easy/val', transform=transform)

    
    os.makedirs(f'{DATA_DIR}/checkpoints-2cls', exist_ok=True)
    loader_for_training = get_dataloader(batch_size=128, split='train', shuffle=True)

    # you can modify the for loop below to train more models
    for i in tqdm(range(1), desc='Training models..'):
        model = construct_rn9(2).to(memory_format=torch.channels_last).cuda()
        model = train(model, loader_for_training, model_id=i)

    ckpt_files = sorted(list((DATA_DIR / Path('./checkpoints-2cls')).rglob('*.pt')))
    # ckpts = [torch.load(ckpt), map_location='cpu' for ckpt in ckpt_files]


    batch_size = 128
    model = construct_rn9(2)
    criterion = CrossEntropyLoss(label_smoothing=0.0)
    # create_criterion(model, lr=0.4, momentum=0.9, weight_decay=5e-4)
    weights = ckpt_files
    train_loader = get_dataloader(batch_size=batch_size, split='train')
    test_loader = get_dataloader(batch_size=batch_size, split='val', augment=False)
    lr = 0.4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print("len test loader:", len(test_loader.dataset))
    use_nested_loop_for_dot_product = False # via einsum
    float_labels = False # depends on your loss function

   
    matrix = vectorized_calculate_tracin_score(
        model=model,
        criterion=criterion,
        weights_paths=weights,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        lr=lr,
        device=device,
        use_nested_loop_for_dot_product=use_nested_loop_for_dot_product,
        float_labels=float_labels,
    )

    print("TracIn Score Matrix shape:", matrix.shape)
    matrix_path = DATA_DIR / "tracin_score_matrix.pt"
    torch.save(matrix_path, matrix)
    print(f"TracIn Score Matrix saved at: {matrix_path}")