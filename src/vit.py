import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import fixed_image_standardization
from sklearn.metrics import confusion_matrix
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from vit_pytorch import ViT
from byol_pytorch import BYOL
from data_loader import get_loader, read_dataset, CompositeDataset
from model import FaceRecognitionCNN
from utils import write_json, copy_file, count_parameters

transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.1),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(device))

datasets = read_dataset(
    '../dataset/mtcnn/', transform=transform,
    max_images_per_video=500, max_videos=1000,
    window_size=10, splits_path='../dataset/splits/'
)
# only neural textures c40 and original c40
datasets = {
    k: v for k, v in datasets.items() 
    if ('original' in k or 'neural' in k or 'face2face' in k or 'faceswap' in k or 'deepfakes' in k) and 'c23' in k
}
print('Using training data: ')
print('\n'.join(sorted(datasets.keys())))

trains, vals, tests = [], [], []
for data_dir_name, dataset in datasets.items():
    train, val, test = dataset
    # repeat original data multiple times to balance out training data
    compression = data_dir_name.split('_')[-1]
    num_tampered_with_same_compression = len({x for x in datasets.keys() if compression in x}) - 1
    count = 1 if 'original' not in data_dir_name else num_tampered_with_same_compression
    for _ in range(count):
        trains.append(train)
    vals.append(val)
    tests.append(test)
    
train_dataset, val_dataset, test_dataset = CompositeDataset(*trains), CompositeDataset(*vals), CompositeDataset(*tests)

tqdm.write('train data size: {}, validation data size: {}'.format(len(train_dataset), len(val_dataset)))

train_loader = get_loader(
    train_dataset, 32, shuffle=True, num_workers=8
)
val_loader = get_loader(
    val_dataset, 32, shuffle=True, num_workers=8
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('training on', device)

IMG_SIZE = 36
PATCH_SIZE = 4
DIM = 1024
MLP_DIM = 1024
DEPTH = 4
HEADS = 4
CHANNELS = 192

class Encoder2DViT(nn.Module):
    def __init__(self, face_recognition_cnn_path=None):
        super(Encoder2DViT, self).__init__()

        face_cnn = FaceRecognitionCNN()
        
        if face_recognition_cnn_path is not None:
            face_cnn = nn.DataParallel(face_cnn)
            state_dict = torch.load(face_recognition_cnn_path, map_location='cpu')
            face_cnn.load_state_dict(state_dict)

        if face_recognition_cnn_path:
            modules = list(face_cnn.module.resnet.children())[:-12]
            self.encoder2d = nn.Sequential(*modules)

        else:
            modules = list(face_cnn.resnet.children())[:-12]
            self.encoder2d = nn.Sequential(*modules)
    
        del face_cnn
        
        self.vit = ViT(image_size = IMG_SIZE, patch_size = PATCH_SIZE, num_classes = 5, dim = DIM, depth = DEPTH, heads = HEADS,
                       mlp_dim = MLP_DIM, dropout = 0.1, emb_dropout = 0.1, channels = CHANNELS)
        
    
    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        images = images.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch_size * depth, num_channels, height, width)
        out = self.encoder2d(images)
        out = self.vit(out)
                
        return  out
    
model = Encoder2DViT('./model/facenet/model.pt')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-3
)

# decrease learning rate if validation accuracy has not increased
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=1/4, patience=2, verbose=True,
)

def save_model_checkpoint(epoch, model, val_acc):
    
    model_dir = os.path.join('./model', 'encvit')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'model.pt')
    torch.save(model.state_dict(), model_path)

    model_info = {
        'epoch': epoch,
        'val_acc': val_acc[0],
        'model_str': str(model)
    }
    json_path = os.path.join(model_dir, 'info.json')
    write_json(model_info, json_path)

    #src_model_file = os.path.join('facenet', 'model.py')
    #dest_model_file = os.path.join(model_dir, 'model.py')
    #copy_file(src_model_file, dest_model_file)

    tqdm.write(f'New checkpoint saved at {model_path}')


def print_training_info(batch_accuracy, loss, step):
    log_info = 'Training - Loss: {:.4f}, Accuracy: {:.4f}'.format(loss.item(), batch_accuracy)
    tqdm.write(log_info)

    #writer.add_scalar('training loss', loss.item(), step)
    #writer.add_scalar('training acc', batch_accuracy, step)


def print_validation_info(criterion, device, model, val_loader, epoch, step):
    model.eval()
    with torch.no_grad():
        loss_values = []
        all_predictions = []
        all_targets = []
        targets = []
        outputs = []
        for video_ids, frame_ids, images, target in val_loader:
            images = images.to(device)
            target = target.to(device)
            target = target.long()
            tiled_target = tile(target, 0, 10)
            output = model(images)
            loss = criterion(output, tiled_target)
            loss_values.append(loss.item())
            targets.append(tiled_target)
            outputs.append(output)
            #predictions = outputs > 0.0
            #all_predictions.append(predictions)
            #all_targets.append(targets)
            #if args.debug:
            #    tqdm.write(outputs)
            #    tqdm.write(predictions)
            #    tqdm.write(targets)
        
        val_loss = sum(loss_values) / len(loss_values)
        
        outputs = torch.cat(outputs, 0)
        targets = torch.cat(targets, 0)
        
        val_accuracy = float((outputs.argmax(1)).eq(targets).sum()) / len(targets)
        
        total_target = targets.unique(return_counts=True)[1]
        pristine = ((outputs.argmax(1) == 0) * (targets == 0)).sum() / total_target[0]
        face2face = ((outputs.argmax(1) == 1) * (targets == 1)).sum() / total_target[1]
        faceswap = ((outputs.argmax(1) == 2) * (targets == 2)).sum() / total_target[2]
        neural = ((outputs.argmax(1) == 3) * (targets == 3)).sum() / total_target[3]
        deepfake = ((outputs.argmax(1) == 4) * (targets == 4)).sum() / total_target[4]
        
        tqdm.write(
            'Validation - Loss: {:.3f}, Acc: {:.3f}, Pr: {:.3f}, Ff: {:.3f}, Fs: {:.3f}, Nt: {:.3f}, Df: {:.3f}'.format(
                val_loss, val_accuracy, pristine, face2face, faceswap, neural, deepfake
            )
        )
        
    return val_accuracy, pristine, face2face, faceswap, neural, deepfake, val_loss

total_step = len(train_loader)
step = 1
best_val_loss = 1.5
for epoch in range(5):
    for i, (video_ids, frame_ids, images, targets) in \
            tqdm(enumerate(train_loader), desc=f'training epoch {epoch}', total=len(train_loader)):
        model.train()
        # Set mini-batch dataset
        images = images.to(device)
        targets = targets.to(device)

        # Forward, backward and optimize
        outputs = model(images)
        targets = targets.long()
        tiled_targets = tile(targets, 0, 10)
        loss = criterion(outputs, tiled_targets)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        batch_accuracy = float((outputs.argmax(1)).eq(tiled_targets).sum()) / len(tiled_targets)
        
        

        # Print log info
        step += 1
        
        if (i + 1) % 37806 == 0:
            print_training_info(batch_accuracy, loss, step)

        if (i + 1) % 37806 == 0:
            val_acc, pr_acc, ff_acc, fs_acc, nt_acc, df_acc, val_loss = print_validation_info(
                criterion, device, model, val_loader, epoch, step
            )
            if val_loss < best_val_loss:
                save_model_checkpoint(epoch, model, (val_acc, pr_acc, ff_acc, fs_acc, nt_acc, df_acc))
                best_val_loss = val_loss

    # validation step after full epoch
    val_acc, pr_acc, ff_acc, fs_acc, nt_acc, df_acc, val_loss = print_validation_info(
        criterion, device, model, val_loader, epoch, step
    )
    lr_scheduler.step(val_loss)
    if val_loss < best_val_loss:
        save_model_checkpoint(epoch, model, (val_acc, pr_acc, ff_acc, fs_acc, nt_acc, df_acc))
        best_val_loss = val_loss

    #if epoch == 0:
    #    for m in model.resnet.parameters():
    #        m.requires_grad_(True)
    #    tqdm.write('Fine tuning on')
    
model = Encoder2DViT()
state_dict = torch.load('./model/encvit/model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.to(device)

test_loader = get_loader(
    test_dataset, 32, shuffle=True, num_workers=2, drop_last=False
)
with torch.no_grad():
    loss_values = []
    all_predictions = []
    all_targets = []
    targets = []
    outputs = []
    for video_ids, frame_ids, images, target in tqdm(test_loader):
        images = images.to(device)
        target = target.to(device)
        target = target.long()
        tiled_target = tile(target, 0, 10)
        output = model(images)
        targets.append(tiled_target)
        outputs.append(output)
        loss = criterion(output, tiled_target)
        loss_values.append(loss.item())

#                 predictions = outputs > 0.0
#                 all_predictions.append(predictions)
#                 all_targets.append(targets)

    val_loss = sum(loss_values) / len(loss_values)

    outputs = torch.cat(outputs, 0)
    targets = torch.cat(targets, 0)
        
    val_accuracy = float((outputs.argmax(1)).eq(targets).sum()) / len(targets)

    total_target = targets.unique(return_counts=True)[1]
    pristine = ((outputs.argmax(1) == 0) * (targets == 0)).sum() / total_target[0]
    face2face = ((outputs.argmax(1) == 1) * (targets == 1)).sum() / total_target[1]
    faceswap = ((outputs.argmax(1) == 2) * (targets == 2)).sum() / total_target[2]
    neural = ((outputs.argmax(1) == 3) * (targets == 3)).sum() / total_target[3]
    deepfake = ((outputs.argmax(1) == 4) * (targets == 4)).sum() / total_target[4]
    tqdm.write(
        'Test - Loss: {:.3f}, Acc: {:.3f}, Pr: {:.3f}, Ff: {:.3f}, Fs: {:.3f}, Nt: {:.3f}, Df: {:.3f}'.format(
            val_loss, val_accuracy, pristine, face2face, faceswap, neural, deepfake
        )
    )
    
cm = confusion_matrix(targets.cpu().numpy(), outputs.cpu().argmax(1).numpy())
print(cm)