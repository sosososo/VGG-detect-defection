from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import vggmodel
from config import *
import mynet
import pickle

from visualize import make_dot


def train_model(model, criterion, optimizer, scheduler, f , num_epochs=25):
    print('Start train ******')
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_rec=0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        f.write('-' * 10)
        f.write('\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            tn=0.0
            fn=0.0
            tp=0.0
            fp=0.0
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data
                #totalsize +=len(inputs)
                #print(totalsize)
                # wrap them in Variable
                if USE_CUDA:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'val':
                    targets=labels.data
                    for (pred,target)  in zip(preds,targets):
                        if target == 1:
                            if pred == 1:
                                tp+=1.0
                            else:
                                fn+=1.0
                        else:
                            if pred == 0:
                                tn+=1.0
                            else:
                                fp+=1.0

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':
                print('Val result')
                print('pos num: {}\ntp:  {}\nfn:  {}\ntn:  {}\nfp:  {}'.format((tp+fn+tn+fp),tp,fn,tn,fp))
                f.write('Val result:\npos num: {}\ntp:  {}\nfn:  {}\ntn:  {}\nfp:  {}\n'.format((tp+fn+tn+fp),tp,fn,tn,fp))
                if tp+fn >0.0:
                    rec = tp /(tp+ fn)
                    print('rec:  {:4f}'.format(rec))
                    f.write('rec:  {:4f}\n'.format(rec))
                if tp+fp >0.0:
                    pre = tp /(tp+ fp)
                    print('pre:  {:4f}'.format(pre))
                    f.write('pre:  {:4f}\n'.format(pre))
                if tp+fn>0.0:
                    mdr = fn /(tp+ fn)
                    print('mdr:  {:4f}'.format(mdr))
                    f.write('mdr:  {:4f}\n'.format(mdr))
                if fp+tn>0.0:
                    fdr = fp /(fp+ tn)
                    print('fdr:  {:4f}'.format(fdr))
                    f.write('fdr:  {:4f}\n'.format(fdr))
                print('')
            #
            if phase == 'val' and best_rec < rec:
                best_rec=rec
            if phase == 'val' and best_acc <  epoch_acc:
                best_acc=epoch_acc
                best_model_wts = model.state_dict()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    f.write('Best val Acc: {:4f}\n'.format(best_acc))
    f.write('Best val Rec: {:4f}\n'.format(best_rec))




    if epoch > 0 and epoch % 5 == 0:
        torch.save(best_model_wts,os.path.join(param_dir, '{}_param_epoch{}_rec{:4f}.pth'.format(model_type,epoch, rec)))

    # load best model weights
    print('Finish train ******')
    print('--------')
    print('Start save model ******')
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, os.path.join(param_dir, '{}_{}'.format(model_type, param_name)))
    print('Finish save model ******')
    return model

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data,0)

if __name__ == '__main__':

    # data_transform,
    # pay attention that the input of Normalize() is Tensor
    # and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomSizedCrop(224),
            transforms.Scale(vgg_input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)

        ]),
        'val': transforms.Compose([
            transforms.Scale(vgg_input_size),  # 256 init
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    net_data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomSizedCrop(224),
            transforms.Scale(net_input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)

        ]),
        'val': transforms.Compose([
            transforms.Scale(net_input_size),  # 256 init
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    print('Prepare data ******')
    print('data path:   %s' % data_dir)
    file_path=os.path.join(result_dir,'{}_{}'.format(model_type, train_result_name))
    if not os.path.exists(file_path):
        os.mknod(file_path)
    f= open(file_path,'w')

    f.write('train param:\n')
    f.write('model type:  {}\n'.format(model_type))
    f.write('data path:{}\n'.format(data_dir))

    # your image data file
    if 'net' in model_type:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                net_data_transforms[x]) for x in ['train', 'val']}
    # wrap your data and label into Tensor
    if 'vgg' in model_type:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x]) for x in ['train', 'val']}


    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    f.write('train data num:  {}\n'.format(dataset_sizes['train']))
    f.write('val data num:  {}\n'.format(dataset_sizes['val']))

    # use USE_CUDA or not
    # USE_CUDA = torch.cuda.is_available()
    # USE_CUDA = False

    # get model and replace the original fc layer with your fc layer
    if model_type == 'vgg16':
        model_ft = vggmodel.vgg16(pretrained=True, num_classes=2)
    if model_type == 'vgg11':
        model_ft = vggmodel.vgg11(pretrained=True, num_classes=2)
    if model_type == 'vgg9':
        model_ft = vggmodel.vgg9(pretrained=False, num_classes=2)
        model_ft.apply(weights_init)
        pretrained_dick=torch.load(model_param_location['vgg11'])
        model_dict=model_ft.state_dict()
        pretrained_dick={k:v for k,v in pretrained_dick.items() if k in model_dict}
        model_dict.update(pretrained_dick)
        model_ft.load_state_dict(model_dict)

    if model_type == 'net9':
        model_ft = mynet.net9(pretrained=False, num_classes=2)
        model_ft.apply(weights_init)
        #print(model_ft)

    if model_type == 'net7':
        model_ft = mynet.net7(pretrained=False, num_classes=2)
        model_ft.apply(weights_init)
        #print(model_ft)


    f.write('model ********\n')
    #pickle.dump(model_ft,f)
    for name, param in model_ft.named_parameters():
        f.write('{}:    {}\n'.format(name,param.size() ))

    if USE_CUDA:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           f=f,
                           num_epochs=epoch
                           )

    f.close()


