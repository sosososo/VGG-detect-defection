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
import mynet
from config import *
from visualize import make_dot



if __name__ == '__main__':

    # data_transform,
    # pay attention that the input of Normalize() is Tensor
    # and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image

    vgg_test_transform=transforms.Compose([
        transforms.Scale(vgg_input_size),  # 256 init
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    net_test_transform=transforms.Compose([
        transforms.Scale(net_input_size),  # 256 init
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if 'net' in model_type:
        test_dataset =datasets.ImageFolder(test_dir,net_test_transform)
    # wrap your data and label into Tensor
    if 'vgg' in model_type:
        test_dataset =datasets.ImageFolder(test_dir,vgg_test_transform)


    # wrap your data and label into Tensor
    test_dataloders=torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=False, 
                                                pin_memory=True
                                                )
    print(test_dataset)
#    print(test_dataloders)
    test_datasize = len(test_dataset)
    print('test size:')
    print(test_datasize)

    # use USE_CUDA or not
    # USE_CUDA = torch.cuda.is_available()
    # USE_CUDA = False

    # get model and replace the original fc layer with your fc layer


    if model_type == 'vgg16':
        model = vggmodel.vgg16(pretrained=False, num_classes=2)
    if model_type == 'vgg11':
        model = vggmodel.vgg11(pretrained=False, num_classes=2)
    if model_type == 'vgg11_bn':
        model = vggmodel.vgg11_bn(pretrained=True, num_classes=2)
    if model_type == 'vgg9':
        model = vggmodel.vgg9(pretrained=False, num_classes=2)
    if model_type == 'net9':
        model = mynet.net9(pretrained=False, num_classes=2)
    if model_type == 'vgg13':
        model = vggmodel.vgg13(pretrained=False, num_classes=2)
    if model_type == 'vgg19':
        model = vggmodel.vgg19(pretrained=False, num_classes=2)

    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)


    trained_dict=torch.load(
            os.path.join(param_dir, '{}_{}'.format(model_type, param_name)))

    model.load_state_dict(trained_dict)

    if USE_CUDA:
        model = model.cuda()


    # set the model to eval model
    model.eval()



    tn=0.0
    fn=0.0
    tp=0.0
    fp=0.0
    file_path=os.path.join(result_dir,'{}_{}'.format(model_type, result_name))
    if not os.path.exists(file_path):
        os.mknod(file_path)
    f= open(file_path,'w')
    f.write('{} result:\n\n'.format(os.path.join(param_dir,param_name)))
    f.write('target  pred  img_path\n')
    index = 0
    print('start test ******')
    since = time.time()
    for data in test_dataloders:
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        if USE_CUDA:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs=model(inputs)
        _, preds = torch.max(outputs.data, 1)
        targets=labels.data
        for (pred,target) in zip(preds,targets):
            img_path =test_dataset.imgs[index][0]
            #img_path = 'path'
            index +=1
            f.write('  {}     {}        {}\n'.format(target,pred,img_path))
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

    time_end=time.time() -since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_end // 60, time_end % 60))
    print('Testing complete in {} ms'.format(
        (int(round(time_end * 1000)/index))))
    f.write('\n')
    f.write('Testing complete in {:.0f}m {:.0f}s\n'.format(
        time_end // 60, time_end % 60))
    f.write('Testing complete in {} ms'.format(
        (int(round(time_end * 1000)/index))))
    f.write('pos num: {}\ntp:  {}\nfn:  {}\ntn:  {}\nfp:  {}'.format((tp+fn+tn+fp),tp,fn,tn,fn))
    print('pos num: {}\ntp:  {}\nfn:  {}\ntn:  {}\nfp:  {}'.format((tp+fn+tn+fp),tp,fn,tn,fn))
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
    f.close()



