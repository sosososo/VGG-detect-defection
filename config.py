
USE_CUDA = True

momentum = 0.95

weight_decay = 0.0005

lr = 0.001

batch_size = 48

data_dir = '/home/sj/workspacce/vgg16/data'

test_dir = '/home/sj/workspacce/vgg16/data/val'

param_dir = '/home/sj/workspacce/vgg16/param-bn'

param_name = 'best_param-bn.pth'


result_dir = '/home/sj/workspacce/vgg16'
result_name = 'val_result.txt'
train_result_name = 'train_record.txt'

num_classes = 2

shuffle = True

print_freq = 400

net_input_size = 96 # 64 for 2*2  96 for 3*3

vgg_input_size = 224

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

weight = [ 1 , 3]

epoch=15


model_param_location={
    'alexnet': '/home/sj/workspacce/vgg16/param/alexnet-owt-4df8aa71.pth',
    'vgg11':'/home/sj/workspacce/vgg16/param/vgg11-bbd30ac9.pth',
    'vgg11_bn':'/home/sj/workspacce/vgg16/param/vgg11_bn-6002323d.pth',
    'vgg13': '/home/sj/workspacce/vgg16/param/vgg13-c768596a.pth',
    'vgg16': '/home/sj/workspacce/vgg16/param/vgg16-397923af.pth',
    'vgg19': '/home/sj/workspacce/vgg16/param/vgg19-dcbb9e9d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_type='vgg19'