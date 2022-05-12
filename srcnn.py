import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 8
keep_prob = 1

# dataset loader
kg_f = np.load('/home/lllei/AI_localization/L05/F15/singobs/200040/kg_f_5y.npy')
kg_f = kg_f[1200:,None,:,:]
kg_t = np.load('/home/lllei/AI_localization/L05/F15/singobs/200040/kg_t_5y.npy')
kg_t = kg_t[1200:,None,:,:]

print('kg shape', kg_f.shape)

tensor_x = torch.Tensor(kg_f) # transform to torch tensor
tensor_y = torch.Tensor(kg_t)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)


# Implementation of CNN/ConvNet Model
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, 28, 28, 32)
        # Pool -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(21,9), stride=1, padding=0, bias=True),
            torch.nn.ReLU())
        # L2 ImgIn shape=(?, 14, 14, 32)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=(15,15), stride=1, padding=0, bias=True),
            torch.nn.ReLU())

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, kernel_size=(5,5), stride=1, padding=0, bias=True))

    def forward(self, x):
        W = 240
        H = 960
        kwd = 9
        kht = 21
        kwd1 = 15
        kht1 = 15
        kwd2 = 5
        kht2 = 5
        swd = 1
        sht = 1
        pwd = int((W - 1 - (W - kwd) / swd) // 2 + (W - 1 - (W - kwd1) / swd) // 2 + (W - 1 - (W - kwd2) / swd) // 2)
        pht = int((H - 1 - (H - kht) / sht) // 2 + (H - 1 - (H - kht1) / sht) // 2 + (H - 1 - (H - kht2) / sht) // 2)
        
        x = F.pad(x, (pwd, pwd, pht, pht), 'circular')
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


#instantiate CNN model
model = CNN()
model
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

learning_rate = 1e-5
criterion = torch.nn.MSELoss()    # Softmax is internally computed.
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

print('Training the Deep Learning network ...')
train_cost = []
train_accu = []

training_epochs = 5
total_batch = 6000 // batch_size

# print('Size of the training dataset is {}'.format(mnist_train.data.size()))
# print('Size of the testing dataset'.format(mnist_test.data.size()))
print('Batch size is : {}'.format(batch_size))
print('Total number of batches is : {0:2.0f}'.format(total_batch))
print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))

for epoch in range(training_epochs):
    avg_cost = 0
    for i, (batch_X, batch_Y) in enumerate(data_loader):
        X = Variable(batch_X)    # image is already size of (28x28), no reshape
        Y = Variable(batch_Y)    # label is not one-hot encoded

        optimizer.zero_grad() # <= initialization of the gradients
        
        # forward propagation
        x = X.to(device)
        hypothesis = model(x)
        y = Y.to(device)
        cost = criterion(hypothesis, y) # <= compute the loss function
        
        # Backward propagation
        cost.backward() # <= compute the gradient of the loss/cost function     
        optimizer.step() # <= Update the gradients
             
        # Print some performance to monitor the training
        prediction = hypothesis.data.max(dim=1)[1]
        train_accu.append(((prediction.data == y.data).float().mean()).item())
        train_cost.append(cost.item())   
        if i % 200 == 0:
            print("Epoch= {},\t batch = {},\t cost = {:.6e},\t accuracy = {}".format(epoch+1, i, train_cost[-1], train_accu[-1]))
       
        avg_cost += cost.data / total_batch

    print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))


print('Learning Finished!')