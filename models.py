import torch 
#32 layers
class Model(torch.nn.Module):
    def __init__(self, numClasses):
        super(Model, self).__init__()
        #this the convolution part
        self.conv1 = torch.nn.Conv2d(3, 32, 7)
        self.conv2 = torch.nn.Conv2d(32, 16, 6)
        self.conv3 = torch.nn.Conv2d(16, 8, 5)
        self.conv4 =  torch.nn.Conv2d(8, 4, 4)
        self.conv5 = torch.nn.Conv2d(4, 2, 3)
        self.flatten = torch.nn.Flatten()
        #this is regressor layer
        self.numClasses = numClasses
        self.input = torch.nn.Linear(3872, 1000)
        self.hidden1 = torch.nn.Linear(1000, 500)
        self.output = torch.nn.Linear(500, numClasses)


    def forward(self, x):
        print("forwarding")
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.flatten(out)
        out = self.input(out)
        out = self.hidden1(out)
        out = self.output(out)



