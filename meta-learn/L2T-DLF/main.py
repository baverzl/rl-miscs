import torch
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DLoss(nn.Module):

    def __init__(self, num_classes=10, state_size=13, hidden=10):
        super(DLoss, self).__init__()
        self.num_classes = num_classes
        self.V = nn.Linear(state_size, hidden, bias=False)
        self.W = nn.Linear(hidden, num_classes * num_classes, bias=False)

    def __get_phi_matrix(self, s):
        return self.W(F.softmax(self.V(s))).view(self.num_classes, self.num_classes)

    def forward(self, output, target, state):
        target_ = F.one_hot(target).float()
        batch_size = target.shape[0]
        # 64 x 10 x 10
        phi_ = self.__get_phi_matrix(state).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        weighted_prob = torch.matmul(phi_, output.unsqueeze(dim=2))

        ##import pdb; pdb.set_trace()
        target_ = target_.unsqueeze(dim=1)
        output = torch.matmul(target_, weighted_prob) # bs x 1 x num_classes , bs x num_classes x 1

        # print(torch.einsum("bij, bjk -> bik", (phi_, torch.log(output).unsqueeze(dim=2))))
        return torch.mean(-F.sigmoid(output))


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output.shape)

        state = torch.randn([13])
        loss = DLoss()(output, target, state)

        import pdb; pdb.set_trace()

        # loss.backward()
         # instead of using loss.backward(), use torch.autograd.grad() to compute gradients
        loss_grads = grad(loss, model.parameters(), create_graph=True)

        # compute the second order derivative w.r.t. each parameter
        d2loss = []
        for grd in loss_grads:
            grd = grd.view(-1)
            # grd = grd.squeeze()
            for grd_ in grd:
                grad2 = grad(grd_, model.parameters(), create_graph=True)
                d2loss.append(grad2)
        
        print(d2loss)

        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

batch_size = 64
lr=0.001
momentum=0.999
epochs=100

if __name__ == '__main__':

    device = torch.device("cpu")

    kwargs = {}

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

