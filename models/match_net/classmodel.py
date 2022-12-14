import torch
from torch import nn
from torch.autograd import Variable

class AudioClassificationNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kinds):
        super(AudioClassificationNet, self).__init__()
        self.input_size = input_size
        self.kinds = kinds
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.kinds)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = torch.relu(self.fc1(feat))
        feat = torch.relu(self.fc2(feat))
        return feat


class VideoClassificationNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kinds):
        super(VideoClassificationNet, self).__init__()
        self.input_size = input_size
        self.kinds = kinds
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.kinds)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = torch.relu(self.fc1(feat))
        feat = torch.relu(self.fc2(feat))
        return feat


class ClassificationNet(torch.nn.Module):
    def __init__(self, Vinput_size = 512, Ainput_size = 128, output_size = 64, layers_num = 3, dropout_ratio = 0.3):
        super(ClassificationNet, self).__init__()
        self.Vinput_size = Vinput_size
        self.Ainput_size = Ainput_size
        self.output_size = output_size
        self.layers_num = layers_num
        self.init_params()

        self.AClassNet = AudioClassificationNet(self.Ainput_size, self.output_size, 28)
        self.VClassNet = VideoClassificationNet(self.Vinput_size, self.output_size, 28)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, Vfeat, Afeat):
        h_0 = Variable(torch.zeros(self.layers_num, Afeat.size(0),
                                   self.output_size),
                       requires_grad=False)
        c_0 = Variable(torch.zeros(self.layers_num, Afeat.size(0),
                                   self.output_size),
                       requires_grad=False)

        if Vfeat.is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        alogits = 0
        vlogits = 0
        for i in range(10):
            alogits += self.AClassNet(Afeat[:, :, i])
            vlogits += self.VClassNet(Vfeat[:, :, i])
        alogits /= 10
        vlogits /= 10
        if self.training:
            return alogits, vlogits
            
        else:
            a_predict_var = torch.argmax(alogits, dim = 1)
            v_predict_var = torch.argmax(vlogits, dim = 1)
            
            output = []
            for i in range(len(a_predict_var)):
                if a_predict_var[i] != v_predict_var[i]:
                    output.append(torch.tensor([1000, 0]))
                else: 
                    output.append(torch.tensor([0, 1000]))

            return torch.stack(output)

