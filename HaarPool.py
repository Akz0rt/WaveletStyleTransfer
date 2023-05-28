import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils.core import feature_wct
from utils.io import open_image, load_segment, compute_label_info
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def haar_transform(in_channels, pool = True):
    '''базисні вектори'''
    L = 1/np.sqrt(2)*np.ones((1,2))
    H = 1/np.sqrt(2)*np.ones((1,2))
    H[0,0] = -1*H[0,0]
    '''формування матриці перетворення Хаара'''
    LL = np.transpose(L)*L
    LH = np.transpose(L)*H
    HL = np.transpose(H)*L
    HH = np.transpose(H)*H

    filter_LL = torch.from_numpy(LL).unsqueeze(0)
    filter_LH = torch.from_numpy(LH).unsqueeze(0)
    filter_HL = torch.from_numpy(HL).unsqueeze(0)
    filter_HH = torch.from_numpy(HH).unsqueeze(0)

    if pool:
        conv = nn.Conv2d
    else:
        conv = nn.ConvTranspose2d
    '''зменшення розмірності данних із збереженням кількості каналів тензора'''
    LL = conv(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = conv(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = conv(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = conv(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False
    '''двовимірне вейвлет-перетворення Хаара'''
    LL.weight.data.copy_(filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1))
    LH.weight.data.copy_(filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1))
    HL.weight.data.copy_(filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1))
    HH.weight.data.copy_(filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1))

    return LL, LH, HL, HH

class HaarPool2D(nn.Module):
    def __init__(self, in_channels):
        super(HaarPool2D, self).__init__()
        self.LL, self.LH, self.HL, self.HH = haar_transform(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class HaarUnpool2D(nn.Module):
    def __init__(self, in_channels):
        super(HaarUnpool2D, self).__init__()
        self.LL, self.LH, self.HL, self.HH = haar_transform(in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3,3,1)
        self.conv1_1 = nn.Conv2d(3,64,3)
        self.conv1_2 = nn.Conv2d(64,64,3)
        self.pool1 = HaarPool2D(64)

        self.conv2_1 = nn.Conv2d(64,128,3)
        self.conv2_2 = nn.Conv2d(128,128,3)
        self.pool2 = HaarPool2D(128)

        self.conv3_1 = nn.Conv2d(128,256,3)
        self.conv3_2 = nn.Conv2d(256,256,3)
        self.conv3_3 = nn.Conv2d(256,256,3)
        self.conv3_4 = nn.Conv2d(256,256,3)
        self.pool3 = HaarPool2D(256)

        self.conv4_1 = nn.Conv2d(256,512,3)

    def forward(self,x):
        skips = {}
        for level in [1,2,3,4]:
            x = self.encode(x, skips, level)

        return x

    def encode(self, x, skips, level):
        if level == 1:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            out = self.relu(self.conv1_2(self.pad(out)))
            skips['conv1_2'] = out
            LL, LH, HL, HH = self.pool1(out)
            skips['pool1'] = [LH, HL, HH]
            return LL
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            out = self.relu(self.conv2_2(self.pad(out)))
            skips['conv2_2'] = out
            LL, LH, HL, HH = self.pool2(out)
            skips['pool2'] = [LH, HL, HH]
            return LL
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            out = self.relu(self.conv3_2(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))
            skips['conv3_4'] = out
            LL, LH, HL, HH = self.pool3(out)
            skips['pool3'] = [LH, HL, HH]
            return LL
        elif level == 4:
            return self.relu(self.conv4_1(self.pad(x)))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512,256,3)

        self.unpool3 = HaarUnpool2D(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_1 = nn.Conv2d(256, 128, 3)

        self.unpool2 = HaarUnpool2D(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.conv2_1 = nn.Conv2d(128, 64, 3)

        self.unpool1 = HaarUnpool2D(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.conv1_1 = nn.Conv2d(64, 3, 3)

    def forward(self, x, skips):
        for level in [4,3,2,1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            LH, HL, HH = skips['pool3']
            original = skips['conv3_4']
            out = self.unpool3(out, LH, HL, HH, original)
            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            LH ,HL, HH = skips['pool2']
            original = skips['conv2_2']
            out = self.unpool2(out, LH, HL, HH, original)
            return self.relu(self.conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            original = skips['conv1_2']
            out = self.unpool1(out, LH, HL, HH, original)
            return self.relu(self.conv1_2(self.pad(out)))
        else:
            return self.conv1_1(self.pad(x))

class NST:
    def __init__(self):

        self.transfer_at = set(['encoder','skip'])

        self.device = torch.device('cpu')
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.encoder.load_state_dict(torch.load('data/encoder.pth', map_location=lambda storage, loc: storage), strict=False)
        self.decoder.load_state_dict(torch.load('data/decoder.pth', map_location=lambda storage, loc: storage), strict=False)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_features(self, x):
        skips = {}
        features = {'encoder': {}, 'decoder': {}}
        for level in [1,2,3,4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                features['encoder'][level] = x
        return features, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feats, content_skips = content, {}
        style_feats, style_skips = self.get_features(style)

        encoding_level = [1,2,3,4]
        skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1,2,3,4]:
            content_feats = self.encode(content_feats, content_skips, level)
            if 'encoder' in self.transfer_at and level in encoding_level:
                content_feats = feature_wct(content_feats, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
        if 'skip' in self.transfer_at:
            for skip in skip_level:
                for i in [0,1,2]:
                    content_skips[skip][i] = feature_wct(content_skips[skip][i], style_skips[skip][i],
                                                         content_segment, style_segment,
                                                         label_set, label_indicator,
                                                         alpha=alpha, device=self.device)
        for level in [4,3,2,1]:
            content_feats = self.decode(content_feats, content_skips, level)
        return content_feats

    
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    content_file = askopenfilename(title='Оберіть зображення контенту')
    style_file = askopenfilename(title='Оберіть зображення стилю')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content = open_image(content_file, 1000).to(device)
    style = open_image(style_file, 1000).to(device)
    content_segment = load_segment(None, 224)
    style_segment = load_segment(None, 224)
    model = NST()
    with torch.no_grad():
        stylized_image = model.transfer(content, style, content_segment, style_segment, alpha=0.2)
    save_image(stylized_image.clamp(0,1), 'stylized_image.png', padding=0)

