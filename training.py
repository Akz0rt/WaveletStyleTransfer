from HaarPool import Encoder, Decoder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def gram_matrix(input):
    '''
    Обчислення матриці Грама для вхідного тензора
    '''
    channels, height, width = input.size()
    features = input.view(channels, height * width)
    gram_matrix = torch.mm(features, features.t())
    return gram_matrix.div(channels * height * width)


def style_loss_function(input, style_targets):
    style_loss = 0.0
    for input_feat, target_feat in zip(input, style_targets):
        input_gram = gram_matrix(input_feat)
        target_gram = gram_matrix(target_feat)
        loss = criterion(input_gram, target_gram)
        style_loss += loss
    return style_loss


# training_process
def train():
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    encoder.to(device)
    decoder.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        print(epoch+1)
        for data in trainloader:
            i += 1
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(trainloader)}]')
            inputs, targets = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            skips = {}
            style_targets = []
            encoded = inputs
            for level in [1, 2, 3, 4]:
                encoded = encoder.encode(encoded,skips,level)
                style_targets.append(encoded)
            decoded = encoded
            for level in [4, 3, 2, 1]:
                decoded = decoder.decode(decoded, skips, level)

            content_loss = criterion(decoded, inputs)
            style_loss = style_loss_function(decoded, inputs)

            total_loss = 0.6*content_loss + 0.4*style_loss

            total_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # stat
            running_loss += total_loss.item()
            if i % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(trainloader)}], Loss: {running_loss/200:.4f}')
                running_loss = 0.0


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # download CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=12)

    encoder = Encoder()
    print('enc_loaded')
    decoder = Decoder()
    print('dec_loaded')

    criterion = nn.MSELoss()
    print('lossf_loaded')

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    print('opt_loaded')

    train()

    torch.save(encoder.state_dict(), 'data/encoder.pth')
    torch.save(decoder.state_dict(), 'data/decoder.pth')
