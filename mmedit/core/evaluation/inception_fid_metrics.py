import mmcv
import torch


@torch.no_grad()
def extract_features(loader, inception, device):
    # pbar = ProgressBar(len(loader))
    pbar = mmcv.ProgressBar(len(loader))

    feature_list = []

    for data in loader:
        img = data['img']
        pbar.update()
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features
