import gdown

if __name__ == '__main__':
    model_path_1 = 'model_hw1_1.pth'
    url_1 = 'https://drive.google.com/u/0/uc?id=1kmsjKczimx4p-UNH4g1Y-sWxz-_aJipv&export=download'
    gdown.download(url_1, model_path_1)
    model_path_2 = 'model_hw1_2.pth'
    url_2 = 'https://drive.google.com/u/0/uc?id=1r5DDQ3LZEZ77-bIwrS0YJtyCcSYnJAqO&export=download'
    gdown.download(url_2, model_path_2)