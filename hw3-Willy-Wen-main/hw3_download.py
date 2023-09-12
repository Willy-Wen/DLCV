import gdown

if __name__ == '__main__':

    save_path = 'checkpointV3.pth'
    url = 'https://drive.google.com/uc?id=14ke0_FFc-W7ZJj25ZEEacYGQx8sJpfuP&export=download'
    gdown.download(url, save_path)
