import gdown

if __name__ == '__main__':

    save_path = 'logs/nerf_synthetic/dvgo_hotdog/coarse_last.tar'
    url = 'https://drive.google.com/uc?id=1YSo7AhXLhrdpgWDcow4ab9Ws5_XPgMrv&export=download'
    gdown.download(url, save_path)

    save_path = 'fine_last.tar'
    url = 'https://drive.google.com/uc?id=1UyOD3g2zXoOZo6s5BztyWiYwqXyFrQoq&export=download'
    gdown.download(url, save_path)

    save_path = 'best_model_C.pt'
    url = 'https://drive.google.com/uc?id=1SG8d5PqyiucDBZEGOJnlpUTp90cjKery&export=download'
    gdown.download(url, save_path)
    







