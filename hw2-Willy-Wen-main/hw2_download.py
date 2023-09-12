import gdown

if __name__ == '__main__':

    model_1_path = 'generator.pth'
    url_1 = 'https://drive.google.com/u/0/uc?id=1zGIgWN8vGOvce_uLH9-1znS4z3_TSF72&export=download'
    gdown.download(url_1, model_1_path)

    model_2_path = 'cddpm.pth'
    url_2 = 'https://drive.google.com/u/0/uc?id=14q7gfsZpRBW_VZMQP0LL59VPUhjPZYoQ&export=download'
    gdown.download(url_2, model_2_path)

    model_3_path = 'feature_extractor_mnistm2svhn.pth'
    url_3 = 'https://drive.google.com/u/0/uc?id=1wm6XjjU63Mn95OsfWIoOwZHYae9mpqYH&export=download'
    gdown.download(url_3, model_3_path)

    model_4_path = 'label_predictor_mnistm2svhn.pth'
    url_4 = 'https://drive.google.com/u/0/uc?id=16XpHLrNZjWqbQxUSKUTeU4Z9Fzu0HTXX&export=download'
    gdown.download(url_4, model_4_path)

    model_5_path = 'feature_extractor_mnistm2usps.pth'
    url_5 = 'https://drive.google.com/u/0/uc?id=1UOO1b1zMJ7pXOJdYmcn621hzdvDuszuU&export=download'
    gdown.download(url_5, model_5_path)

    model_6_path = 'label_predictor_mnistm2usps.pth'
    url_6 = 'https://drive.google.com/u/0/uc?id=10CJRlylqE5EsHbO8puiBVmtOJ997hdLV&export=download'
    gdown.download(url_6, model_6_path)
