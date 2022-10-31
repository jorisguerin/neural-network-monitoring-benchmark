import augly.image as imaugs

accepted_datasets = ["cifar10_train", "cifar100_train", "svhn_train",
                     "cifar10_test", "cifar100_test", "svhn_test",
                     "tiny_imagenet_test", "lsun_test"]

mean_transform = {
    "resnet": [0.4914, 0.4822, 0.4465],
    "densenet": [125.3 / 255, 123.0 / 255, 113.9 / 255]
}
std_transform = {
    "resnet": [0.2023, 0.1994, 0.2010],
    "densenet": [63.0 / 255, 62.1 / 255.0, 66.7 / 255.0]
}

additional_transforms = {
    "brightness": imaugs.Brightness(factor=3),
    "blur": imaugs.Blur(radius=2),
    "pixelization": imaugs.Pixelization(ratio=0.5),
    "shuffle_pixels":  imaugs.ShufflePixels(factor=0.1)
}

accepted_attacks = ["fgsm", "deepfool", "pgd"]

datasets_path = "./Data"
path_tinyImagenet = datasets_path + "/Imagenet_resize/"
path_lsun = datasets_path + "/LSUN_resize/"

url_tinyImagenet = "https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz?dl=1"
url_lsun = "https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1"
