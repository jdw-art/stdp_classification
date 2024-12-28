import numpy as np
import torch
import pytorch_ssim
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn.metrics.pairwise import cosine_similarity
from SpykeTorch import utils
from SpykeTorch import functional as sf


def compute_average_ssim(images, targets):
    # 转换PyTorch张量为NumPy数组
    images = images.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    # 初始化一个列表来存储每对图像的SSIM
    ssim_values = []

    # 确保图像的形状是[batch_size, c, h, w]
    assert images.shape == targets.shape

    batch_size, channels, height, width = images.shape

    # 遍历每个样本
    for i in range(batch_size):
        image = images[i].squeeze()
        target = targets[i].squeeze()

        # 计算单个图像的SSIM
        ssim = compare_ssim(image, target, data_range=1.0)

        # 将SSIM值添加到列表中
        ssim_values.append(ssim)

    # 计算SSIM的平均值
    average_ssim = np.mean(ssim_values)

    return average_ssim

# def compute_average_ssim(images, targets):
#     if torch.cuda.is_available():
#         images = images.cuda()
#         targets = targets.cuda()
#
#     average_ssim = pytorch_ssim.ssim(images, targets)
#
#     return average_ssim


def compute_ssim(outputs, targets):
    if torch.cuda.is_available():
        outputs = outputs.cuda()
        targets = targets.cuda()

    image_size = outputs.size(0)

    ssim_value = 0

    for i in range(image_size):
        # ssim_value += pytorch_ssim.ssim(outputs[i], targets[i])
        ssim_value += compute_average_ssim(outputs[i], targets[i])

    ssim = ssim_value / image_size

    return ssim


def compute_accuracy_ssim(outputs, targets, threshold=0.8):
    if torch.cuda.is_available():
        outputs = outputs.cuda()
        targets = targets.cuda()

    image_size = outputs.size(0)
    correct_count = 0

    for i in range(image_size):
        ssim_value = pytorch_ssim.ssim(outputs[i], targets[i])
        if ssim_value >= threshold:
            correct_count += 1

    accuracy = correct_count / image_size

    return accuracy



def compute_average_psnr(images, targets):
    # 转换PyTorch张量为NumPy数组
    images = images.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    # 初始化一个列表来存储每对图像的SSIM
    psnr_values = []

    # 确保图像的形状是[batch_size, c, h, w]
    assert images.shape == targets.shape

    batch_size, channels, height, width = images.shape

    # 遍历每个样本
    for i in range(batch_size):
        image = images[i].squeeze()
        target = targets[i].squeeze()

        image_shape = image.shape
        target_shape = target.shape

        # 计算单个图像的SSIM
        psnr = compare_psnr(image, target, data_range=1.0)

        # 将SSIM值添加到列表中
        psnr_values.append(psnr)

    # 计算SSIM的平均值
    average_psnr = np.mean(psnr_values)

    return average_psnr


def compute_average_cosine_similarity(images, targets):
    # 转换PyTorch张量为NumPy数组
    images = images.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    # 初始化一个列表来存储每对图像的余弦相似度
    cosine_sim_values = []

    # 确保图像的形状是[batch_size, c, h, w]
    assert images.shape == targets.shape

    batch_size, channels, height, width = images.shape

    # 遍历每个样本
    for i in range(batch_size):
        image = images[i].reshape(-1)
        target = targets[i].reshape(-1)

        # 计算单个图像的余弦相似度
        cosine_sim = cosine_similarity([image], [target])[0, 0]

        # 将余弦相似度值添加到列表中
        cosine_sim_values.append(cosine_sim)

    # 计算余弦相似度的平均值
    average_cosine_sim = np.mean(cosine_sim_values)

    return average_cosine_sim



# 添加椒盐噪声
def salt_pepper_noise(image, ratio, noise_range=(0.0, 1.0)):
    """
    Add salt and pepper noise to a torch tensor representing an image.

    Args:
        image (torch.Tensor): Input image tensor with pixel values in the range [0, 1].
        ratio (float): Probability of adding salt and pepper noise to each pixel.
        noise_range (tuple): Range for the noise values (min, max).

    Returns:
        torch.Tensor: Image tensor with salt and pepper noise.
    """
    output = image.clone()

    # Generate random noise mask
    noise_mask = torch.rand(image.shape) < ratio

    # Generate random noise values in the specified range
    noise_values = torch.rand(image.shape) * (noise_range[1] - noise_range[0]) + noise_range[0]

    # Apply salt and pepper noise
    salt_mask = noise_mask & (torch.rand(image.shape) > 0.5)
    pepper_mask = noise_mask & ~salt_mask

    # Set salt and pepper pixels to the random noise values
    output[salt_mask] = noise_values[salt_mask]
    output[pepper_mask] = noise_values[pepper_mask]

    return output


# 掩蔽图像
def apply_mask(image, mask_ratio):
    """
    Apply a mask to a normalized image tensor.

    Args:
        image (torch.Tensor): Input image tensor with pixel values in the range [0, 1].
        mask_ratio (float): Ratio of the image to be masked, ranging from 0.0 to 1.0.

    Returns:
        torch.Tensor: Image tensor with the specified mask applied.
    """
    output = image.clone()

    # Calculate the height of the masked region
    mask_height = int(image.shape[-2] * mask_ratio)

    # Apply the mask to the lower part of the image
    output[:, :, -mask_height:] = 0.0

    return output


# def gaussian_perturb_image(img, sigma=0.1):
#     #print(img.shape)
#     if len(img.shape) != 1:
#         total_img_len = np.prod(np.array(img.shape))
#         img = img.reshape(total_img_len)
#     N = len(img)
#     variance = torch.tensor(np.identity(N) * sigma).float()
#     perturb = torch.normal(0, sigma, size=[N, ])
#     noisy_image = torch.clamp(torch.abs(img + perturb), 0, 1)
#     return noisy_image.reshape([28, 28])

def gaussian_perturb_image(img, sigma=0.1):
    # 方式一，通过协方差矩阵生成高斯噪声

    # batch_size, channel, h, w = img.size()
    # # 将输入图像展平
    # if len(img.shape) != 1:
    #     img = img.reshape(-1)
    # N = len(img)
    # # 创建协方差矩阵
    # variance = torch.tensor(np.identity(N) * sigma).float()
    # # 生成高斯扰动
    # perturb = torch.normal(0, sigma, size=[N, ])
    # # 添加扰动并夹紧
    # noisy_image = torch.clamp(torch.abs(img + perturb), 0, 1)
    # # 重新塑形为原始图像形状
    # return noisy_image.reshape([batch_size, channel, h, w])

    # 方式二：通过随机噪声模拟高斯噪声
    # 保持输入形状
    batch_size, channel, h, w = img.size()
    # 生成与图像形状相同的高斯噪声
    perturb = torch.randn_like(img) * sigma
    # 添加噪声并夹紧值范围
    noisy_image = torch.clamp(img + perturb, 0, 1)
    return noisy_image

def save_tensor_to_file(tensor, file_path):
    """
    将形状为 [1, 100, 100] 的 tensor 保存到本地文件。
    :param tensor: 要保存的 PyTorch tensor
    :param file_path: 保存的文件路径
    """
    if tensor.shape != (1, 100, 100):
        raise ValueError("Tensor must have shape [1, 100, 100]")
    torch.save(tensor, file_path)

def load_tensor_from_file(file_path):
    """
    从文件读取 tensor 并确保其形状为 [1, 100, 100]。
    :param file_path: 读取的文件路径
    :return: 形状为 [1, 100, 100] 的 PyTorch tensor
    """
    tensor = torch.load(file_path)
    if tensor.shape != (1, 100, 100):
        raise ValueError("Loaded tensor does not have shape [1, 100, 100]")
    return tensor

# DOG滤波器
def dog_filter(image, filter, timesteps):
    image = image * 255
    image = image.float()
    image = image.unsqueeze(0).to('cpu')
    image = filter(image)
    image = sf.local_normalization(image, 8)
    temporal_transform = utils.Intensity2Latency(timesteps)
    image = temporal_transform(image).sign().byte()
    return image

class S1Transform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt+=1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()

if __name__ == "__main__":

    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='/usr/common/datasets/MNIST', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    # # 获取一个batch的图像
    # dataiter = iter(dataloader)
    # images, labels = dataiter.next()

    images, labels = None, None
    for i, sample in enumerate(dataloader):
        images, labels = sample
        break

    # 获取一张图像
    image = images[0, 0, :, :]

    # 显示原始图像
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.show()

    # 添加高斯噪声
    noisy_image = gaussian_perturb_image(image, sigma=0.1)
    # noisy_image = apply_mask(image, mask_ratio=0.1)

    # 显示添加噪声后的图像
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image')
    plt.show()


