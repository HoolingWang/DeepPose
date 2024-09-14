import os

import torch
import numpy as np
from PIL import Image

import transforms
from model import create_deep_pose_model
from utils import draw_keypoints


def main():
    img_hw = [256, 256]
    num_keypoints = 13
    img_path = "/media/fys/1e559c53-a2a9-468a-a638-74933f8e167c/whn/project/img_kpt/DeepPose/tomato_20220725061.jpg"
    weights_path = "./weights/model_weights_209.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.AffineTransform(scale_prob=0., rotate_prob=0., shift_prob=0., fixed_size=img_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = np.array(Image.open(img_path))
    h, w, c = img.shape
    target = {"box": [0, 0, w, h]}
    img_tensor, target = transform(img, target=target)
    # expand batch dimension
    img_tensor = img_tensor.unsqueeze_(0)

    # create model
    model = create_deep_pose_model(num_keypoints=num_keypoints)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location="cpu")["model"])
    model.to(device)

    # prediction
    model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type):
            pred = torch.squeeze(model(img_tensor.to(device))).reshape([-1, 2]).cpu().numpy()
            print(pred)

        wh_tensor = np.array(img_hw[::-1], dtype=np.float32).reshape([1, 2])
        pred = pred * wh_tensor  # rel coord to abs coord
        pred = transforms.affine_points_np(pred, target["m_inv"].numpy())
        draw_keypoints(img, coordinate=pred, save_path="predict.jpg", radius=10)


if __name__ == '__main__':
    main()
