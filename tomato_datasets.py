# 数据集的组织形式类似是用lambing标注的关键点检测数据集
"""

dataset structure:

├── Image
│   ├── train:1.jpg...
│   └── val:1.json...
└── Label
    ├── train:2.jpg...
    └── val:2.json...
"""

import json
import os
import cv2
import torch
import torch.utils.data as data
import numpy as np
from typing import List, Tuple

class WFLWDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        super().__init__()
        self.img_root = os.path.join(root, "Images", "train" if train else "val")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        
        # 设置 JSON 注释文件夹路径
        self.anno_root = os.path.join(root, "Labels", "train" if train else "val")
        assert os.path.exists(self.anno_root), "path '{}' does not exist.".format(self.anno_root)

        # 加载所有图像文件名和对应的 JSON 文件
        self.img_paths: List[str] = []
        self.json_paths: List[str] = []
        for img_file in os.listdir(self.img_root):
            img_path = os.path.join(self.img_root, img_file)
            json_file = img_file.replace('.jpg', '.json')  # 假设图像文件是 .jpg 格式，JSON 文件对应相同的文件名
            json_path = os.path.join(self.anno_root, json_file)
            
            if os.path.exists(json_path):
                self.img_paths.append(img_path)
                self.json_paths.append(json_path)
        
        self.transforms = transforms
        self.keypoints: List[np.ndarray] = []
        self.face_rects: List[List[int]] = []

        # 读取所有 JSON 文件
        for json_path in self.json_paths:
            with open(json_path, "r") as f:
                annotation = json.load(f)
                keypoints = [shape['points'][0] for shape in annotation['shapes']]
                keypoints = np.array(keypoints, dtype=np.float32)

                # 如果 JSON 文件包含边界框信息，解析它；否则假设整张图像为人脸区域
                face_rect = [0, 0, annotation.get('imageWidth', 0), annotation.get('imageHeight', 0)]

                self.keypoints.append(keypoints)
                self.face_rects.append(face_rect)

    @staticmethod
    def collate_fn(batch_infos: List[Tuple[torch.Tensor, dict]]):
        imgs, ori_keypoints, keypoints, m_invs = [], [], [], []
        for info in batch_infos:
            imgs.append(info[0])
            ori_keypoints.append(info[1]["ori_keypoint"])
            keypoints.append(info[1]["keypoint"])
            m_invs.append(info[1]["m_inv"])

        imgs_tensor = torch.stack(imgs)
        keypoints_tensor = torch.stack(keypoints)
        ori_keypoints_tensor = torch.stack(ori_keypoints)
        m_invs_tensor = torch.stack(m_invs)

        targets = {"ori_keypoints": ori_keypoints_tensor,
                   "keypoints": keypoints_tensor,
                   "m_invs": m_invs_tensor}
        return imgs_tensor, targets

    def __getitem__(self, idx: int):
        img_bgr = cv2.imread(self.img_paths[idx], flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        target = {
            "box": self.face_rects[idx],  # 边界框信息
            "ori_keypoint": self.keypoints[idx],  # 原始关键点信息
            "keypoint": self.keypoints[idx]  # 预处理后的关键点信息
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    train_dataset = WFLWDataset("../TomatoSet", train=True)
    print("the length of train dataset is:", len(train_dataset))

    eval_dataset = WFLWDataset("../TomatoSet", train=False)
    print("the length of train dataset is:", len(eval_dataset))
