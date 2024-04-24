import glob
import os
import random
import numpy as np

import skimage
import torch
import torch.utils.data as data
from torchvision import transforms
import PIL.Image as pil

from mde.data.utils import generate_depth_map, pil_loader


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
     data_path: path to day drive
    height
    width
    frame_idxs: how many frames upward
    num_scales,
    drives=[9], which drives to load 
    is_train=False,
    img_ext=".jpg",
    
    """

    def __init__(
        self,
        data_path,
        height,
        width,
        frame_idxs,
        num_scales,
        drives=[9],
        is_train=False,
        img_ext=".jpg",
    ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.drive_directories = [f"{data_path.split('/')[-1]}_drive_{'{:04d}'.format(drive)}_sync" for drive in drives]
        self.filenames=dict()
        self.frames=[]
        for drive_dir in self.drive_directories:
            self.filenames[drive_dir]=dict()
            for side in [2,3]:
                self.filenames[drive_dir][side]= sorted(
                    glob.glob(
                        os.path.join(
                            data_path,
                            drive_dir,
                            f'image_0{side}','data','*.jpg'
                            )),
                            key=lambda x: int(x.split('/')[-1].split('.')[0]
                                )
                            )
                self.frames.extend(self.filenames[drive_dir][side]) 
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = pil.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp
            )

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        path = self.frames[index].split('/')
        folder = path[-4]

        if len(path) == 3:
            frame_index = int(path[-1].replace('.jpg',''))
        else:
            frame_index = 0

        side = str(int(path[-3].split('_')[-1]))
        for i in self.frame_idxs:
            inputs[("color", frame_index+i, -1)] = self.get_color(
                index,do_flip
            )

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2**scale)
            K[1, :] *= self.height // (2**scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.Tensor(K)
            inputs[("inv_K", scale)] = torch.Tensor(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", frame_index+i, -1)]
            del inputs[("color_aug", frame_index+i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, index):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        path = self.frames[0].split('/')
        scene_name = path[-4]
        frame_index = int(path[-1].replace('.jpg',''))

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        return os.path.isfile(velo_filename)

    def get_color(self, index,do_flip):
        color = self.loader(self.frames[index])

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        return self.filenames[folder][side][frame_index]

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path)

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt,
            self.full_res_shape[::-1],
            order=0,
            preserve_range=True,
            mode="constant",
        )

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


# class KITTIOdomDataset(KITTIDataset):
#     """KITTI dataset for odometry training and testing"""

#     def __init__(self, *args, **kwargs):
#         super(KITTIOdomDataset, self).__init__(*args, **kwargs)

#     def get_image_path(self, folder, frame_index, side):
#         f_str = "{:06d}{}".format(frame_index, self.img_ext)
#         image_path = os.path.join(
#             self.data_path,
#             "sequences/{:02d}".format(int(folder)),
#             "image_{}".format(self.side_map[side]),
#             f_str,
#         )
#         return image_path


# class KITTIDepthDataset(KITTIDataset):
#     """KITTI dataset which uses the updated ground truth depth maps"""

#     def __init__(self, *args, **kwargs):
#         super(KITTIDepthDataset, self).__init__(*args, **kwargs)

#     def get_image_path(self, folder, frame_index, side):
#         f_str = "{:010d}{}".format(frame_index, self.img_ext)
#         image_path = os.path.join(
#             self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str
#         )
#         return image_path

#     def get_depth(self, folder, frame_index, side, do_flip):
#         f_str = "{:010d}.png".format(frame_index)
#         depth_path = os.path.join(
#             self.data_path,
#             folder,
#             "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
#             f_str,
#         )

#         depth_gt = pil.open(depth_path)
#         depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
#         depth_gt = np.array(depth_gt).astype(np.float32) / 256

#         if do_flip:
#             depth_gt = np.fliplr(depth_gt)

#         return depth_gt