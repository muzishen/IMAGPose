import json
import random
from typing import List

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageChops
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class ResizeAspect(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):
        w, h = img.size
        if w / h > self.output_size[1] / self.output_size[0]:
            oh = self.output_size[0]
            ow = int(self.output_size[0] * w / h)
        else:
            ow = self.output_size[1]
            oh = int(self.output_size[1] * h / w)
        return img.resize((ow, oh), Image.BICUBIC)


def augmentation(images, transform, state=None):
    if state is not None:
        torch.set_rng_state(state)
    if isinstance(images, List):
        transformed_images = [transform(img) for img in images]
        ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
    else:
        ret_tensor = transform(images)  # (c, h, w)
    return ret_tensor

def concat_big_img(img_list,  width, height, state):

    scale_transform = transforms.Compose(
            [
                ResizeAspect((height, width)),
                transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip(),
            ]
        )

    img1 = augmentation(img_list[0], scale_transform, state[0])
    img2 = augmentation(img_list[1], scale_transform, state[1])
    img3 = augmentation(img_list[2], scale_transform, state[2])
    img4 = augmentation(img_list[3], scale_transform, state[3])


    width, height = img1.size

    if  len(img1.getbands()) == 1:
        final_image = Image.new('L', (width * 2, height * 2))
    else:
        final_image = Image.new('RGB', (width * 2, height * 2))

    # concat image
    final_image.paste(img1, (0, 0))
    final_image.paste(img2, (width, 0))
    final_image.paste(img3, (0, height))
    final_image.paste(img4, (width, height))
    return final_image

class HumanPoseDataset(Dataset):
    def __init__(
            self,
            width,
            height,
            img_scale=(1.0, 1.0),
            img_ratio=(0.9, 1.0),
            drop_ratio=0.1,
            json_file=['./data/fashion.json', './data/market1501.json'],

    ):
        super().__init__()
        if isinstance(json_file, str):
            with open(json_file, 'r') as file:
                self.data = json.load(file)

        elif isinstance(json_file, list):
            for file_path in json_file:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if not hasattr(self, 'data'):
                        self.data = data
                    else:
                        self.data.extend(data)
        else:
            raise ValueError("Input should be either a JSON file path (string) or a list")

        print('=========', len(self.data))


        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio


        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.ref_cond_transform = transforms.Compose(
            [
                ResizeAspect((height, width)),
                transforms.RandomCrop((height, width)),
                transforms.ToTensor(),
            ]
        )

        self.ref_vae_transform = transforms.Compose(
            [
                ResizeAspect((height, width)),
                transforms.RandomCrop((height, width)),
                transforms.ToTensor(),
            ]
        )
        self.drop_ratio = drop_ratio




    def __getitem__(self, index):
        image_meta = self.data[index]
        image_paths = [f for f in image_meta if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".webp") or f.endswith(".jpeg") or f.endswith(".bmp")]

        while True:
            batch_images = [random.choice(image_paths) for _ in range(4)]
            if len(set(batch_images)) != 1:
                break
        # read frames and kps
        person_pil_image_list = []
        pose_pil_image_list = []


        for img_path in batch_images:
            person_pil_image_list.append(Image.open(img_path).convert("RGB"))
            pose_pil_image_list.append(Image.open(img_path.replace('/image/', '/dwpose/')).convert("RGB"))

        # transform
        state1 = torch.get_rng_state()
        state2 = torch.get_rng_state()
        state3 = torch.get_rng_state()
        state4 = torch.get_rng_state()
        state = [state1, state2, state3, state4]

        # recover to big image
        person_pil_big_image =  concat_big_img(person_pil_image_list, self.width, self.height, state)
        pixel_values_person = self.pixel_transform(person_pil_big_image)

        # recover to big pose
        pose_pil_big_image = concat_big_img(pose_pil_image_list, self.width, self.height, state)
        pixel_values_pose = self.cond_transform(pose_pil_big_image)


        while True:
            random_list = [random.choice([0, 1]) for _ in range(4)]
            if random_list.count(0) != 4 and random_list.count(1) != 4:
                break

        #image + mask recover to big image
        image_mask_pil_image_list = [ImageChops.multiply(person_pil_image, Image.new('RGB', person_pil_image.size, (mask*255, mask*255, mask*255))) for person_pil_image, mask in zip(person_pil_image_list, random_list)]
        image_mask_big_image = concat_big_img(image_mask_pil_image_list, self.width, self.height, state)
        pixel_values_image_mask = self.pixel_transform(image_mask_big_image)

        # setting flag label
        white1 = Image.new("L", (self.width //8, self.height//8), 255)
        flag_label = [ImageChops.multiply(white1, Image.new('L', white1.size, (random_list[0]*255))),
                      ImageChops.multiply(white1, Image.new('L', white1.size, (random_list[1]*255))),
                        ImageChops.multiply(white1, Image.new('L', white1.size, (random_list[2]*255))),
                        ImageChops.multiply(white1, Image.new('L', white1.size, (random_list[3]*255)))]
        flag_label_mask_big_image = concat_big_img(flag_label, self.width//8, self.height//8, state)
        pixel_values_flag_label = self.cond_transform(flag_label_mask_big_image)

        # ref img
        index_list = [i for i, value in enumerate(random_list) if value == 1]
        random_index = random.choice(index_list)
        ">>>> add ref image for clip >>>>"
        clip_ref_img = self.clip_image_processor( images=person_pil_image_list[random_index], return_tensors="pt").pixel_values[0]
        ">>>> add ref image for vae >>>>"
        vae_ref_img = augmentation(person_pil_image_list[random_index], self.ref_vae_transform, state[random_index])
        sample = dict(
            pixel_values_person=pixel_values_person,
            pixel_values_pose=pixel_values_pose,
            pixel_values_image_mask= pixel_values_image_mask,
            pixel_values_flag_label = pixel_values_flag_label,
            clip_ref_img=clip_ref_img,
            vae_ref_img =vae_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.data)





if __name__ == "__main__":
    import tqdm

    dataset = HumanPoseDataset(
        width=512,
        height=768,
        img_scale=(1.0, 1.0),
        json_file=['./data/fashion.json', './data/market1501.json'],
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4
    )
    for _ in range(1):
        for batch in tqdm.tqdm(train_dataloader):
            pixel_values_vid = batch["pixel_values_person"]
            print(batch["pixel_values_person"].shape, batch['pixel_values_pose'].shape, batch['clip_ref_img'].shape, batch['pixel_values_image_mask'].shape, batch['pixel_values_flag_label'].shape)
