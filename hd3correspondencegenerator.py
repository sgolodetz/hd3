import cv2
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import data.flowtransforms as transforms

from argparse import ArgumentParser

from hd3model import HD3Model


class HD3CorrespondenceGenerator:
    # CONSTRUCTORS

    def __init__(self, *, task: str, encoder: str, decoder: str, corr_range: [int], context: bool, model_path: str):
        # Create the model.
        logging.info("Creating model")
        self.__model = HD3Model(task, encoder, decoder, corr_range, context).cuda()
        # logging.info(self.__model)
        logging.info("Created model")

        self.__model = torch.nn.DataParallel(self.__model).cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

        # Load the checkpoint.
        logging.info("Loading checkpoint {}".format(model_path))
        checkpoint = torch.load(model_path)
        self.__model.load_state_dict(checkpoint["state_dict"], strict=True)
        logging.info("Loaded checkpoint {}".format(model_path))

        # Put the model into evaluation mode.
        self.__model.eval()

        # Set up the transform to apply to each image.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.__transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # PUBLIC METHODS

    def generate_correspondences(self, img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # TODO
        logging.info("Transforming images")
        img_list = [img1, img2]
        label_list = []
        img_list, label_list = self.__transform(img_list, label_list)
        img_list = [img.unsqueeze(0) for img in img_list]
        logging.info("Transformed images")

        with torch.no_grad():
            # TODO
            logging.info("Copying images to device")
            img_list = [img.to(torch.device("cuda")) for img in img_list]
            logging.info("Copied images")

            # TODO
            # h, w, _ = img1.shape
            # resized_img_list = [F.interpolate(img, (h, w), mode='bilinear', align_corners=True) for img in img_list]
            resized_img_list = img_list

            # TODO
            logging.info("Running model")
            output = self.__model(
                img_list=resized_img_list,
                label_list=label_list,
                get_vect=True,
                get_prob=True,
                get_epe=False
            )
            logging.info("Ran model")


def to_rgb(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, :, [2, 1, 0]])


def main():
    # Set up logging.
    logging.basicConfig(level=logging.INFO)

    # Parse any command-line arguments.
    parser = ArgumentParser()

    # parser.add_argument("--remove_moving", action="store_true", help="whether to remove the moving objects")
    # parser.add_argument("--remove_static", action="store_true", help="whether to remove the static scene")
    #
    # # 0 = reflectances, 1 = uniform colour per frame, 2 = image colours
    # parser.add_argument("--rendering_type", "-r", type=int, default=0, help="the rendering type")
    #
    # parser.add_argument("--start_frame", "-s", type=int, default=0, help="the start frame")
    # parser.add_argument("--end_frame", "-e", type=int, default=5, help="the end frame")
    # parser.add_argument("--frame_step", type=int, default=1, help="the frame step")

    args = parser.parse_args()

    # Create an HD^3 flow-based correspondence generator.
    flow_gen = HD3CorrespondenceGenerator(
        task="flow",
        encoder="dlaup",
        decoder="hda",
        corr_range=[4, 4, 4, 4, 4],
        context=True,
        model_path="model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth"
    )

    # Load in the images.
    bgr1: np.ndarray = cv2.imread("/media/data/datasets/omd/primary/occlusion_2_static/stereo/000000_left.png")
    bgr2: np.ndarray = cv2.imread("/media/data/datasets/omd/primary/occlusion_2_static/stereo/000001_left.png")

    # Generate the optic flow.
    flow_gen.generate_correspondences(to_rgb(bgr1), to_rgb(bgr2))


if __name__ == "__main__":
    main()
