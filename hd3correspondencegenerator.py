import cv2
import logging
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import data.flowtransforms as transforms
import utils.flowlib as fl

from argparse import ArgumentParser

from hd3model import HD3Model
from models.hd3_ops import *


class HD3CorrespondenceGenerator:
    # CONSTRUCTORS

    def __init__(self, *, task: str, encoder: str, decoder: str, corr_range: [int], context: bool, model_path: str):
        self.__corr_range = corr_range
        self.__task = task

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
            h, w, _ = img1.shape
            resized_img_list = [F.interpolate(img, (h, w), mode='bilinear', align_corners=True) for img in img_list]

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

            scale_factor = 1 / 2**(7 - len(self.__corr_range))
            output['vect'] = resize_dense_vector(output['vect'] * scale_factor, h, w)

            pred_vect = output['vect'].data.cpu().numpy()
            pred_vect = np.transpose(pred_vect, (0, 2, 3, 1))
            curr_vect = pred_vect[0]

            if self.__task == "flow":
                vis_flo = fl.flow_to_image(curr_vect)
            else:
                vis_flo = fl.flow_to_image(fl.disp2flow(curr_vect))
            vis_flo = cv2.cvtColor(vis_flo, cv2.COLOR_RGB2BGR)

            # Determine whether this is disparity or flow (see get_visualization and prob_gather)
            dim = output['vect'].size(1)

            # Gather all the probabilities from the different maps into a single probability map.
            pred_prob = prob_gather(output['prob'], normalize=True, dim=dim)

            # Resize the downsampled probability image to be the same size as the disparity/flow images.
            H, W = resized_img_list[0].size()[2:]
            if pred_prob.size(2) != H or pred_prob.size(3) != W:
                pred_prob = F.interpolate(pred_prob, (pred_vect.shape[1], pred_vect.shape[2]), mode='nearest')

            # Convert to a greyscale image.
            np_pred_prob = np.uint8(pred_prob[0][0].data.cpu().numpy() * 255)

            return curr_vect, vis_flo, np_pred_prob


def to_rgb(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, :, [2, 1, 0]])


def main():
    # Set up logging.
    logging.basicConfig(level=logging.INFO)

    # Parse any command-line arguments.
    parser = ArgumentParser()

    parser.add_argument("--sequence_name", type=str, default="primary/cars_3_static", help="the sequence name")
    parser.add_argument("--start_frame", "-s", type=int, default=0, help="the start frame")
    parser.add_argument("--end_frame", "-e", type=int, default=100, help="the end frame")
    parser.add_argument("--frame_step", type=int, default=1, help="the frame step")

    args = parser.parse_args()

    # Create an HD^3 stereo correspondence generator.
    stereo_gen = HD3CorrespondenceGenerator(
        task="stereo",
        encoder="dlaup",
        decoder="hda",
        corr_range = [4, 4, 4, 4, 4, 4],
        context=True,
        model_path="model_zoo/hd3sc_things_kitti-368975c0.pth"
    )

    # Create an HD^3 flow correspondence generator.
    flow_gen = HD3CorrespondenceGenerator(
        task="flow",
        encoder="dlaup",
        decoder="hda",
        corr_range=[4, 4, 4, 4, 4],
        context=True,
        model_path="model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth"
    )

    sequence_dir = os.path.join("/media/data/datasets/omd", args.sequence_name)

    for idx in range(args.start_frame, args.end_frame, args.frame_step):
        left0 = cv2.imread(os.path.join(sequence_dir, "stereo/{:06d}_left.png".format(idx)))
        right0 = cv2.imread(os.path.join(sequence_dir, "stereo/{:06d}_right.png".format(idx)))
        left1 = cv2.imread(os.path.join(sequence_dir, "stereo/{:06d}_left.png".format(idx + args.frame_step)))

        # Generate the disparities.
        curr_vect, vis_flo, np_pred_prob = stereo_gen.generate_correspondences(to_rgb(left0), to_rgb(right0))
        cv2.imshow("Disparities", vis_flo)
        cv2.imshow("Disparities Confidence", np_pred_prob)
        cv2.waitKey(1)

        # Generate the optic flow.
        curr_vect, vis_flo, np_pred_prob = flow_gen.generate_correspondences(to_rgb(left0), to_rgb(left1))
        cv2.imshow("Flow", vis_flo)
        cv2.imshow("Flow Confidence", np_pred_prob)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
