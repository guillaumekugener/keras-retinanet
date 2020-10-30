import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import keras

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.image import read_image_bgr, resize_image, preprocess_image
from ..utils.keras_version import check_keras_version
from ..utils.tf_version import check_tf_version

# Make our parse
parser = argparse.ArgumentParser(description='Detection script for a RetinaNet network on mp4')

parser.add_argument('--model-path',       help='path to the model snapshot')
parser.add_argument('--file',             help='path to the video file to run detection on')
parser.add_argument('--class-file',       help='path to the tsv with the classes')
parser.add_argument('--video-id',         help='video id for this file')
parser.add_argument('--save-file',        help='path of to save file')

parser.add_argument('--backbone',         help='the backbone of the model.', default='resnet50')
parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)


args = parser.parse_args()


# The script
video_file = args.file
backbone = models.backbone(args.backbone)
model_path = args.model_path

model = models.load_model(model_path, backbone_name=args.backbone)
model = models.convert_model(model, anchor_params=None, pyramid_levels=None)

# Load the video file
cv_video = cv2.VideoCapture(video_file)

detections = pd.DataFrame({'vid': [], 'frame': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'score': [], 'label': []})
score_threshold = args.score_threshold
max_detections = args.max_detections

while True:
    count += 1    
    ret, raw_image = cv_video.read()
    if not ret:
        break
            
    image        = backbone.preprocess_image(raw_image.copy())
    image, scale = resize_image(image)

    if keras.backend.image_data_format() == 'channels_first':
        image = image.transpose((2, 0, 1))

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

    
    # correct boxes for image scale
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes      = boxes[0, indices[scores_sort], :]
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]
    image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    d = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    # Turn into a pandas df
    dataset = pd.DataFrame({
        'vid': [args.video_id for i in range(d.shape[0])],
        'frame': [count for i in range(d.shape[0])],
        'x1': d[:, 0], 
        'y1': d[:, 1],
        'x2': d[:, 2],
        'y2': d[:, 3],
        'score': d[:, 4],
        'label': d[:, 5]
    })
    
    detections = pd.concat([detections, dataset], ignore_index=True)

# Now save the detections somewhere after we change the labels
class_map = pd.read_csv(args.class_file, header=None)
class_dict = {}
class_index = []
for i in range(len(class_map[0])):
    class_dict[class_map[0][i]] = i
    class_index.append(class_map[0][i])

detections['label'] = [class_index[int(i)] for i in detections['label']]

# Save file
detections.to_csv(args.save_file, index=False, sep=',')



