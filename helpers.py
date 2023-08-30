import json
from pose_estimator import PoseEstimator
import cv2

def read_poses_json():
    with open('poses.json') as jsonfile:
        pose_dict = json.load(jsonfile)
        poses = list(pose_dict.keys())

    return poses, pose_dict

def extract_poses(img_path):
    p = PoseEstimator()

    img = cv2.imread(img_path)
    coord = p.get_pose_coords(img)
    angles = p.get_angles(coord)
    colour_dict = p.get_angle_colour_dummy(angles)
    annotated_img = p.get_annotated_image(img, coord, colour_dict)

    cv2.imwrite('templates/processed_cache/1.jpg', annotated_img)

    return angles

def save_angles(angle_dict, posename):
    with open('poses.json', 'r') as poses:
        pose_dict = json.load(poses)
        pose_dict[posename] = angle_dict
    with open('poses.json', 'w') as poses:
        json.dump(pose_dict, poses)