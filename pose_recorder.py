from pose_estimator import PoseEstimator
import json
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_path", help = "The input image filename")
parser.add_argument("-p", "--posename", help = "The name of the pose")
args = parser.parse_args()

try:
    if not args.img_path or not args.posename:
        raise Exception('')
    img_path = args.img_path
    posename = args.posename
except Exception as e:
    print(e)
    print("Usage: python pose_recorder.py --img_path <image path> --posename <pose name>")
    exit()

img = cv2.imread(img_path)

p = PoseEstimator()

coord = p.get_pose_coords(img)
angles = p.get_angles(coord)
colour_dict = p.get_angle_colour_dummy(angles)
annotated_img = p.get_annotated_image(img, coord, colour_dict)

print("Pose estimation complete, validate the predictions: ")
print("1. Press Y to register the pose")
print("2. Press Q to discard")

while True:
    if coord:
        try:
            cv2.imshow("image", annotated_img)
            key = cv2.waitKey(0)
            if key == 113:
                print(f"Pose discarded")
                break
            elif key == 121:
                with open('poses.json', 'r') as poses:
                    pose_dict = json.load(poses)
                pose_dict[posename] = angles
                
                with open('poses.json', 'w') as poses:
                    json.dump(pose_dict, poses)
                
                print(f"Succesfully registered {posename}")
                break
        except ValueError as ve:
            print(ve)
cv2.destroyAllWindows()

