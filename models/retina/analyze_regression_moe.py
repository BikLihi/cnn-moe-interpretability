import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from models.retina.retinanet import model
from models.moe_layer.static_gating_networks import SingleWeightingGatingNetwork


# Colors for plots: expert_colors=[(0, 255, 136), (0, 255, 136), (0, 45, 255), (0, 45, 255)], result_color=(255, 129, 86), scale_factor=5)

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption, scale_factor=1, color=(255, 255, 255)):
    b = np.array(box).astype(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.75 * scale_factor
    (test_width, text_height), baseline = cv2.getTextSize(caption, font, font_size, 2)
    cv2.rectangle(image, (b[0], b[1]), (b[0] + test_width, b[1] - text_height - baseline), color,thickness=cv2.FILLED)
    cv2.putText(image, caption, (b[0], b[1] - 10), font, font_size, (0, 0, 0), 2)


def analyze_regression(image_path, model, output_folder, inactivate_experts=False, active_experts=False, expert_colors=None, result_color=None, class_list=None, show_score=False, scale_factor=1):

    classes = [line.strip() for line in open(class_list, 'r')]

    labels = {}
    for key, value in enumerate(classes):
        labels[key] = value
    retinanet = model
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()

    for img_name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            # combined prediction
            if active_experts or inactivate_experts:
                scores, classification, boxes = retinanet(image.cuda().float(), return_expert_predictions=True)
            else:
                scores, classification, boxes = retinanet(image.cuda().float(), return_expert_predictions=False)
            if active_experts or inactivate_experts:
                weights_regression = retinanet.weights_regression
                transformed_anchors_experts = retinanet.finalExpertPredictions

            # Compute predictions of inactive experts
            # if inactivate_experts:
            #     top_k_temp = retinanet.regressionModel.gate.top_k
            #     retinanet.regressionModel.gate.top_k = 4
            #     scores_inactive, classification_inactive, transformed_anchors_inactive = retinanet(image.cuda().float(), return_expert_predictions=True)
            #     weights_regression_inactive = retinanet.weights_regression
            #     transformed_anchors_experts = retinanet.finalExpertPredictions
            #     retinanet.regressionModel.gate.top_k = top_k_temp

            # save gate
            #gate_temp = retinanet.regressionModel.gate

            # # single expert predictions
            # scores_experts, classification_experts, transformed_anchors_experts = [], [], []
            # for i in range(retinanet.regressionModel.num_experts):
            #     singleGate = SingleWeightingGatingNetwork(in_channels=256, expert_index=i, num_experts=retinanet.regressionModel.num_experts, name='SingleWeightGatingNetwork')
            #     retinanet.regressionModel.gate = singleGate
            #     scores_temp, classification_temp, transformed_anchors_temp = retinanet(image.cuda().float())
            #     scores_experts.append(scores_temp)
            #     classification_experts.append(classification_temp)
            #     transformed_anchors_experts.append(transformed_anchors_temp)

            
            # retinanet.regressionModel.gate = gate_temp
            idxs = np.where(scores.cpu() > 0.5)

            if active_experts or inactivate_experts:
                expert_weights = weights_regression[idxs]
                if expert_colors is None:
                    expert_colors = [(0, 255, 221), (252, 203, 153), (58, 40, 255), (255, 127, 0)]


            # plot bounding boxes
            for j in range(idxs[0].shape[0]):
                #print('resulting weights')
                #print(weights_regression[j])
                # plot inactive expert predictions
                if inactivate_experts:
                    for i in range(2, 4):
                        expert_index = torch.topk(expert_weights, 4).indices[j][i]
                        bbox = transformed_anchors_experts[expert_index][idxs[0][j]]
                        x1 = int(bbox[0] / scale)
                        y1 = int(bbox[1] / scale)
                        x2 = int(bbox[2] / scale)
                        y2 = int(bbox[3] / scale)
                        if expert_colors:
                            color = expert_colors[i]
                        else:
                            color = (0, 0, 249)
                        cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=color, thickness=4*scale_factor)
                # plot single expert predictions
                if active_experts:
                    for i in range(2):
                        expert_index = torch.topk(expert_weights, 2).indices[j][i]
                        bbox = transformed_anchors_experts[expert_index][idxs[0][j]]
                        x1 = int(bbox[0] / scale)
                        y1 = int(bbox[1] / scale)
                        x2 = int(bbox[2] / scale)
                        y2 = int(bbox[3] / scale)
                        if expert_colors:
                            color = expert_colors[i]
                        else:
                            color = (86, 255, 170)
                        cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=color, thickness=4*scale_factor)

                

                # Plot final prediction
                bbox = boxes[idxs[0][j], :]
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                score = scores[j]
                if show_score:
                    caption = '{} {:.3f}'.format(label_name, score)
                else:
                    caption = '{}'.format(label_name)
                if result_color is None:
                    result_color=(255, 255, 255)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=result_color, thickness=4*scale_factor)
                # draw_caption(image_orig, (x1, y1, x2, y2), caption, scale_factor)
            # cv2.imshow('detections', image_orig)
            path = os.path.join(output_folder, img_name.split('.')[0] + '_pred.png')
            print(path)
            print()
            cv2.imwrite(path, image_orig)
