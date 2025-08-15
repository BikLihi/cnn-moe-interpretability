from pycocotools.cocoeval import COCOeval
import json
import torch


def evaluate_coco(dataset, model, threshold=0.05):

    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        running_examples_per_expert_regression = None
        running_expert_importance_regression = None
        running_examples_per_expert_classification = None
        running_expert_importance_classification = None
        running_examples_per_expert = None
        running_expert_importance = None


        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            if hasattr(model, 'examples_per_expert_regression'):
                if running_examples_per_expert_regression is not None:
                    running_examples_per_expert_regression += model.examples_per_expert_regression
                else:
                    running_examples_per_expert_regression = model.examples_per_expert_regression

                if running_expert_importance_regression is not None:
                    running_expert_importance_regression += model.expert_importance_regression
                else:
                    running_expert_importance_regression = model.expert_importance_regression

            if hasattr(model, 'examples_per_expert_classification'):
                if running_examples_per_expert_classification is not None:
                    running_examples_per_expert_classification += model.examples_per_expert_classification
                else:
                    running_examples_per_expert_classification = model.examples_per_expert_classification

                if running_expert_importance_classification is not None:
                    running_expert_importance_classification += model.expert_importance_classification
                else:
                    running_expert_importance_classification = model.expert_importance_classification

            if hasattr(model, 'weights'):
                if running_examples_per_expert is not None:
                    running_examples_per_expert += model.examples_per_expert
                else:
                    running_examples_per_expert = model.examples_per_expert

                if running_expert_importance is not None:
                    running_expert_importance += model.expert_importance
                else:
                    running_expert_importance = model.expert_importance


            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.catIds = list(dataset.coco_labels_inverse.keys())
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print('Evaluation finished')

        return {
            'coco_eval': coco_eval,
            'examples_per_expert_cls': running_examples_per_expert_classification,
            'importance_cls': running_expert_importance_classification,
            'examples_per_expert_reg': running_examples_per_expert_regression,
            'importance_reg': running_expert_importance_regression,
            'examples_per_expert': running_examples_per_expert,
            'importance': running_expert_importance
        }