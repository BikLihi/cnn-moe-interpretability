import argparse
import torch
from torchvision import transforms
import wandb
from retinanet import model
from retinanet.dataloader import CocoDataset, CocoSubDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

 
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', default='data/coco', help='Path to COCO directory')
    parser.add_argument('--model_path', default='model_final.pt', help='Path to model', type=str)

    parser = parser.parse_args(args)

    class_file = 'data/coco/subset.names'
    classes = [line.strip() for line in open(class_file, 'r')]
    dataset_val = CocoSubDataset(parser.coco_path, classes=classes, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=False)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.load(parser.model_path)
        # retinanet.load_state_dict(torch.load(parser.model_path))
        # retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path)['model_state_dict'])
        # retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    # retinanet.module.freeze_bn()

    result = coco_eval.evaluate_coco(dataset_val, retinanet)

if __name__ == '__main__':
    main()
