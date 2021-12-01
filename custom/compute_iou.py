import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    # with open(join(devkit_dir, 'info.json'), 'r') as fp:
    #   info = json.load(fp)
    num_classes = 5 # np.int(info['classes'])
    print(('Num classes', num_classes))
    name_classes = np.array(['1', '2', '3', '4', '5'])# np.array(info['label'], dtype=np.str)
    mapping = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])# np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = image_path_list# join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x.split(' ')[0]).replace('jpg', 'png') for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]).replace('png', 'jpg') for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        label = np.array(Image.open(gt_imgs[ind]))
        pred = np.array(Image.open(pred_imgs[ind]).resize((label.shape[1], label.shape[0])))
        label = label_mapping(label, mapping)
        print(f"pred: {pred.shape}; label: {label.shape}")
        if len(label.shape) == 3 and label.shape[2]==4:
            label = label[:,:,0]
        print(f"len label: {len(label.flatten())}; len pred: {len(pred.flatten())}")
        if len(label.flatten()) != len(pred.flatten()):
            print(('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind])))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print(('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist)))))
    print(f"hist: {hist}")
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print(('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))))
    print(('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))))
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/custom_real_list', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)
