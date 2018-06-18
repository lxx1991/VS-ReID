import os
import torch
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from core import models
from core.utils import flow as flo
from core.utils.disp import labelcolormap
from core.utils.config import Config
from core.utils.bbox import gen_bbox, label_to_prob, combine_prob, prob_to_label, IoU
from core.utils.pickle_io import pickle_dump, pickle_load
import cv2

patch_shape = (449, 449)

use_flip = True
bbox_occ_th = 0.3
reid_th = 0.5


def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


def predict(st, en, step, instance_list):

    global pred_prob
    global bbox_cnt

    for th in range(st, en, step):
        if (step == 1):
            warp_prob = flo.get_warp_label(flow1[th - 1], flow2[th], combine_prob(pred_prob[th - 1]))
        elif (step == -1):
            warp_prob = flo.get_warp_label(flow2[th + 1], flow1[th], combine_prob(pred_prob[th + 1]))

        bbox = gen_bbox(prob_to_label(warp_prob), range(instance_num), True)
        new_instance_list = []
        abort = True

        temp = prob_to_label(combine_prob(pred_prob[th - step]))

        for i in range(instance_num):
            if (th == st) or (np.count_nonzero(orig_mask[i]) <= np.count_nonzero(temp == (i + 1)) * 10):
                if np.abs(bbox_cnt[th][i] - th) <= np.abs((st - step) - th):
                    continue
                if i in instance_list:
                    abort = False
                    bbox_cnt[th][i] = (st - step)
                    new_instance_list.append(i)
                else:
                    for j in instance_list:
                        if IoU(bbox[i, :], bbox[j, :]) > 1e-6:
                            new_instance_list.append(i)
                            break

        if abort:
            break

        new_instance_list = sorted(new_instance_list)
        temp_image = frames[th].astype(float)

        f_prob = [np.zeros([bbox[idx, 3] - bbox[idx, 1], bbox[idx, 2] - bbox[idx, 0], 2]) for idx in new_instance_list]
        image_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 3), float)
        flow_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 2), float)
        warp_label_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 1), float)

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            warp_label_patch[i, ..., 0] = cv2.resize(warp_prob[bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx + 1], patch_shape).astype(float)
            image_patch[i, ...] = cv2.resize(temp_image[int(0.5 + bbox[idx, 1] * fr_h_r):int(0.5 + bbox[idx, 3] * fr_h_r),
                                                        int(0.5 + bbox[idx, 0] * fr_w_r):int(0.5 + bbox[idx, 2] * fr_w_r), :], patch_shape).astype(float)
            if (step == 1):
                flow_patch[i, ...] = cv2.resize(flow2[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], :], patch_shape).astype(float)
            else:
                flow_patch[i, ...] = cv2.resize(flow1[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], :], patch_shape).astype(float)

        image_patch = torch.from_numpy(image_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        warp_label_patch = torch.from_numpy(warp_label_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        flow_patch = torch.from_numpy(flow_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()

        with torch.no_grad():
            prob = model(image_patch, flow_patch, warp_label_patch)
            prob = torch.nn.functional.softmax(prob, dim=1)

        if use_flip:
            image_patch = flip(image_patch, 3)
            warp_label_patch = flip(warp_label_patch, 3)
            flow_patch = flip(flow_patch, 3)
            flow_patch[:, 0, ...] = -flow_patch[:, 0, ...]
            with torch.no_grad():
                prob_f = model(image_patch, flow_patch, warp_label_patch)
                prob_f = torch.nn.functional.softmax(prob_f, dim=1)
            prob_f = flip(prob_f, 3)
            prob = (prob + prob_f) / 2.0

        prob = prob.data.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            f_prob[i] += cv2.resize(prob[i, ...], (bbox[idx, 2] - bbox[idx, 0], bbox[idx, 3] - bbox[idx, 1]))

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            pred_prob[th][..., idx * 2] = 1
            pred_prob[th][..., idx * 2 + 1] = 0
            pred_prob[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx * 2] = f_prob[i][..., 0]
            pred_prob[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx * 2 + 1] = f_prob[i][..., 1]


def predict_single(th, i, bbox, in_warp_label_patch):
    # first time
    global pred_prob

    new_instance_list = [i]

    # source
    temp = gen_bbox(prob_to_label(combine_prob(pred_prob[th])), range(instance_num), True)
    result = prob_to_label(combine_prob(pred_prob[th][temp[i, 1]:temp[i, 3] + 1, temp[i, 0]:temp[i, 2] + 1, :]))
    result = np.unique(result)
    for j in result:
        if (j != 0) and ((j - 1) not in new_instance_list):
            new_instance_list.append(j - 1)

    # unknow anything about i
    pred_prob[th][(pred_prob[th][..., i * 2 + 1] > 0.5), 0::2] = 1
    pred_prob[th][(pred_prob[th][..., i * 2 + 1] > 0.5), 1::2] = 0

    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    bbox_enlarge = 0.0
    bbox[0] = np.clip((bbox[0] - bbox_enlarge * bbox_w), 0, pred_prob[th].shape[1] - 1)
    bbox[2] = np.clip((bbox[2] + bbox_enlarge * bbox_w), bbox[0] + 1, pred_prob[th].shape[1])
    bbox[1] = np.clip((bbox[1] - bbox_enlarge * bbox_h), 0, pred_prob[th].shape[0] - 1)
    bbox[3] = np.clip((bbox[3] + bbox_enlarge * bbox_h), bbox[1] + 1, pred_prob[th].shape[0])
    bbox = bbox.astype(int)

    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    temp_image = frames[th].astype(float)

    for T in range(5):
        image_patch = np.zeros((1, patch_shape[1], patch_shape[0], 3), float)
        flow_patch = np.zeros((1, patch_shape[1], patch_shape[0], 2), float)

        image_patch[0, ...] = cv2.resize(temp_image[int(0.5 + bbox[1] * fr_h_r):int(0.5 + bbox[3] * fr_h_r), int(0.5 + bbox[0] * fr_w_r):int(0.5 + bbox[2] * fr_w_r), :], patch_shape).astype(float)
        if T == 0:
            warp_label_patch = cv2.resize(in_warp_label_patch, patch_shape)[np.newaxis, ..., np.newaxis].astype(float)
        else:
            warp_label_patch = cv2.resize(pred_prob[th][bbox[1]:bbox[3], bbox[0]:bbox[2], i * 2 + 1], patch_shape)[np.newaxis, ..., np.newaxis].astype(float)

        flow_patch[0, ...] = cv2.resize(flow2[th][bbox[1]:bbox[3], bbox[0]:bbox[2], :], patch_shape).astype(float)

        image_patch = torch.from_numpy(image_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        warp_label_patch = torch.from_numpy(warp_label_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        flow_patch = torch.from_numpy(flow_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()

        with torch.no_grad():
            prob = model(image_patch, flow_patch, warp_label_patch)
            prob = torch.nn.functional.softmax(prob, dim=1)

        if use_flip:
            image_patch = flip(image_patch, 3)
            warp_label_patch = flip(warp_label_patch, 3)
            flow_patch = flip(flow_patch, 3)
            flow_patch[:, 0, ...] = -flow_patch[:, 0, ...]
            with torch.no_grad():
                prob2 = model(image_patch, flow_patch, warp_label_patch)
                prob2 = torch.nn.functional.softmax(prob2, dim=1)

            prob2 = flip(prob2, 3)
            prob = (prob + prob2) / 2.0

        prob = prob.data.cpu().numpy().transpose(0, 2, 3, 1)
        prob = cv2.resize(prob[0, ...], (bbox[2] - bbox[0], bbox[3] - bbox[1]))

        pred_prob[th][..., i * 2] = 1
        pred_prob[th][..., i * 2 + 1] = 0
        pred_prob[th][bbox[1]:bbox[3], bbox[0]:bbox[2], i * 2] = prob[..., 0]
        pred_prob[th][bbox[1]:bbox[3], bbox[0]:bbox[2], i * 2 + 1] = prob[..., 1]

    temp = pred_prob[th][..., (i * 2):(i * 2 + 2)].copy()
    pred_prob[th][(pred_prob[th][..., i * 2 + 1] > 0.5), 0::2] = 1
    pred_prob[th][(pred_prob[th][..., i * 2 + 1] > 0.5), 1::2] = 0
    pred_prob[th][..., (i * 2):(i * 2 + 2)] = temp.copy()

    # target
    bbox = gen_bbox(prob_to_label(combine_prob(pred_prob[th])), range(instance_num), True)
    result = prob_to_label(combine_prob(pred_prob[th][bbox[i, 1]:bbox[i, 3], bbox[i, 0]:bbox[i, 2], :]))
    result = np.unique(result)
    for j in result:
        if (j != 0) and ((j - 1) not in new_instance_list):
            new_instance_list.append(j - 1)
    new_instance_list = sorted(new_instance_list)

    for T in range(5):
        # second time
        warp_prob = combine_prob(pred_prob[th])
        bbox = gen_bbox(prob_to_label(warp_prob), range(instance_num), True)

        image_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 3), float)
        flow_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 2), float)
        warp_label_patch = np.zeros((len(new_instance_list), patch_shape[1], patch_shape[0], 1), float)

        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            warp_label_patch[i, ..., 0] = cv2.resize(warp_prob[bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx + 1], patch_shape).astype(float)
            image_patch[i, ...] = cv2.resize(temp_image[int(0.5 + bbox[idx, 1] * fr_h_r):int(0.5 + bbox[idx, 3] * fr_h_r),
                                                        int(0.5 + bbox[idx, 0] * fr_w_r):int(0.5 + bbox[idx, 2] * fr_w_r), :], patch_shape).astype(float)
            flow_patch[i, ...] = cv2.resize(flow2[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], :], patch_shape).astype(float)

        image_patch = torch.from_numpy(image_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        warp_label_patch = torch.from_numpy(warp_label_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()
        flow_patch = torch.from_numpy(flow_patch.transpose(0, 3, 1, 2)).contiguous().float().cuda()

        with torch.no_grad():
            prob = model(image_patch, flow_patch, warp_label_patch)
            prob = torch.nn.functional.softmax(prob, dim=1)

        if use_flip:
            image_patch = flip(image_patch, 3)
            warp_label_patch = flip(warp_label_patch, 3)
            flow_patch = flip(flow_patch, 3)
            flow_patch[:, 0, ...] = -flow_patch[:, 0, ...]

            with torch.no_grad():
                prob2 = model(image_patch, flow_patch, warp_label_patch)
                prob2 = torch.nn.functional.softmax(prob2, dim=1)
            prob2 = flip(prob2, 3)
            prob = (prob + prob2) / 2.0

        prob = prob.data.cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(len(new_instance_list)):
            idx = new_instance_list[i]
            f_prob = cv2.resize(prob[i, ...], (bbox[idx, 2] - bbox[idx, 0], bbox[idx, 3] - bbox[idx, 1]))
            pred_prob[th][..., idx * 2] = 1
            pred_prob[th][..., idx * 2 + 1] = 0
            pred_prob[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx * 2] = f_prob[..., 0]
            pred_prob[th][bbox[idx, 1]:bbox[idx, 3], bbox[idx, 0]:bbox[idx, 2], idx * 2 + 1] = f_prob[..., 1]


def update_appear():
    global pred_prob
    global appear
    global location

    for th in range(appear.shape[0]):
        bbox = gen_bbox(prob_to_label(combine_prob(pred_prob[th])), range(instance_num))
        for i in range(appear.shape[1]):
            appear[th, i] = ((bbox[i, 2] - bbox[i, 0]) * (bbox[i, 3] - bbox[i, 1]) > 1)
            if appear[th, i] > 0:
                location[th, i, 0] = float(bbox[i, 2] + bbox[i, 0]) / 2
                location[th, i, 1] = float(bbox[i, 3] + bbox[i, 1]) / 2
            else:
                location[th, i, :] = location[th - 1, i, :]


def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation')
    parser.add_argument('testset', type=str)
    parser.add_argument('config', help='config file path')
    # ========================= Model Configs ==========================
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cache', type=str, default='')
    args = parser.parse_args()

    return args


def main():

    def save_frame(th, do_pause, dir_name='', vis=True):
        result = prob_to_label(combine_prob(pred_prob[th]))
        result_show = np.dstack((colors[result, 0], colors[result, 1], colors[result, 2])).astype(np.uint8)
        if args.output != '' and dir_name != '':
            out_file = os.path.join(dataset_dir, 'Results', 'Segmentations', resolution, args.output, dir_name, video_dir, '%05d.png' % th)
            if not os.path.exists(os.path.split(out_file)[0]):
                os.makedirs(os.path.split(out_file)[0])
            if vis:
                cv2.imwrite(out_file, result_show)
            else:
                cv2.imwrite(out_file, result)
        temp = cv2.resize(frames[th], frame_0.shape[-2::-1]) * 0.3 + result_show * 0.7
        return
        cv2.imshow('Result', temp.astype(np.uint8))
        if do_pause:
            cv2.waitKey()
        else:
            cv2.waitKey(100)

    colors = labelcolormap(256)

    global pred_prob, frames, flow1, flow2, orig_mask, \
        model, instance_num, fr_h_r, fr_w_r, appear, bbox_cnt, \
        location, patch_shapes

    args = parse_args()
    cfg = Config.from_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model = []
    model = getattr(models, cfg.model.name)(cfg.model)

    if os.path.isfile(cfg.weight):
        print(("=> loading checkpoint '{}'".format(cfg.weight)))
        checkpoint = torch.load(cfg.weight)
        model.load_state_dict(checkpoint['state_dict'], True)
        print(("=> loaded checkpoint"))
    else:
        raise (("=> no checkpoint found at '{}'".format(cfg.weight)))
    model = model.cuda()
    model.eval()

    cudnn.benchmark = True

    # Setup dataset
    dataset_dir = os.path.join('data/DAVIS')
    resolution = '480p'
    imageset_dir = os.path.join(dataset_dir, 'ImageSets', '2017', args.testset + '.txt')

    video_list = []
    for line in open(imageset_dir).readlines():
        if line.strip() != '':
            video_list.append(line.strip())

    person_all = pickle_load(os.path.join(dataset_dir, 'PersonSearch', resolution, args.testset + '.pkl'), encoding='latin')
    object_all = pickle_load(os.path.join(dataset_dir, 'ObjectSearch', resolution, args.testset + '.pkl'), encoding='latin')
    category_all = pickle_load(os.path.join(dataset_dir, 'Class', resolution, args.testset + '.pkl'), encoding='latin')
    frame_cnt = 0

    use_cache = (args.cache != '')
    video_cnt = -1

    for video_dir in video_list:
        video_cnt += 1
        frame_dir = os.path.join(dataset_dir, 'JPEGImages', resolution, video_dir)
        frame_fr_dir = os.path.join(dataset_dir, 'JPEGImages', 'Full-Resolution', video_dir)
        label_dir = os.path.join(dataset_dir, 'Annotations', resolution, video_dir)
        flow_dir = os.path.join(dataset_dir, 'Flow', resolution, video_dir)
        cache_dir = os.path.join(dataset_dir, 'Cache', resolution, args.cache, video_dir)
        frames_num = len(os.listdir(frame_dir))

        if (video_cnt % args.gpu_num != args.gpu):
            frame_cnt += frames_num
            continue

        frame_0 = cv2.imread(os.path.join(frame_dir, '%05d.jpg' % 0))
        label_0 = cv2.imread(os.path.join(label_dir, '%05d.png' % 0), cv2.IMREAD_UNCHANGED)

        instance_num = label_0.max()

        frames = [None for _ in range(frames_num)]
        pred_prob = [None for _ in range(frames_num)]
        flow1 = [None for _ in range(frames_num)]
        flow2 = [None for _ in range(frames_num)]
        person_reid = [[None for _ in range(instance_num)] for _ in range(frames_num)]
        object_reid = [[None for _ in range(instance_num)] for _ in range(frames_num)]
        category = category_all[video_dir]
        orig_mask = [None for _ in range(instance_num)]

        frames[0] = cv2.imread(os.path.join(frame_fr_dir, '%05d.jpg' % 0))
        fr_h_r = float(frames[0].shape[0]) / float(frame_0.shape[0])
        fr_w_r = float(frames[0].shape[1]) / float(frame_0.shape[1])
        pred_prob[0] = label_to_prob(label_0, instance_num)
        person_reid[0] = person_all[frame_cnt]
        object_reid[0] = object_all[frame_cnt]

        save_frame(0, False, 'result', False)

        bbox = gen_bbox(label_0, range(instance_num), True)
        for i in range(instance_num):
            orig_mask[i] = pred_prob[0][bbox[i, 1]:bbox[i, 3], bbox[i, 0]:bbox[i, 2], i * 2 + 1]

        for th in range(1, frames_num):
            frames[th] = cv2.imread(os.path.join(frame_fr_dir, '%05d.jpg' % th))
            pred_prob[th] = label_to_prob(np.zeros_like(label_0, np.uint8), instance_num)
            flow1[th - 1] = flo.readFlow(os.path.join(flow_dir, '%05d.flo' % (th - 1)))
            flow2[th] = flo.readFlow(os.path.join(flow_dir, '%05d.rflo' % th))
            person_reid[th] = person_all[frame_cnt + th]
            object_reid[th] = object_all[frame_cnt + th]

        bbox_cnt = -1000 * np.ones((frames_num, instance_num))
        bbox_cnt[0, :] = 0

        for th in range(frames_num):
            for i in range(instance_num):
                person_reid[th][i] = person_reid[th][i][:, [0, 1, 2, 3, 5]]
                object_reid[th][i] = object_reid[th][i][:, [0, 1, 2, 3, 5]]
        frame_cnt += frames_num

        cache_file = os.path.join(cache_dir, '%s.pkl' % video_dir)

        if (use_cache and os.path.exists(cache_file)):
            pred_prob, bbox_cnt = pickle_load(cache_file, encoding='latin')
        else:
            predict(1, frames_num, 1, range(instance_num))
            if use_cache:
                if not os.path.exists(os.path.split(cache_file)[0]):
                    os.makedirs(os.path.split(cache_file)[0])
                pickle_dump((pred_prob, bbox_cnt), cache_file)

        appear = np.zeros((frames_num, instance_num)).astype(int)
        location = np.zeros((frames_num, instance_num, 2)).astype(int)
        update_appear()

        for th in range(frames_num):
            save_frame(th, False, 'draft', True)

        for reid_target in ['person', 'object']:
            cache_file = os.path.join(cache_dir, '%s_%s.pkl' % (reid_target, video_dir))
            reid_score = None

            if (use_cache and os.path.exists(cache_file)):
                pred_prob, bbox_cnt = pickle_load(cache_file, encoding='latin')
            else:
                target_instance = []
                for i in range(instance_num):
                    if (reid_target == 'object' or category[i][123] > 0.5):  # person is 123
                        target_instance.append(i)
                exec("reid_score = %s" % reid_target)
                draft_cnt = 0
                while (True):
                    max_score = 0
                    for i in range(1, frames_num - 1):
                        temp_label = prob_to_label(combine_prob(pred_prob[i]))
                        bbox_i = gen_bbox(temp_label, range(instance_num), False, 0.99)
                        for j in target_instance:
                            if bbox_cnt[i, j] != i and reid_score[i][j].shape[0] > 0:
                                bbox_id = np.argmax(reid_score[i][j][:, 4])
                                # retrieval
                                if (appear[i, j] == 0):
                                    x1, y1, x2, y2 = reid_score[i][j][bbox_id, 0:4]
                                    if (reid_score[i][j][bbox_id, 4] > max_score and reid_score[i][j][bbox_id, 4] > reid_th):

                                        bbox_now = reid_score[i][j][bbox_id, 0:4].astype(int)
                                        result = np.bincount(temp_label[bbox_now[1]:bbox_now[3] + 1, bbox_now[0]:bbox_now[2] + 1].flatten(), minlength=j + 2)
                                        flag = True

                                        if flag:
                                            for occ_instance in np.where(result[1:] > 0)[0]:
                                                if (IoU(bbox_now, bbox_i[occ_instance]) > bbox_occ_th):
                                                    flag = False

                                        if flag:
                                            for k in target_instance:
                                                if (k != j and appear[i, k] == 0 and reid_score[i][k][bbox_id, 4] > reid_score[i][j][bbox_id, 4]):
                                                    flag = False

                                        if flag:
                                            max_frame = i
                                            max_instance = j
                                            max_bbox = reid_score[i][j][bbox_id, 0:4]
                                            max_score = reid_score[i][j][bbox_id, 4]

                    if (max_score == 0):
                        break

                    bbox_cnt[max_frame, max_instance] = max_frame

                    predict_single(max_frame, max_instance, max_bbox, orig_mask[max_instance])
                    save_frame(max_frame, False, '%s_%05d_checkpoint' % (reid_target, draft_cnt))

                    temp = 0
                    for i in range(max_frame - 1, -1, -1):
                        if appear[i, max_instance] != 0:
                            temp = i
                            break
                    predict(max_frame - 1, temp, -1, [max_instance])

                    temp = frames_num
                    for i in range(max_frame + 1, frames_num, 1):
                        if appear[i, max_instance] != 0:
                            temp = i
                            break
                    predict(max_frame + 1, temp, 1, [max_instance])
                    update_appear()

                    for th in range(frames_num):
                        save_frame(th, False, '%s_%05d' % (reid_target, draft_cnt))

                    draft_cnt = draft_cnt + 1

                for th in range(frames_num):
                    save_frame(th, False, '%s' % reid_target)

                if use_cache:
                    pickle_dump((pred_prob, bbox_cnt), cache_file)

        for th in range(frames_num):
            save_frame(th, False, 'result', False)


if __name__ == '__main__':
    main()
