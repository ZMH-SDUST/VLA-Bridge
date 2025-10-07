# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/21 20:18
@Auther ： Zzou
@File ：draw_boxes.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/19 10:08
@Auther ： Zzou
@File ：draw_attention.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：

CLIP_models_adapter_prior2.py
return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]

"""
import cv2
import seaborn as sns

import os
import sys
import torch
import random
import warnings
import pdb
import argparse
import vcoco_text_label, hico_text_label
import pocket
import pocket.advis
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import torch.nn.functional as F

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.ops.boxes import box_iou
from upt_tip_cache_model_free_finetune_distill3 import build_detector
from utils_tip_cache_and_union_finetune import custom_collate, DataFactory
from pocket.ops import relocate_to_cpu, relocate_to_cuda
from hico_text_label import hico_unseen_index

sys.path.append('detr')
warnings.filterwarnings("ignore")


def tranverse_and_get_hoi_cooccurence(dataset):
    category = dataset.num_interation_cls
    hoi_cooccurence = torch.zeros(category, category)
    for anno in dataset._anno:
        num_gt = len(anno['hoi'])
        for i in range(num_gt):
            for j in range(i + 1, num_gt):
                ## need to judge if anno['hoi'][i] and anno['hoi'][j] are the same pair
                h_iou = box_iou(torch.as_tensor(anno['boxes_h'][i:i + 1]), torch.as_tensor(anno['boxes_h'][j:j + 1]))
                o_iou = box_iou(torch.as_tensor(anno['boxes_o'][i:i + 1]), torch.as_tensor(anno['boxes_o'][j:j + 1]))
                if min(h_iou.item(), o_iou.item()) > 0.5:
                    if anno['hoi'][i] == anno['hoi'][j]:
                        continue
                    hoi_cooccurence[anno['hoi'][i], anno['hoi'][j]] += 1
                    hoi_cooccurence[anno['hoi'][j], anno['hoi'][i]] += 1
    hoi_cooccurence = hoi_cooccurence.t() / (hoi_cooccurence.sum(dim=-1) + 1e-9)
    hoi_cooccurence = hoi_cooccurence.t()
    return hoi_cooccurence


def hico_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(hico_text_label.hico_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr


def vcoco_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(vcoco_text_label.vcoco_hoi_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr


def vcoco_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
    """
    The interaction classes corresponding to an object-verb pair

    HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
    index if the pair is valid, None otherwise

    Returns:
        list[list[117]]
    """
    lut = np.full([num_object_cls, num_action_cls], None)
    for i, j, k in class_corr:
        lut[j, k] = i
    return lut.tolist()


def swig_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
    """
    The interaction classes corresponding to an object-verb pair

    class_corr: List[(hoi_id, object_id, action_id)]

    Returns:
        list[list[407]]
    """
    lut = np.full([num_object_cls, num_action_cls], None)
    for hoi_id, object_id, action_id in class_corr:
        lut[object_id, action_id] = hoi_id
    return lut.tolist()


def swig_object_to_interaction(num_object_cls, _class_corr):
    """
    class_corr: List[(x["id"], x["object_id"], x["action_id"])]

    Returns:
        list[list]
    """
    obj_to_int = [[] for _ in range(num_object_cls)]
    for hoi_id, object_id, action_id in _class_corr:
        obj_to_int[object_id].append(hoi_id)
    return obj_to_int


def swig_object_to_verb(num_object_cls, _class_corr):
    """
    class_corr: List[(x["id"], x["object_id"], x["action_id"])]

    Returns:
        list[list]
    """
    obj_to_verb = [[] for _ in range(num_object_cls)]
    for hoi_id, object_id, action_id in _class_corr:
        obj_to_verb[object_id].append(action_id)
    return obj_to_verb


def swig_verb2interaction(num_action_cls, num_interaction_cls, class_corr):
    '''
    Returns: List[hoi_id] = action_id
    '''
    v2i = np.full([num_interaction_cls], None)
    for hoi_id, object_id, action_id in class_corr:
        v2i[hoi_id] = action_id
    return v2i.tolist()


def visualise_pred_image(image, output, actions, action=None, thresh=0.2, save_filename=None, failure=False):
    """Visualise bounding box pairs in the whole image by classes"""
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    scores = output['scores']
    pred = output['labels']

    if action is not None:
        plt.cla()
        if failure:
            keep = torch.nonzero(torch.logical_and(scores < thresh, pred == action)).squeeze(1)
        else:
            keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')
        # pdb.set_trace()
        if len(keep) == 0: return
        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
        plt.cla()
        return


def visualise_gt_image(image, boxes_human, boxes_object, actions, action, save_filename):
    """Visualise bounding box pairs in the whole image by classes"""

    # 期望左上角和右下角坐标

    if action is not None:
        plt.cla()
        pocket.utils.draw_box_pairs(image, boxes_human, boxes_object, width=5)
        plt.imshow(image)
        plt.axis('off')
        for i in range(len(boxes_human)):
            txt = plt.text(*boxes_human[i, :2], "1.00", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
        plt.cla()
        return
    print("")


@torch.no_grad()
def main(rank, args):
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # 1, Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 2, select CLIP model
    torch.cuda.set_device(rank)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'
    if args.clip_model_name == 'ViT-B-32':
        args.clip_model_name = 'ViT-B/32'

    # 3, get dataset
    if args.dataset == "swig":
        args.dataset_file = 'swig'
        trainset = build_dataset(image_set='train', args=args)
        testset = build_dataset(image_set='val', args=args)
        class_coor = [(x["id"], x["object_id"], x["action_id"]) for x in SWIG_INTERACTIONS]
        if args.eval:  # TODO ????
            class_coor = [(x["id"], x["object_id"], x["action_id"]) for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
        object_n_verb_to_interaction = swig_object_n_verb_to_interaction(num_object_cls=1000, num_action_cls=407,
                                                                         class_corr=class_coor)
        trainset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.object_n_verb_to_interaction = object_n_verb_to_interaction
        object_to_interaction = swig_object_to_interaction(num_object_cls=1000, _class_corr=class_coor)
        trainset.object_to_interaction = object_to_interaction
        testset.object_to_interaction = object_to_interaction
        object_to_verb = swig_object_to_verb(num_object_cls=1000, _class_corr=class_coor)
        trainset.object_to_verb = object_to_verb
        testset.object_to_verb = object_to_verb

        verb2interaction = swig_verb2interaction(num_action_cls=407, num_interaction_cls=14130, class_corr=class_coor)
    else:
        trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                               clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                               num_classes=args.num_classes)
        testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                              clip_model_name=args.clip_model_name)
        verb2interaction = None

    if args.dataset == 'vcoco':
        class_corr = vcoco_class_corr()
        trainset.dataset.class_corr = class_corr
        testset.dataset.class_corr = class_corr
        object_n_verb_to_interaction = vcoco_object_n_verb_to_interaction(num_object_cls=len(trainset.dataset.objects),
                                                                          num_action_cls=len(trainset.dataset.actions),
                                                                          class_corr=class_corr)
        trainset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction

    if args.training_set_ratio < 0.9:
        print(f'[INFO]: using {args.training_set_ratio} trainset to train!')
        sub_trainset, valset = trainset.dataset.split(args.training_set_ratio)
        trainset.dataset = sub_trainset
        trainset.keep = [i for i in range(len(sub_trainset))]

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    args.human_idx = 0
    if args.dataset == 'swig':
        object_n_verb_to_interaction = train_loader.dataset.object_n_verb_to_interaction
    else:
        object_n_verb_to_interaction = train_loader.dataset.dataset.object_n_verb_to_interaction

    if args.dataset == 'hicodet':
        if args.num_classes == 117:
            object_to_target = train_loader.dataset.dataset.object_to_verb
        elif args.num_classes == 600:
            object_to_target = train_loader.dataset.dataset.object_to_interaction

        if args.zs:
            object_to_target = train_loader.dataset.zs_object_to_target
    elif args.dataset == 'vcoco':
        if args.num_classes == 24:
            object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        elif args.num_classes == 236:
            raise NotImplementedError
    elif args.dataset == 'swig':
        if args.num_classes == 407:
            object_to_target = train_loader.dataset.object_to_verb
        elif args.num_classes == 14130 or args.num_classes == 5539:
            object_to_target = train_loader.dataset.object_to_interaction

    print('[INFO]: num_classes', args.num_classes)
    if args.dataset == 'vcoco' or args.dataset == 'swig':
        num_anno = None
    else:
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        if args.num_classes == 117:
            num_anno = torch.as_tensor(trainset.dataset.anno_action)

    # 4, get model
    upt = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction,
                         clip_model_path=args.clip_dir_vit, num_anno=num_anno, verb2interaction=verb2interaction)
    if args.dataset == 'hicodet' and args.eval:  ## after building model, manually change obj_to_target
        if args.num_classes == 117:
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb
        else:
            upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_interaction
    if args.pseudo_label:  ## if we generate pseudo label for unseen verbs,
        pdb.set_trace()
        upt.object_class_to_target_class = test_loader.dataset.dataset.object_to_verb

    # 5, load pretrained weights
    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.dataset == 'swig' and args.eval:
            ckpt = checkpoint['model_state_dict']
            test_hoi_ids = torch.as_tensor(
                [interaction["id"] for interaction in SWIG_INTERACTIONS if interaction["evaluation"] == 1])
            ckpt['clip_head.prompt_learner.token_prefix'] = ckpt['clip_head.prompt_learner.token_prefix'][test_hoi_ids,
                                                            :, :]
            ckpt['clip_head.prompt_learner.token_suffix'] = ckpt['clip_head.prompt_learner.token_suffix'][test_hoi_ids,
                                                            :, :]
            upt.load_state_dict(ckpt)
        else:
            upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    if args.zs and args.fill_zs_verb_type == 1:
        upt.refresh_unseen_verb_cache_mem()  ## whether refresh unseen weights after loading weights (during test)

    if args.vis_tor != 1 and (args.eval or args.cache):
        upt.logit_scale_HO = torch.nn.Parameter(upt.logit_scale_HO * args.vis_tor)
        upt.logit_scale_U = torch.nn.Parameter(upt.logit_scale_U * args.vis_tor)

    for p in upt.detector.parameters():
        p.requires_grad = False

    for n, p in upt.clip_head.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'):
            p.requires_grad = True
            # print(n)
        elif 'adaptermlp' in n or "prompt_learner" in n:
            p.requires_grad = True
            # print(n)
        else:
            p.requires_grad = False

    if args.frozen_classifier != None:
        frozen_name_lst = []
        if 'HO' in args.frozen_classifier:
            frozen_name_lst.append('adapter_HO')
        if 'U' in args.frozen_classifier:
            frozen_name_lst.append('adapter_U')
        if 'T' in args.frozen_classifier:
            frozen_name_lst.append('adapter_union')

        for n, p in upt.named_parameters():
            if 'clip_head' in n or 'detector' in n:
                continue
            if n.split('.')[0] in frozen_name_lst:
                p.requires_grad = False

    if args.label_learning:
        for n, p in upt.named_parameters():
            if 'clip_head' in n or 'detector' in n:
                continue
            if 'label_' in n:
                p.requires_grad = True

    actions = testset.dataset.verbs if args.dataset == 'hicodet' else testset.dataset.actions
    object_n_verb_to_interaction = testset.dataset.object_n_verb_to_interaction

    # attention hook
    # output, 1: 577 (24*24=576)

    upt.cuda()
    upt.eval()

    '''------------------this is for draw boxes of prediction------------------'''
    for index in range(len(testset)):
        image, target = testset[index]
        image = relocate_to_cuda(image)
        output = upt([image])
        output = relocate_to_cpu(output)
        image = testset.dataset.load_image(
            os.path.join(testset.dataset._root, testset.dataset.filename(index))
        )
        filename = target['filename'].split('.')[0] + '_pred.png'

        for action_idx in range(len(actions)):
            # action_idx = args.action
            action_name = actions[action_idx].replace(' ', '_')
            base_path = f'visualization_pred_raw/{args.dataset}/{action_name}'
            if args.zs:
                base_path = f'visualization_pred_raw/zs/{args.zs_type}/{args.dataset}/{action_name}'
            if args.failure:
                base_path = f'visualization_pred_raw_fail/{args.dataset}/{action_name}'
            os.makedirs(base_path, exist_ok=True)
            visualise_pred_image(image, output[0], actions, action=action_idx,
                                 thresh=args.action_score_thresh, save_filename=os.path.join(base_path, filename),
                                 failure=args.failure)

    '''------------------this is for draw boxes of gt------------------'''

    #
    # for index in range(len(testset)):
    #     (image, target), filename = testset.dataset[testset.keep[index]]
    #     # image, target = testset[index]
    #     image = testset.dataset.load_image(
    #         os.path.join(testset.dataset._root, testset.dataset.filename(index))
    #     )
    #     filename = filename.split('.')[0] + '_gt.png'
    #     action_indes= set(list(target['verb']))
    #     for action_idx in action_indes:
    #         action_name = actions[action_idx].replace(' ', '_')
    #         base_path = f'visualization_gt/{args.dataset}/{action_name}'
    #         os.makedirs(base_path, exist_ok=True)
    #         inx = [i for i,ids in enumerate(target["verb"]) if ids == action_idx]
    #         boxes_human = target['boxes_h'][inx]
    #         boxes_object = target['boxes_o'][inx]
    #         visualise_gt_image(image, boxes_human, boxes_object, actions, action=action_idx,
    #                            save_filename=os.path.join(base_path, filename)
    #                            )


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0
    args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')

    # parser.add_argument('--resume', default='checkpoints/hico/ckpt_65674_14.pt', help='Resume from a model')
    parser.add_argument('--resume', default='checkpoints/hico/ckpt_04691_01.pt', help='Resume from a model')

    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--failure', default=False, type=bool)
    parser.add_argument('--visual_mode', default='vit', type=str)

    #### add CLIP vision transformer
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    ### ViT-L/14@336px START: emb_dim: 768
    # >>> vision_width: 1024,  vision_patch_size(conv's kernel-size&&stride-size): 14,
    # >>> vision_layers(#layers in vision-transformer): 24 ,  image_resolution:336;
    # >>> transformer_width:768, transformer_layers: 12, transformer_heads:12
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)

    # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-L/14@336px----END----

    ### ViT-B-16 START
    # parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    # parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)

    # # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int)  # 13 -77
    parser.add_argument('--use_insadapter', action='store_true')
    parser.add_argument('--use_distill', action='store_true')
    parser.add_argument('--use_consistloss', action='store_true')

    parser.add_argument('--use_mean', action='store_true')  # 13 -77
    parser.add_argument('--logits_type', default='HO+U+T', type=str)  # 13 -77 # text_add_visual, visual
    parser.add_argument('--num_shot', default='2', type=int)  # 13 -77 # text_add_visual, visual
    parser.add_argument('--file1',
                        default='./hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p',
                        type=str)
    parser.add_argument('--prior_type', type=str, default='cbe', choices=['cbe', 'cb', 'ce', 'be', 'c', 'b', 'e'])
    parser.add_argument('--obj_affordance', action='store_true')  ## use affordance embedding of objects
    parser.add_argument('--training_set_ratio', type=float, default=1.0)
    parser.add_argument('--frozen_classifier', type=str, default=None)

    parser.add_argument('--zs', action='store_true', default=True)  ## zero-shot
    parser.add_argument('--zs_type', type=str, default='rare_first',
                        choices=['rare_first', 'non_rare_first', 'unseen_verb', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'])
    parser.add_argument('--action_score_thresh', default=0.2, type=float)

    parser.add_argument('--hyper_lambda', type=float, default=2.8)
    parser.add_argument('--use_weight_pred', action='store_true')
    parser.add_argument('--fill_zs_verb_type', type=int, default=0, )  # (for init) 0: random; 1: weighted_sum,
    parser.add_argument('--pseudo_label', action='store_true')
    parser.add_argument('--tpt', action='store_true')
    parser.add_argument('--vis_tor', type=float, default=1.0)
    parser.add_argument('--adapter_num_layers', type=int, default=1)

    ## prompt learning
    parser.add_argument('--N_CTX', type=int, default=24)  # number of context vectors
    parser.add_argument('--CSC', type=bool, default=False)  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    parser.add_argument('--use_templates', action='store_true')
    parser.add_argument('--LA', action='store_true')  ## Language Aware
    parser.add_argument('--LA_weight', default=0.6, type=float)  ## Language Aware(loss weight)

    parser.add_argument('--feat_mask_type', type=int, default=0, )  # 0: dropout(random mask); 1: None
    parser.add_argument('--num_classes', type=int, default=117, )
    parser.add_argument('--prior_method', type=int, default=0)  ## 0: instance-wise, 1: pair-wise, 2: learnable
    parser.add_argument('--vis_prompt_num', type=int, default=50)  ##  (prior_method == learnable)
    parser.add_argument('--box_proj', type=int, default=0, )  ## 0: None; 1: f_u = ROI-feat + MLP(uni-box)
    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last'])
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--label_learning', action='store_true')
    parser.add_argument('--label_choice', default='random',
                        choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first',
                                 'non_rare_first', 'rare+non_rare'])
    parser.add_argument('--use_mlp_proj', action='store_true')

    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')

    ## **************** arguments for deformable detr **************** ##
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'), )
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    ## **************** arguments for deformable detr **************** ##
    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    # mp.spawn(main, nprocs=args.world_size, args=(args,))
    if args.world_size == 1:
        main(0, args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
