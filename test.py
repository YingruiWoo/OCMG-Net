import os, sys
import shutil
import time
import argparse
import torch
import numpy as np
import random

from net.OCMG_Net import Network
from net.utils import get_sign
from utils.misc import get_logger, seed_all
from dataset import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data
import scipy.spatial as spatial


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--data_set', type=str, default='PCPNet') # 'FamousShape'
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--ckpt_dirs', type=str, default='001', help="can be multiple directories, separated by ',' ")
    parser.add_argument('--ckpt_iter', type=str, default='900')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--testset_list', type=str, default='testset_all.txt') # 'testset_FamousShape' for FamousShape Dataset
    parser.add_argument('--eval_list', type=str,
                        default=['testset_no_noise.txt', 'testset_low_noise.txt', 'testset_med_noise.txt', 'testset_high_noise.txt', 
                                 'testset_vardensity_striped.txt', 'testset_vardensity_gradient.txt'],
                        # ['testset_noise_clean.txt', 'testset_noise_low.txt', 'testset_noise_med.txt', 'testset_noise_high.txt',
                        #  'testset_density_stripe.txt', 'testset_density_gradient.txt'] for FamousShape Dataset
                        nargs='*', help='list of .txt files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=700)
    parser.add_argument('--pcl_size', type=int, default=1200)
    parser.add_argument('--knn_l1', type=int, default=16)
    parser.add_argument('--knn_l2', type=int, default=32)
    parser.add_argument('--knn_h1', type=int, default=32)
    parser.add_argument('--knn_h2', type=int, default=16)
    parser.add_argument('--knn_d', type=int, default=16)
    parser.add_argument('--knn_g', type=int, default=8)
    parser.add_argument('--sparse_patches', type=eval, default=True, choices=[True, False],
                        help='test on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_pn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--matric', type=str, default='CND', choices=['CND', 'RMSE'])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            sparse_patches=args.sparse_patches,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
            pcl_size=args.pcl_size,
            seed=args.seed,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
    return test_dset, test_dataloader


### Arguments
args = parse_arguments()
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
print('Arguments:\n %s\n' % arg_str)

seed_all(args.seed)
PID = os.getpid()

assert args.gpu >= 0, "ERROR GPU ID!"
_device = torch.device('cuda:%d' % args.gpu)

### Datasets and loaders
test_dset, test_dataloader = get_data_loaders(args)


def normal_error(normal_gts, normal_preds, eval_file='log.txt', matric='CND'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()

    errors   = []
    errors_o = []
    o_percents = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5  = []
    pgp_alpha = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented rms
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))
        ang_o = np.rad2deg(np.arccos(nn))
        
        o_percent = np.where(nn < 0, 1., 0.)

        ### Error metric
        errors.append(np.sqrt(np.mean(np.square(ang))))
        errors_o.append(np.sqrt(np.mean(np.square(ang_o))))
        o_percents.append(np.sum(o_percent)/np.sum(np.ones_like(o_percent, dtype=np.float)))
        ### Portion of good points
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(90):
            pgp_alpha_shape.append(sum([j < alpha for j in ang_o]) / float(len(ang_o)))

        pgp_alpha.append(pgp_alpha_shape)

    avg_errors   = np.mean(errors)
    avg_errors_o = np.mean(errors_o)
    avg_o_percents = np.mean(o_percents)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string('%s per shape: ' % matric + str(errors))
    log_string('%s per shape oriented: ' % matric + str(errors_o))
    log_string('per shape percent: ' + str(o_percents))
    log_string('%s not oriented (shape average): ' % matric + str(avg_errors))
    log_string('%s oriented (shape average): ' % matric + str(avg_errors_o))
    log_string('percent: ' + str(avg_o_percents))
    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + np.array2string(avg_pgp_alpha, separator=', '))
    log_file.close()

    return avg_errors, avg_errors_o


def test(ckpt_dir, ckpt_iter):
    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal_FS')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_save_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=_device)
    model = Network(num_in=args.patch_size,
                    num_in_g=args.pcl_size,
                    knn_l1=args.knn_l1,
                    knn_l2=args.knn_l2,
                    knn_h1=args.knn_h1,
                    knn_h2=args.knn_h2,
                    knn_d=args.knn_d,
                    knn_g=args.knn_g,
                    ).to(_device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Num_params: %d' % num_params)
    logger.info('Num_params_trainable: %d' % trainable_num)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]

    num_batch = len(test_dataloader)
    normal_prop = torch.zeros([shape_patch_count, 3])

    total_time = 0
    for batchind, data in enumerate(test_dataloader, 0):
        pcl_pat = data['pcl_pat'].to(_device)          # (B, N, 3)
        data_trans = data['pca_trans'].to(_device)
        pcl_sample = data['pcl_sample'].to(_device)
        pre_oriented_normal = data['pre_oriented_center_normal'].to(_device)
        mst_sign =  data['mst_sign'].unsqueeze(-1).to(_device)

        start_time = time.time()
        with torch.no_grad():
            n_est, n_sign_p, n_sign_n,  _, trans = model(pcl_pat, pcl_sample, pre_oriented_normal)
        end_time = time.time()
        elapsed_time = 1000 * (end_time - start_time)  # ms
        total_time += elapsed_time

        if batchind % 5 == 0:
            batchSize = pcl_pat.size()[0]
            logger.info('[%d/%d] %s: elapsed_time per point/patch: %.3f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], elapsed_time / batchSize))

        n_sign = torch.where(n_sign_p > n_sign_n, 1.0, -1.0)
        n_est[:, :] = torch.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)
        initial_sign = torch.sign((pre_oriented_normal * n_est).sum(dim=1, keepdims=True))
        p_sign = torch.ones_like(initial_sign, dtype=torch.float32)
        initial_sign = torch.where(initial_sign == 0, p_sign, initial_sign)
        n_est = n_est * initial_sign * get_sign(n_sign.squeeze(), min_val=-1.0)[:, None] * mst_sign

        if data_trans is not None:
            ### transform predictions with inverse PCA rotation (back to world space)
            n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

        ### Save the estimated normals to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

            if shape_patches_remaining <= batch_patches_remaining:
                normals_to_write = normal_prop.cpu().numpy()

                ### for faster reading speed in the evaluation
                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy')
                np.save(save_path, normals_to_write)
                if args.save_pn:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals')
                    np.savetxt(save_path, normals_to_write)
                logger.info('saved normal: {} \n'.format(save_path))

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = torch.zeros([shape_patch_count, 3])

    logger.info('Total Time: %.2f s, Shape Num: %d' % (total_time/1000, shape_num))
    return output_dir, file_save_dir


def eval(normal_gt_path, normal_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'test_summary_FS')
    os.makedirs(eval_summary_dir, exist_ok=True)

    all_avg_errors = []
    all_avg_errors_o = []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list)
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shapes
        normal_gts = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            shape_gt = shape.split('_noise_')[0]
            xyz_ori = load_data(filedir=normal_gt_path, filename=shape + '.xyz', dtype=np.float32)
            xyz_gt = load_data(filedir=normal_gt_path, filename=shape_gt + '.xyz', dtype=np.float32)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape_gt + '.normals', dtype=np.float32)  # (N, 3)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            ### eval with sparse point sets
            points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)      # (n,)
            sys.setrecursionlimit(int(max(1000, round(xyz_gt.shape[0] / 10))))
            kdtree = spatial.cKDTree(xyz_gt, 10)
            qurey_points = xyz_ori[points_idx, :]
            _, nor_idx = kdtree.query(qurey_points)
            if args.matric == 'CND':
                normal_gt = normal_gt[nor_idx, :]
            elif args.matric == 'RMSE':
                normal_gt = normal_gt[points_idx, :]
            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]

            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

        ### compute CND per-list
        avg_errors, avg_errors_o = normal_error(normal_gts=normal_gts,
                                                normal_preds=normal_preds,
                                                eval_file=os.path.join(eval_summary_dir, cur_list[:-4] + '_evaluation_results.txt'),
                                                matric=args.matric)
        all_avg_errors.append(avg_errors)
        all_avg_errors_o.append(avg_errors_o)
        print('%s: %f' % (args.matric, avg_errors))

    s = ('\n {} \n All %s not oriented (shape average): {} | Mean: {}\n' % args.matric).format(
                normal_pred_path, str(all_avg_errors), np.mean(all_avg_errors))
    s = ('\n {} \n All %s oriented (shape average): {} | Mean: {}\n' % args.matric).format(
                normal_pred_path, str(all_avg_errors_o), np.mean(all_avg_errors_o))
    print(s)

    ### delete the output point normals
    if not args.save_pn:
        shutil.rmtree(normal_pred_path)
    return all_avg_errors, all_avg_errors_o



if __name__ == '__main__':
    ckpt_dirs = args.ckpt_dirs.split(',')

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        eval_dict_o = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)

        output_dir, file_save_dir = test(ckpt_dir=ckpt_dir, ckpt_iter=args.ckpt_iter)
        if not output_dir or args.data_set == 'Semantic3D':
            continue
        all_avg_errors, all_avg_errors_o = eval(normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                                                normal_pred_path=file_save_dir,
                                                output_dir=output_dir)

        s = '%s: %s | Mean: %f\n' % (args.ckpt_iter, str(all_avg_errors), np.mean(all_avg_errors))
        s_o = '%s: %s | Mean: %f\n' % (args.ckpt_iter, str(all_avg_errors_o), np.mean(all_avg_errors_o))
        log_file_sum.write(s)
        log_file_sum.write(s_o)
        log_file_sum.flush()
        eval_dict += s
        eval_dict_o += s_o

        log_file_sum.close()
        s = ('\n All %s not oriented (shape average): \n{}\n' % args.matric).format(eval_dict)
        print(s)
        s_o = ('\n All %s oriented (shape average): \n{}\n' % args.matric).format(eval_dict_o)
        print(s_o)


