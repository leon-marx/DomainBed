# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
import torch.utils.data
from tqdm import tqdm

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lm import lm_misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    # Own arguments:
    parser.add_argument('--gpu', type=str, default="4")
    parser.add_argument('--hidden_sizes', type=str, default=None)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--latent_size', type=int, default=None)
    parser.add_argument('--save_best_every_checkpoint', action='store_true')

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    if "LM" not in args.algorithm:
        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(
                args.algorithm, args.dataset)
        else:
            hparams = hparams_registry.random_hparams(
                args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed))
    else:
        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(
                args.algorithm, args.dataset, hidden_sizes=args.hidden_sizes, K = args.K, ckpt_path=args.ckpt_path, lamb=args.lamb, latent_size=args.latent_size)
        else:
            hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(
                args.hparams_seed, args.trial_seed), hidden_sizes=args.hidden_sizes, K = args.K, ckpt_path=args.ckpt_path, lamb=args.lamb, latent_size=args.latent_size)

    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.lr is not None:
        hparams['lr'] = args.lr

    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda:" + args.gpu
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env)*args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_)*args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    # eval_loaders = [FastDataLoader(
    eval_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits + out_splits)]
    eval_weights = [None for _, weights in (
        in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        print("Tried to load state dict -> CORRUPTED!")
        # algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    eval_minibatches_iterator = zip(*eval_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size']
                          for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    progress_bar = tqdm(range(start_step, n_steps))
    train_loss = [0 for i in range(10)]
    train_kld = [0 for i in range(10)]
    train_recon = [0 for i in range(10)]
    cond_dict = {
            0: "art_painting",
            1: "cartoon",
            2: "photo",
            3: "sketch",
        }
    eval_cond_dict = {
            0: "art_painting_id",
            1: "cartoon_id",
            2: "photo_id",
            3: "art_painting_ood",
            4: "cartoon_ood",
            5: "photo_ood",
        }
    if args.save_best_every_checkpoint:
        best_loss = np.inf
    for step in progress_bar:
        step_start_time = time.time()
        if "LM" in args.dataset:
            minibatches_device = []
            for x, y in next(train_minibatches_iterator):
                x["image"] = x["image"].to(device)
                x["domain"] = x["domain"].to(device)
                y = y.to(device)
                minibatches_device.append((x, y))
        else:
            minibatches_device = [(x.to(device), y.to(device))
                                for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                          for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        if "LM" in args.dataset:
            step_vals, step_train_loss, step_train_kld, step_train_recon = algorithm.update(minibatches_device, uda_device, return_train_loss=True, split_loss=True, output_dir=args.output_dir)
            train_loss.pop(0)
            train_loss.append(step_train_loss)
            train_kld.pop(0)
            train_kld.append(step_train_kld)
            train_recon.pop(0)
            train_recon.append(step_train_recon)
        else:
            step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # evals = zip(eval_loader_names, eval_loaders, eval_weights)
            # for name, loader, weights in evals:
            #     if "LM" in args.algorithm:
            #         # acc = lm_misc.accuracy(algorithm, loader, weights, device)
            #         acc = None
            #     else:
            #         acc = misc.accuracy(algorithm, loader, weights, device)
            #     results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / \
                (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                # misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            # misc.print_row([results[key] for key in results_keys],
            #                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
            if (args.save_best_every_checkpoint) and (step > 0):
                if results["loss"] < best_loss:
                    print("")
                    print(f"new record at step: {step}")
                    print(f"old best: {best_loss}")
                    print(f"new best: {results['loss']}")
                    best_loss = results['loss']
                    save_checkpoint(f'best_model.pkl')

                    if not os.path.exists(os.path.join(args.output_dir, f"images")):
                        os.makedirs(os.path.join(args.output_dir, f"images"))

                    for i, batch in enumerate(next(train_minibatches_iterator)):
                        images = batch[0]["image"][:4].to(device)
                        classes = batch[1][:4].to(device)
                        enc_conditions = batch[0]["domain"][:4].to(device)
                        dec_conditions = batch[0]["domain"][:4].to(device)
                        reconstructions = algorithm.run(images, classes, enc_conditions, dec_conditions, raw=True)
                        fig = plt.figure(figsize=(16, 8))
                        fig.suptitle(cond_dict[i], fontsize=24)
                        for j in range(reconstructions.shape[0]):
                            image_plt = images[j].permute(1, 2, 0).cpu()
                            reconstruction_plt = reconstructions[j].permute(1, 2, 0).cpu()
                            plt.subplot(2, 4, 1+j)
                            plt.xticks([])
                            plt.yticks([])
                            plt.imshow(image_plt)
                            plt.subplot(2, 4, 5+j)
                            plt.xticks([])
                            plt.yticks([])
                            plt.imshow(reconstruction_plt)
                        plt.savefig(os.path.join(args.output_dir, f"images/train_{cond_dict[i]}.png"))
                        plt.close()

                    for i, batch in enumerate(next(eval_minibatches_iterator)):
                        images = batch[0]["image"][:4].to(device)
                        classes = batch[1][:4].to(device)
                        enc_conditions = batch[0]["domain"][:4].to(device)
                        dec_conditions = batch[0]["domain"][:4].to(device)
                        reconstructions = algorithm.run(images, classes, enc_conditions, dec_conditions, raw=True)
                        fig = plt.figure(figsize=(16, 8))
                        fig.suptitle(eval_cond_dict[i], fontsize=24)
                        for j in range(reconstructions.shape[0]):
                            image_plt = images[j].permute(1, 2, 0).cpu()
                            reconstruction_plt = reconstructions[j].permute(1, 2, 0).cpu()
                            plt.subplot(2, 4, 1+j)
                            plt.xticks([])
                            plt.yticks([])
                            plt.imshow(image_plt)
                            plt.subplot(2, 4, 5+j)
                            plt.xticks([])
                            plt.yticks([])
                            plt.imshow(reconstruction_plt)
                        plt.savefig(os.path.join(args.output_dir, f"images/eval_{eval_cond_dict[i]}.png"))
                        plt.close()

        progress_bar.set_description("Loss: {:0.2f}, KLD: {:0.2f}, Rec: {:0.2f}".format(np.mean(train_loss), np.mean(train_kld), np.mean(train_recon)))

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
