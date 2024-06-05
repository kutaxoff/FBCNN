import linecache
import os.path
from os.path import join as pjoin
import math
import argparse
import time
import random
import tracemalloc
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
import matplotlib.pyplot as plt
import datetime
from utils.prefetch_dataloader import PrefetchingDataLoader, OverlappingDataLoader
from utils.caching_dataloader import EpochCachingDataLoader
import pickle
import humanize
import psutil


def display_top(snapshot, logger, key_type="lineno", limit=3):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    logger.info("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        logger.info(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logger.info("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.info("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    logger.info("Total allocated size: %.1f KiB" % (total / 1024))


def total_memory_used(logger):
    memory_info = psutil.Process(os.getpid()).memory_info()
    rss = humanize.naturalsize(memory_info.rss)
    vms = humanize.naturalsize(memory_info.vms)
    logger.info(f"Memory usage: {rss} / {vms}")


"""
# --------------------------------------------
# training code for FBCNN
"""


def main(json_path="options/train_fbcnn_graydouble.json"):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=json_path, help="Path to option JSON file."
    )

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt["path"].items() if "pretrained" not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"], net_type="G"
    )
    opt["path"]["pretrained_netG"] = init_path_G
    current_step = init_iter

    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # lr_milestones = np.array(opt['train']["G_scheduler_milestones"])

    # indices = np.where(lr_milestones < current_step)[0]
    # if len(indices) > 0:
    #     crossed_milestone_index = indices[-1]
    #     # opt['train']["G_scheduler_milestones"] = [
    #     #     milestone - current_step
    #     #     for milestone in opt['train']["G_scheduler_milestones"][
    #     #         crossed_milestone_index + 1 :
    #     #     ]
    #     # ]
    #     opt['train']["G_optimizer_lr"] = opt['train']["G_optimizer_lr"] * (
    #         opt['train']["G_scheduler_gamma"] ** (crossed_milestone_index + 1)
    #     )
    #     print(f'Update learning rate: {opt["train"]["G_optimizer_lr"]}, milestones: {opt["train"]["G_scheduler_milestones"]}')


    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = "train"
    utils_logger.logger_info(
        logger_name,
        os.path.join(opt["path"]["root"], opt["task"], logger_name + ".log"),
    )
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    seed = 42
    logger.info("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    """
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    """

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt["datasets"]["train"]["dataset_type"]

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            print(
                "Dataset [{:s} - {:s}] is created.".format(
                    train_set.__class__.__name__, dataset_opt["name"]
                )
            )
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"])
            )
            logger.info(
                "Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size
                )
            )
            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"],
                drop_last=True,
                pin_memory=True,
                persistent_workers=True,
            )
            batch_size = dataset_opt["dataloader_batch_size"]
            if opt["datasets"]["train"]["dataroot_H"] == "./trainsets/Flickr2K":
                cache_dir = os.path.join(
                    opt["path"]["root"],
                    "cached_batches_Flickr2K",
                    f"cached_batches_{batch_size}",
                )
            else:
                cache_dir = os.path.join(
                    opt["path"]["root"],
                    "cached_batches",
                    f"cached_batches_{batch_size}",
                )
            # overlapping_train_loader = OverlappingDataLoader(train_set,
            #                                                  batch_size=dataset_opt['dataloader_batch_size'],
            #                                                  num_workers=dataset_opt['dataloader_num_workers'])
        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            print(
                "Dataset [{:s} - {:s}] is created.".format(
                    test_set.__class__.__name__, dataset_opt["name"]
                )
            )
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
            )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    starting_epoch = current_step // len(train_loader)
    cache_epochs = 1
    custom_train_loader = EpochCachingDataLoader(
        train_loader, cache_dir, starting_epoch, cache_epochs
    )

    mem_logger_name = "memory"
    utils_logger.logger_info(
        mem_logger_name,
        os.path.join(opt["path"]["root"], opt["task"], mem_logger_name + ".log"),
    )
    mem_logger = logging.getLogger(mem_logger_name)

    """
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    """
    model = define_Model(opt)
    print("Model defined", flush=True)

    if opt["merge_bn"] and current_step > opt["merge_bn_startpoint"]:
        logger.info("^_^ -----merging bnorm----- ^_^")
        model.merge_bnorm_test()

    # logger.info(model.info_network())
    print(model.info_network())
    model.init_train()
    print("Model initialized", flush=True)
    # logger.info(model.info_params())

    training_dir = os.path.join(opt["path"]["root"], opt["task"])

    """
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    """
    G_loss_cache_file = pjoin(training_dir, "G_loss.pkl")
    QF_loss_cache_file = pjoin(training_dir, "QF_loss.pkl")
    test_cache_file = pjoin(training_dir, "test.pkl")
    try:
        G_loss_values, G_iterations = pickle.load(open(G_loss_cache_file, "rb"))
        if len(G_iterations) > 0 and G_iterations[-1] > current_step:
            G_iterations = [i for i in G_iterations if i <= current_step]
            G_loss_values = G_loss_values[:len(G_iterations)]
    except Exception as e:
        G_loss_values, G_iterations = [], []
    try:
        QF_loss_values, QF_iterations = pickle.load(open(QF_loss_cache_file, "rb"))
        if len(QF_iterations) > 0 and QF_iterations[-1] > current_step:
            QF_iterations = [i for i in QF_iterations if i <= current_step]
            QF_loss_values = QF_loss_values[:len(QF_iterations)]
    except:
        QF_loss_values, QF_iterations = [], []
    try:
        test_values, test_epochs = pickle.load(open(test_cache_file, "rb"))
        if len(test_epochs) > 0 and test_epochs[-1] > starting_epoch:
            test_epochs = [i for i in test_epochs if i <= starting_epoch]
            test_values = {
                "PSNR": test_values["PSNR"][:len(test_epochs)],
                "SSIM": test_values["SSIM"][:len(test_epochs)],
                "PSNRB": test_values["PSNRB"][:len(test_epochs)],
            }
    except:
        test_values = {
            "PSNR": [],
            "SSIM": [],
            "PSNRB": [],
        }
        test_epochs = []

    G_plot_filename = os.path.join(opt["path"]["root"], opt["task"], "G_loss.png")
    QF_plot_filename = os.path.join(opt["path"]["root"], opt["task"], "QF_loss.png")
    test_plot_filename = os.path.join(opt["path"]["root"], opt["task"], "test-psrn.png")
    ssim_plot_filename = os.path.join(opt["path"]["root"], opt["task"], "test-ssim.png")

    def update_loss_graph(loss_type="G_loss", cache=False):
        loss_values = G_loss_values if loss_type == "G_loss" else QF_loss_values
        iterations = G_iterations if loss_type == "G_loss" else QF_iterations
        if cache:
            pickle.dump(
                (loss_values, iterations),
                open(
                    G_loss_cache_file if loss_type == "G_loss" else QF_loss_cache_file,
                    "wb",
                ),
            )
        # Calculate the trend line
        z = np.polyfit(iterations, loss_values, 1)
        p = np.poly1d(z)

        trail_trend_size = max(len(iterations) // 5, 10)

        # Calculate the trend line for last N values
        plt.figure()
        plt.plot(iterations, loss_values, marker="o", label=loss_type)
        plt.plot(
            iterations, p(iterations), linestyle="--", color="black", label="Trend Line"
        )

        if len(iterations) > trail_trend_size:
            trail_iterations, trail_loss_values = (
                iterations[-trail_trend_size:],
                loss_values[-trail_trend_size:],
            )
            trail_z = np.polyfit(trail_iterations, trail_loss_values, 1)
            trail_p = np.poly1d(trail_z)
            trail_decreases = np.all(np.diff(trail_p(trail_iterations)) < 0)
            trail_label = (
                "Trail is decreasing" if trail_decreases else "Trail is increasing"
            )
            trail_color = "green" if trail_decreases else "red"
            plt.plot(
                trail_iterations,
                trail_p(trail_iterations),
                linestyle="--",
                color=trail_color,
                label=trail_label,
            )
        plt.xlabel("Iterations")
        plt.ylabel(loss_type)
        plt.legend()
        plt.grid(True)
        plt.title(loss_type)
        plt.savefig(G_plot_filename if loss_type == "G_loss" else QF_plot_filename)
        plt.close()

    def update_test_graph():
        pickle.dump((test_values, test_epochs), open(test_cache_file, "wb"))
        plt.figure()
        plt.plot(test_epochs, test_values["PSNR"], marker="o", label="PSNR")
        plt.plot(test_epochs, test_values["PSNRB"], marker="o", label="PSNRB")
        plt.xlabel("Iterations")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid(True)
        plt.title("Test PSRN and PSNRB")
        plt.savefig(test_plot_filename)
        plt.close()
        plt.figure()
        plt.plot(test_epochs, test_values["SSIM"], marker="o", label="SSIM")
        plt.xlabel("Iterations")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid(True)
        plt.title("Test SSIM")
        plt.savefig(ssim_plot_filename)
        plt.close()

    # print(len(train_loader))

    # prefetch_train_loader = PrefetchingDataLoader(train_loader, prefetch_size=4)
    # print(len(prefetch_train_loader))

    batch_size = opt["datasets"]["train"]["dataloader_batch_size"]
    if batch_size < 128:
        accumulation_steps = 128 // batch_size
    else:
        accumulation_steps = 1
    
    skip_lr_update = True

    for epoch in range(starting_epoch, 1000000):  # keep running
        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot, mem_logger, limit=5)
        total_memory_used(mem_logger)

        for i, train_data in enumerate(custom_train_loader):
            current_step += 1
            print(
                "epoch: %d, iter: %d, time: %s"
                % (epoch, current_step, datetime.datetime.now()),
                flush=True,
            )

            # if epoch == 0 and i == 2:
            #     print(torch.cuda.memory_summary())

            if (
                dataset_type == "dnpatch" and current_step % 20000 == 0
            ):  # for 'train400'
                train_loader.dataset.update_data()

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            if not skip_lr_update:
                if opt['train']['G_scheduler_type'] == 'ReduceLROnPlateau':
                    model.update_learning_rate(model.current_log()['G_loss'])
                else:
                    model.update_learning_rate(current_step)
            skip_lr_update = False
            # print(1, flush=True)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            # print(2, flush=True)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            # if (current_step % accumulation_steps == 0) or (i == len(custom_train_loader) - 1):
            # print('OPTIMIZE', flush=True)
            model.optimize_parameters(current_step)
            # print(3, flush=True)

            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt["merge_bn"] and opt["merge_bn_startpoint"] == current_step:
                logger.info("^_^ -----merging bnorm----- ^_^")
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt["train"]["checkpoint_print"] == 0:
                cache_loss_graphs = current_step % opt["train"]["checkpoint_save"] == 0
                logs = model.current_log()  # such as loss
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():  # merge log information into message
                    message += "{:s}: {:.3e} ".format(k, v)
                    if k == "G_loss":
                        G_iterations.append(current_step)
                        G_loss_values.append(v)
                        update_loss_graph("G_loss", cache_loss_graphs)
                    if k == "QF_loss":
                        QF_iterations.append(current_step)
                        QF_loss_values.append(v)
                        update_loss_graph("QF_loss", cache_loss_graphs)
                logger.info(message.rstrip())

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt["train"]["checkpoint_save"] == 0:
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.ipc_collect()
                logger.info("Saving the model.")
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt["train"]["checkpoint_test"] == 0:
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.ipc_collect()

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnrb = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data["H_path"][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt["path"]["images"], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals["E"])
                    H_img = util.tensor2uint(visuals["H"])
                    QF = 1 - visuals["QF"]
                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------

                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    avg_psnr += current_psnr

                    # -----------------------
                    # calculate SSIM
                    # -----------------------

                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)

                    avg_ssim += current_ssim

                    # -----------------------
                    # calculate PSNRB
                    # -----------------------

                    current_psnrb = util.calculate_psnrb(H_img, E_img, border=border)
                    avg_psnrb += current_psnrb

                    logger.info(
                        "{:->4d}--> {:>10s} | PSNR : {:<4.2f}dB | SSIM : {:<4.3f}dB | PSNRB : {:<4.2f}dB".format(
                            idx,
                            image_name_ext,
                            current_psnr,
                            current_ssim,
                            current_psnrb,
                        )
                    )
                    logger.info(
                        "predicted quality factor: {:<4.2f} (actual: {:<4.2f})".format(
                            float(QF), float(test_data["L_qf"])
                        )
                    )

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_psnrb = avg_psnrb / idx

                # testing log
                logger.info(
                    "<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.3f}dB, Average PSNRB : {:<.2f}dB\n".format(
                        epoch, current_step, avg_psnr, avg_ssim, avg_psnrb
                    )
                )
                test_values["PSNR"].append(avg_psnr)
                test_values["SSIM"].append(avg_ssim)
                test_values["PSNRB"].append(avg_psnrb)
                test_epochs.append(epoch)
                update_test_graph()

    logger.info("Saving the final model.")
    model.save("latest")
    logger.info("End of training.")


if __name__ == "__main__":
    # tracemalloc.start()
    main()
