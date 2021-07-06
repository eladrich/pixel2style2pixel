import datetime
import os
import numpy as np
import wandb

from utils import common


class WBLogger:

    def __init__(self, opts):
        wandb_run_name = os.path.basename(opts.exp_dir)
        wandb.init(project="pixel2style2pixel", config=vars(opts), name=wandb_run_name)

    @staticmethod
    def log_best_model():
        wandb.run.summary["best-model-save-time"] = datetime.datetime.now()

    @staticmethod
    def log(prefix, metrics_dict, global_step):
        log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()}
        log_dict["global_step"] = global_step
        wandb.log(log_dict)

    @staticmethod
    def log_dataset_wandb(dataset, dataset_name, n_images=16):
        idxs = np.random.choice(a=range(len(dataset)), size=n_images, replace=False)
        data = [wandb.Image(dataset.source_paths[idx]) for idx in idxs]
        wandb.log({f"{dataset_name} Data Samples": data})

    @staticmethod
    def log_images_to_wandb(x, y, y_hat, id_logs, prefix, step, opts):
        im_data = []
        column_names = ["Source", "Target", "Output"]
        if id_logs is not None:
            column_names.append("ID Diff Output to Target")
        for i in range(len(x)):
            cur_im_data = [
                wandb.Image(common.log_input_image(x[i], opts)),
                wandb.Image(common.tensor2im(y[i])),
                wandb.Image(common.tensor2im(y_hat[i])),
            ]
            if id_logs is not None:
                cur_im_data.append(id_logs[i]["diff_target"])
            im_data.append(cur_im_data)
        outputs_table = wandb.Table(data=im_data, columns=column_names)
        wandb.log({f"{prefix.title()} Step {step} Output Samples": outputs_table})
