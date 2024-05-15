from launch import JeanZayExperiment
import argparse
from pathlib import Path

import os


def parse_mode():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    return args


cmd_modifiers = []
exps = []

exp_name = f"Text_256_DiHpp_B2_wd0.1"
job_name = f"L_text"
jz_exp = JeanZayExperiment(exp_name, job_name)
jz_exp.nodes = 2
jz_exp.num_gpus_per_node = 8
jz_exp.qos = "t3"
jz_exp.account = "syq"
jz_exp.gpu_type = "a100"
jz_exp.time = "20:00:00"

exps.append(jz_exp)

trainer_modifiers = {
    "experiment_name_suffix": exp_name,
    "computer": "cluster-node-a100.yaml",
    "computer.devices": jz_exp.num_gpus_per_node,
    "computer.progress_bar_refresh_rate": 10,
    "computer.num_nodes": jz_exp.nodes,
    "data_dir": Path(os.environ["SCRATCH"]),
}

exp_modifier = {
    "load_weight_from_checkpoint": "/gpfswork/rech/syq/uey53ph/HoMM/checkpoints/256_DiHpp_L2_ft211k.ckpt",
    "data.full_batch_size": 64,
    "model.network.im_size": 32,
    "model.network.input_dim": 4,
    "model.network.dim": 1024,
    "data.size": 256,
    "model.network.kernel_size": 2,
    "model.network.n_layers": 24,
    "model.network.order": 2,
    "model.network.order_expand": 2,
    "model.network.ffw_expand": 2,
    "model.network.n_timesteps": 1000,
    "model.network.dropout": 0.0,
    "model/optimizer": "adamw",
    "model.optimizer.optim.weight_decay": 0.1,
    "model.optimizer.optim.lr": 1e-4,
    "trainer.max_steps": 4001000,
    "model/lr_scheduler": "warmup",
    "model.lr_scheduler.warmup_steps": 10000,
    "computer.num_workers": 10,
    "computer.precision": "16-mixed",
    "computer.num_nodes": 2,
    "computer.devices": 8,
    "computer.strategy": "ddp",
    "trainer.deterministic": False,
    "logger.offline": True,
    "model.instance.latent_vae": True,
    "model.sampler.clip_img_pred": False,
    "model.instance.torch_compile": False,
    "model.ema.beta": 0.9999,
    "model.ema.update_after_step": 50000,
    "model.ema.update_every": 5,
}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
