import datetime
from simple_slurm import Slurm

kwargs = dict(
    cpus_per_task=1,
    job_name=f"or_cv_project_q25",
    output=f"logs/slurm_q25.out",
    error=f"logs/slurm_q25.err",
    time=datetime.timedelta(hours=4),
    partition="killable",
    gpus=1,
    nodes=1,
    ntasks=1,
    mem=10000,
)

slurm = Slurm(**kwargs)
job_id = slurm.sbatch(
    f"""\
python solution/train_main.py -d synthetic_dataset -m XceptionBased --lr 0.001 -b 32 -e 2 -o Adam\
"""
)

# python solution/numerical_analysis.py --model XceptionBased --checkpoint_path checkpoints/synthetic_dataset_XceptionBased_Adam.pt --dataset synthetic_dataset

## python solution/plot_accuracy_and_loss.py -m XceptionBased -j out/synthetic_dataset_XceptionBased_Adam.json -d synthetic_dataset

## python solution/saliency_map.py -m SimpleNet -cpp checkpoints/fakes_dataset_SimpleNet_Adam.pt -d fakes_dataset

## python solution/grad_cam_analysis.py -m SimpleNet -cpp checkpoints/fakes_dataset_SimpleNet_Adam.pt -d fakes_dataset

## python solution/grad_cam_analysis.py -m SimpleNet -cpp checkpoints/synthetic_dataset_SimpleNet_Adam.pt -d synthetic_dataset

## python solution/grad_cam_analysis.py -m XceptionBased -cpp checkpoints/synthetic_dataset_XceptionBased_Adam.pt -d synthetic_dataset
