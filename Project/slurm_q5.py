import datetime
from simple_slurm import Slurm

kwargs = dict(
    cpus_per_task=1,
    job_name=f"or_cv_project_q5",
    output=f"logs/slurm_q5.out",
    error=f"logs/slurm_q5.err",
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
python solution/train_main.py -d fakes_dataset -m SimpleNet --lr 0.001 -b 32 -e 5 -o Adam\
"""
)
