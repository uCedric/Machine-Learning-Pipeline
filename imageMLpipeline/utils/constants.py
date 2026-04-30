# List of supported models, loss functions, optimizers, and schedulers
MODEL_DICT = {"RESNET18": "resnet18"}
LOSS_DICT = {"CROSS_ENTROPY": "cross_entropy"}
OPTIMIZER_DICT = {"ADAM": "adam"}
SCHEDULER_DICT = {"STEP_LR": "step_lr", "COSINE_ANNEALING_LR": "cosine_annealing_lr"}
SCHEDULER_REQUIRED_PARAMS = {
    "step_lr": ["step_size"],
    "cosine_annealing_lr": ["T_max"],
}