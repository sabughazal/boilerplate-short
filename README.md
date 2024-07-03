# boilerplate-short
 A shorter version of the boilerplate template for ML experiments.

### Current Features
- Run Evaluation at a specified epochs interval.
- Saving checkpoints of the model weights at a specified epochs interval.
- Saving the best checkpoint based on the evaluation loss.
- Logging training loss and evaluation loss on Tensorboard.
- Supports tracking different runs and stores the runs outputs.
- Storing run logs that include a summary of the model, the provided inline arguments for the run, the checkpoints, and the Tensorboard logs.

### Difference from the full boilerplate
- Does not support configurable multiple models, only 1 model at a time.
- Does not support configurable multiple datasets, only 1 dataset at a time.
- Does not support configurable multiple optimizers, only 1 optimizer at a time.
- Does not support configurable multiple loss functions, only 1 loss function at a time.
- Does not support learning rate schedulers.
- Does not use config files, only inline arguments.
- Does not support Weights and Biases.
- Does not support resuming from checkpoints.
- Does not support early stopping (patience argument).
- Does not have formatted logs, only standard prints to console.
