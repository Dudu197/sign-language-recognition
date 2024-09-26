# Training the model

This folder contains the code for training the model.

The file `model_training.py` contains the code for training the model using the dataset and hyperparameters.

Because some datasets require a different preprocessing, the code is prepared to receive the dataset DataFrame as a parameter.

The available hyperparameters are:

- `-d` or `--dataset_name`: The name of the dataset. It can be `minds` or `ufop`.
- `-s` or `--seed`: The seed for the random number generator.
- `-vp` or `--validate_people`: The people to be used for validation. It can be a list of integers separated by commas without space.
- `-tp` or `--test_people`: The people to be used for testing. It can be a list of integers separated by commas without space.
- `-lr` or `--learning_rate`: The learning rate for the optimizer.
- `-wd` or `--weight_decay`: The weight decay for the optimizer.
- `-im` or `--image_method`: The method to be used for image representation. It can be `Skeleton-DML`.
- `-m` or `--model`: The model to be used. It can be `resnet18`.
- `-r` or `--ref`: The reference for the model. It can be an integer, representing the reference number of the training (which will be saved in `99_model_output/results`.


## Running the code

### Batch

Since we train using the LOPO strategy, we need to run the code multiple times, changing the people used for validation and testing.

To run the code in batch, you can use the `run_all_batches.sh` script.

```bash
./run_all_batches.sh
```

### Single

To run the code for a single training, you can use the following command:

```bash
python model_training.py -d minds -s 42 -vp 1,2,3 -tp 4,5 -lr 0.001 -wd 0.0001 -im Skeleton-DML -m resnet18 -r 1
```
