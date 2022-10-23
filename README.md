# Automated Program Repair Based on Code Review: How do Pre-trained Transformer Models Perform?

This repository contains the datasets and source codes on which we experimented for _"Automated Program Repair Based on Code Review: How do Pre-trained Transformer Models Perform?"_.

# Fine-tuned Models Download Link

## Fine-tuned Models on the Review4Repair dataset

| Index |    Model Name     |                                       Link                                        |
| :---: | :---------------: | :-------------------------------------------------------------------------------: |
|   1   | Fine-tuned CodeT5 | [link](https://mega.nz/file/zrg1GBaL#6cga1sF86JnPTABpNDlXyU7_6amGCIotHRMtgBuspis) |
|   2   | Fine-tuned PLBART | [link](https://mega.nz/file/P7ATGSbQ#to8fFtQwD3frDIvUNVROCgbDoKGT8lucefz-cFXHQfk) |

## Fine-tuned Models on the dataset by Tufano et.al

| Index |    Model Name     |                                       Link                                        |
| :---: | :---------------: | :-------------------------------------------------------------------------------: |
|   1   | Fine-tuned CodeT5 | [link](https://mega.nz/file/3yYWia4R#vKEk7-Tl1ZCk-r4PlaSpkXPgDK5pgtpExSVTiz07n-c) |
|   2   | Fine-tuned PLBART | [link](https://mega.nz/file/mjpTVLIC#nXuuES3HP6Lp9ShMcKGVOL-4I4lZI7Lqqd9bdxy0uKM) |

# **Inference and Evaluation**

## For CodeT5 model:

Steps:

- First go to the [`APR-using-Pre-trained-models/CodeT5/repo`](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/main/CodeT5/repo) directory
- Install the requiremnts using the following command:

  ```
  pip3 install -r <path of requirements.txt file>
  ```

  e.g:

  ```
  pip3 install -r /content/APR-using-Pre-trained-models/CodeT5/repo/requirements.txt
  ```

- Set your own working directory in the _WORKDIR_ of [exp_with_args.sh](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/blob/42aa3f61e6e94f8bd68ef3475be332ad063bab01/CodeT5/repo/sh/exp_with_args.sh#L1)

- Download your desired models from the [download link](#fine-tuned-models-download-link) provided above and save it in a directory.

- Go the the [CodeT5/repo/sh](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/main/CodeT5/repo/sh) directory.

- Run the inference using the following command:

  ```
  python3 run_exp.py  --model_tag codet5_base \
                      --task <task> \
                      --sub_task <sub_task> \
                      --nbest <nbest> \
                      --test_model_path <test_model_path>
  ```

  **Parameters description:**

  - **`model_tag:`** It represents which model we used for fine-tuning and it is always fixed to `codet5_base` as we used this model.

  - **`task:`** It represents on which datatset we want to do inference. For the Review4Repair dataset, `task` should be `refine_R4R` and for the dataset by Tufano et al., `task` should be `refine_tufano`.

  - **`sub_task:`** It represents the type of the dataset i.e. with or without code review. If the dataset doesn't contain code review, then `sub_task` should be `c`. On the other hand, if the dataset contains code review, then `sub_task` should be `cc`.

  - **`nbest:`** It represents which accuracy we want to calculate i.e. Top-1, Top-5 or Top-10 accuracy.
    So `nbest` value should be `1`, `5` or `10`.

  - **`test_model_path:`** It represents the path to the fine-tuned model we want to do inference.

  _For example, to run inference on the Review4Repair dataset with code review and to see the Top-10 accuracy the command can be like this:_

  ```
  python3 run_exp.py  --model_tag codet5_base \
                      --task refine_R4R \
                      --sub_task cc \
                      --nbest 10 \
                      --test_model_path /content/pytorch_model.bin
  ```

## For PLBART model:

Steps:

- Pre-requisite install `conda` from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):

- Download your desired models from the [download link](#fine-tuned-models-download-link) provided above and save it in a directory.

- Go to the [`APR-using-Pre-trained-models/PLBART`](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/main/PLBART) directory

- Setup the environment using the following command:
  ```
  bash install_env.sh
  ```
- Go to the [scripts/code_refinement](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/main/PLBART/scripts/code_refinement) folder
- Then tokenize the data using the following command:
  ```
  bash prepare.sh <task> <sub_task>
  ```
- Run the inference using the following command:

  ```
  bash run.sh <gpu_id> <dataset_size> <task> <sub_task> <nbest> <test_model_path>
  ```

  **Parameters description:**

  - **`gpu_id:`** It represents gpu id, usually `0` for single gpu.

  - **`dataset_size:`** It represents the dataset size for the setting of `batch size`. It can be `small` or `medium`.

  - **`task:`** It represents on which datatset we want to do inference. For the Review4Repair dataset, `task` should be `refine_R4R` and for the dataset by Tufano et al., `task` should be `refine_tufano`.

  - **`sub_task:`** It represents the type of the dataset i.e. with or without code review. If the dataset doesn't contain code review, then `sub_task` should be `c`. On the other hand, if the dataset contains code review, then `sub_task` should be `cc`.

  - **`nbest:`** It represents which accuracy we want to calculate i.e. Top-1, Top-5 or Top-10 accuracy.
    So `nbest` value should be `1`, `5` or `10`.

  - **`test_model_path:`** It represents the path to the fine-tuned model we want to do inference.

  _For example, to run inference on the Review4Repair dataset with code review and to see the Top-10 accuracy the command can be like this:_

  ```
  !bash run.sh 0 medium refine_R4R cc 10 /content/checkpoint_best.pt
  ```

# Pre-processed Datasets

The datasets can be found in this [directory](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/42aa3f61e6e94f8bd68ef3475be332ad063bab01/CodeT5/repo/data)

- [refine_R4R](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/42aa3f61e6e94f8bd68ef3475be332ad063bab01/CodeT5/repo/data/refine_R4R) directory holds the Review4Repair dataset.
- [refine_tufano](https://github.com/APR-using-Pre-trained-Models/APR-using-Pre-trained-models/tree/42aa3f61e6e94f8bd68ef3475be332ad063bab01/CodeT5/repo/data/refine_tufano) directory holds the dataset by Tufano et al.

In both of the directories mentioned above, there are two sub-directories:

- **`c directory:`** It is the dataset without code review
- **`cc directory:`** It is the dataset with code review.
