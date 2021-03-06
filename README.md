# tf_multi_classification

A tensorflow script to create classification models.
Supports multiple classes. Follow the directory structure below.

## Dir Structure

```
checkpoints/
model/
data_set/
    prediction/
       **uncategorized_image_files
    training/
        class_1_name/
            **class_1_images
        class_2_name/
            **class_2_images
        .
        .
        .
        class_N_name/
            **class_N_images
    validation/
        class_1_name/
            **class_1_images
        class_2_name/
            **class_2_images
        .
        .
        .
        class_N_name/
            **class_N_images
```

## Workflow

1. Populate folders with data as shown above

2. Run train.py. This will save checkpoints and export the trained model after execution

3. Run `predict.py` to categorized images in the prediction folder.

Note: There is no nead to run `train.py` again after you have created a model/checkpoints that you are happy with
