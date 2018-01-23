Mask-Dataset-Generator:

This code takes handbag images off of a shopping catalogue (single object monochrome background objects) and uses background images of various shelves in order to insert the bags into the background images to form annotated object detection bounding boxes and masks.

1. First, several background images with shelves are collected into a directory `background/`.
2. Then collect catalogue images of the handbags (or other retail products) and save it in the directory format `bags/<class_names>/images` where `bags\` is the base dataset directory.
3. Now run:
```shell
    python3 create_dataset.py --bags_dir bags/ --background_dir background/
```

