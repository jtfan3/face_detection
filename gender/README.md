## Sorting gender data
Code found in `sort_gender.ipynb`
Python code to sort UTK data into following file structure

```

gender
    |__ 0
        |__some_male_0.jpg
        |__some_male_1.jpg
        |__some_male_2.jpg
    |__ 1
        |__some_female_0.jpg
        |__some_female_1.jpg
        |__some_female_2.jpg

```

### Training the gender model
Code for loading the data into tf dataset can be found at in trian_gender.ipynb

configurations for the model specs can be found in gender_config.py
    - data paths
    - image shape, rgb
    - dataset (UTK faces https://susanqq.github.io/UTKFace/)

Results for training the model
    - utilizes 85%, 15% of data for train, val
    - achieved accuracy of roughly 93% for both trian/val