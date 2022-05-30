# Tabular Playground Series May 2022

This repository contains my work for the May 2022 edition of the Tabular
Playground Series on Kaggle. There are a couple notebooks I used while
developing a baseline model included, but the once I had a working baseline, I
used python scripts and [DVC](https://dvc.org/) pipelines to track the project.

There is no ground-breaking machine learning happening in this model and it was
mostly pieced together based on forum posts, shared notebooks, and trial and
error. I was more focused on the workflow and incremental development of a
model. I did successfully implement ensembling in a model training pipeline
that can be enabled/disabled for either generating competition submissions or
training a single model for development and testing. It would be interesting to
explore other ways to handle this.

# Notes

When the `train.ensemble` parameter is set to false, one model will be trained,
but when it is set to true, 5 different models will be trained using KFold cross
validation on of the training dataset and the prediction stage will use the mean
prediction from each model in the ensemble when generating the `submission.csv`.

The predicition process takes some time for this competition so I also used a
`predict.predict` parameter to skip generating predictions during development
and testing.
