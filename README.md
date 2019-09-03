# Machine Learning: Precipitation Binary Classifier #
### Data: ###
This one was a doozey, I'm not gunna lie. I had to transform my own data,
that I manually copy and pasted myself from:
https://www.wunderground.com/history/monthly/us/or/eugene/KEUG/date/2016-6

This was how it was all possible, I loaded them all up into txt files,
and then read them directly into pandas data structures using the loader.py
file. From there I used each file to run multiple tests of different methods.
Almost entirely using the sklearn package. 

Luckily I managed to get the loader.py file to process all the contents in the "data" directory,
so each file python file aside from loader.py is just a self-contained method with a few user-defined functions for flexibility.
A huge thing about this implementation is taht you can continuosly put in files of the same format.
I.e. if you place more month files title "(MM)-(YYYY).txt" with the same attributes
as the others, then each method file will continue to process that new file in addition to the older
ones. Nothing is hard-coded to provide more functionality


### Functionality: ### 

Each file can be ran on their own for results using:
>python3 file_method.py

All the files in here will yield significant results and print out their train and test data
respectively. Some of the files are more verbose with warnings than others, but they all run. Hopefully the print statements will be sufficient, but if they arent feel free to shoot me
an email with any questions: bryancarl7@gmail.com


### Significant results: ###

SVM seemed to perform the best overall, barely edging out Logistic Regression. Nearest Neighbor 
seemed to do quite well also with n=1 and n=3, and the decision tree seemed to do decently well.
Linear Regression performed by far the worst out of them all, not breaking 73%.

#### Support Vector Machine #####
SVM Mean Absolute Error on train   : 89.04109589041096%

SVM Mean Absolute Error on test    : 87.67123287671232%

#### Logistic Regression ####

Logistic Mean Absolute Error on train : 88.9269406392694%

Logistic Mean Absolute Error on test : 87.67123287671232%


#### Nearest Neirghbor #####
NN 2 Mean Absolute Error : 84.01826484018264%

NN 3 Mean Absolute Error : 84.47488584474886%

NN 5 Mean Absolute Error : 84.01826484018264%

NN 10 Mean Absolute Error : 83.10502283105023%

(All accuracies done on test set)

#### Decision Tree ####
Tree MeanAbsoluteError on training data  : 100%

Tree MeanAbsoluteError on test data      : 80.36529680365296%

#### Linear Regression ####
Linear MeanAbsoluteError on train: 73.11685042552742%

Linear MeanAbsoluteError on test : 72.43315178534038%
