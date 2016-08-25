## Facebook Checkin Predictions

### Data
- ~27 million rows
- Features: X, Y, Accuracy, Time
- Outcome Variable: Place Id

We are dealing with a highly multi-class classification problem. In this case,
(without breaking it up into subproblems) we would have one vs MANY classification (one vs x hundred-thousand).

#### Breaking Down the Problem
So one of our first challenges is to make this more manageable by breaking it down in a meaningful way.
- **Grid**
   - The first, and perhaps most intuitive way, is to break down the problem as a grid. This is because we understand that x,y coordinates are somewhat of a natural cutoff for certain locations. Although, certain place_ids, may appear in multiple locations (maybe due to multiple locations or just mistakes in checking in). But I'm going to go out on a limb here and say that initially we can fairly assume that location plays a very important part in where you have checked into.
- **Time**
   - We can perhaps further subset the problem by time. Day of the week, hour etc.
- **Accuracy**
   - Again, we may find that problems of accuracy less than a certain cutoff require an entirely different model than those with higher accuracy.
- **Number of Occurrences**
   - We have filtered out any places that have appeared 3 or less times from the dataset of 15 million.

Now, in terms of decision trees and random forests, we see that our classifier is kind of doing this anyway. So once our problem is broken down to a manageable size, it may not be necessary to subset the problem further, since this information may be captured by our model.

What it isn't doing, however, is training an entirely different model (per tree level). We may find this useful and will investigate further upon analyzing misclassified results.


#### Variable Creation

- **X**
- **Y**
- **X/Y and X*Y**
- **Accuracy**
- **Time**
   - **Hr**
   - **Day of Week**
   - **Day of Year**
   - **Month**


#### Analyzing Misclassified Results

**Method**
   - We will run a quick training model on < 150,000 rows (at certain grid locations) to get an idea on which rows we are misclassifying.

   - From there, we will store the _row_id_ and the _place_id_. That's really all the information we need to do a lookup from the original dataframe. (We actually don't need the place_id but it's nice to have for a quick initial diagnostic if we don't want to look up every misclassified row.)



#### Performance
*Some performance results within each loop*

```
Removes singles: 0.002156972885131836seconds
Subset dataset: 0.0010998249053955078seconds
Sort probabilities: 0.00013184547424316406seconds
Predict and sort: 0.21532678604125977seconds
Misclassified: 4.291534423828125e-05seconds
Format df: 0.005101919174194336seconds
Removes singles: 0.0021560192108154297seconds
Subset dataset: 0.0011570453643798828seconds
Sort probabilities: 0.0003879070281982422seconds
Predict and sort: 0.2116999626159668seconds
Misclassified: 4.887580871582031e-05seconds
Time: 13.810248136520386 seconds

Removes singles: 0.002377033233642578seconds
Subset dataset: 0.0013339519500732422seconds
Sort probabilities: 8.106231689453125e-05seconds
Predict and sort: 0.10548114776611328seconds
Misclassified: 1.811981201171875e-05seconds
Format df: 0.004248142242431641seconds
Removes singles: 0.0021049976348876953seconds
Subset dataset: 0.0010199546813964844seconds
Sort probabilities: 0.0002689361572265625seconds
Predict and sort: 0.10912394523620605seconds
Misclassified: 2.5987625122070312e-05seconds
Time: 12.555937051773071 seconds
```

### Final Results
I ended up placing somewhere in the middle of the pack.

First of all, we had model tuning results. These models were examined manually for the most part. Some of these results are:

*For kNN*
```
Fitting 2 folds for each of 3 candidates, totalling 6 fits
[Parallel(n_jobs=2)]: Done   6 out of   6 | elapsed:  1.2min finished
Best parameters set found on development set:
{'weights': 'distance', 'n_neighbors': 10, 'metric': 'manhattan'}
Grid scores on development set:
0.605 (+/-0.010) for {'weights': 'distance', 'n_neighbors': 10, 'metric': 'manhattan'}
0.602 (+/-0.010) for {'weights': 'distance', 'n_neighbors': 12, 'metric': 'manhattan'}
0.601 (+/-0.011) for {'weights': 'distance', 'n_neighbors': 14, 'metric': 'manhattan'}
Detailed classification report:
```

*Or similarly for random forests*
```
Best parameters set found on development set:
{'n_estimators': 400, 'max_depth': 8}
Grid scores on development set:
0.402 (+/-0.056) for {'n_estimators': 300, 'max_depth': 6}
0.441 (+/-0.047) for {'n_estimators': 300, 'max_depth': 7}
0.438 (+/-0.059) for {'n_estimators': 400, 'max_depth': 7}
0.450 (+/-0.044) for {'n_estimators': 400, 'max_depth': 8}
```

(Although it doesn't show it above, random forests actually performed better)

#### Looking at visualizations
None of these really seemed to be telling about why anything was misclassified. Most of the plots were meant to determine if there was any day, hour, region, accuracy level etc. that was causing most of my errors. The low accuracy rows did seem to cause the predictions to be--wait for it--less accurate, but besides that there wasn't much to be determined.

#### Further Analysis
Naturally, I looked through forums to get a better understanding of other's approaches and of where I may have gone wrong or fallen short. Particularly, I liked the blog post by the competition's 2nd place winner linked here: http://blog.kaggle.com/2016/08/02/facebook-v-predicting-check-ins-winners-interview-2nd-place-markus-kliegl/

In that post, Markus Kliegl discusses the use of a Naive Bayes Classifier and some of the data exploration and results he encountered. One interesting pattern in the data was the seasonality of checkins per place id. I actually didn't plot seasonality of place checkins and missed that concept during my model building process. Of course I introduced hour, day of year, and other time variables to the random forest and assumed that it would make the best possible decision at each split (factoring in seasonal behavior) but there could only be so much time-based granularity involved there (even with very deep decision trees).

In the end, having stronger knowledge about using Naive Bayes Classifiers might have been useful, especially because there were so many possible place_ids and the decision tree/random forest approach would likely only have predicted the relatively frequent place_ids. The catch, however, is that implementing the Naive Bayes Classifier would require a reasonably large about of preprocessing and more of a supervised approach. Still, I think it would have been an interesting way to go about solving the problem.
