## Facebook Checkin Predictions

### Sample Workflow

### Linux
head -n 400000 test.csv > fb-checkins/data/test1.csv
head -n 800000 train.csv > fb-checkins/data/train1.csv
sed -n 800001,1600000p train.csv > fb-checkins/data/train2.csv
sed -n 3600000,8607231p test.csv > fb-checkins/data/test34.csv
sed -n 400000,1600000p test.csv > fb-checkins/data/test2.csv
sed -n 400000,800000p test.csv > fb-checkins/data/test1_5.csv

sed -n 800001,3600000p test.csv > fb-checkins/data/test2.csv  <-- ran this


tail -n 4,000,000 A.txt > B.txt

### Facts
test.csv: line count - 8607231

### Python
data = pd.read_csv('../data/train_10.csv')
test1 = pd.read_csv('../../test.csv')
estim = ut.doall(data,{},40,False,False,True)
preds1 = ut.doall(test1,{},40,False,estim,False)

preds1 = predictions
pr1 = list(preds1[0])
df_final = pd.DataFrame(pr1)
df_final.columns = ['row_id','location']

for j in range(1,len(preds1)):
  aa = list(preds1[j])
  df_temp = pd.DataFrame(aa)
  df_temp.columns = ['row_id','location']
  df_final = df_final.append(df_temp)

print('df_final length:')
print(len(df_final))



#### Post Analysis

**Steps**
- Fit Random Forests on 20x20 grid, with 100,000 rows
   - Average the feature importances
   - Average prediction accuracy per grid
   - Look at misclassified rows
      - avg x,y accuracy
      - avg samples in grid --> misclassified heatmap/2d histogram
      - most common place_ids that were misclassified
      - take all of the misclassified row ids
         - run some sort of clustering algorithm on them
         - we can break them into groups
      - think about this:
         - we have certain misclassified rows that have certain attributes, but maybe
           our main concern should be with a well-classified nearby place (that may be)
           causing the model to mess up. This of course will not show up as a particular
           "attribute" under the clustering of misclassified rows.
- Histograms
  - value counts...everything

- Extra variables
   - avg dist to others / grid location density
      - although, we technically do this by fitting so many random forests
   - time -> day, hr
   - grids --> districts
   - maybe it's more meaningful to split the dataset by time or accuracy

**Plot and Print Diagnostics**
`results = ut.train_and_predict(40, 0, 1, 'bottomleft', 250, 7, True)
plt.plot(results[1][0:1000])
plt.show()
a = ['row_id', 'x', 'y', 'accuracy', 'day', 'hr']

avg_importances = []
avg_importances = [x.feature_importances_ for x in models]
[list(x) for x in avg_importances]
avg_imp = [list(x) for x in avg_importances]
avgimp = list(map(list, zip(*avg_imp)))
average_importances = [np.average(x) for x in avgimp]

---------

a = ['x', 'y', 'accuracy', 'day', 'hr']
left_bar = list(range(0,5))
left_bar = [x/10 for x in left_bar]
fig, ax = plt.subplots()
ax.bar(left_bar, importances, width=0.08)
ax.set_title('Feature Importances')
ax.set_xticklabels(a)
tstamp = time.localtime()
tstamp = str(tstamp.tm_mon) + '-' + str(tstamp.tm_mday) + '-' + str(tstamp.tm_min)
plt.savefig('../visualizations/feature-importances-' + tstamp + '.png')
plt.show()`

**Run**
results = ut.train_and_predict(40, 0, 0, 'topleft', 250, 7, False)
predictions = results[0]
fit_scores = results[1]
grid_pops = results[2]
models = results[3]
misclassified = results[4]

**Turn results to writable dataframe**
pr1 = predictions[0]
df_final = pd.DataFrame(pr1)

df_final.columns = ['row_id','location']
for j in range(1,len(predictions)):
  df_temp = pd.DataFrame(predictions[j])
  df_temp.columns = ['row_id','location']
  df_final = df_final.append(df_temp)

**Format Merged Predictions**
merged_copy.to_csv('../data/new_results/triple_preds/bottomright.csv', index=False)

df_final.to_csv('../data/new_results/train12_preds/topleft_predictions.csv', index=False)
**Gather all the csv files**
results = pd.read_csv('../data/new_results/triple_preds/topleft.csv')
results2 = pd.read_csv('../data/new_results/triple_preds/topright.csv')
results = results.append(results2)
results2 = pd.read_csv('../data/new_results/triple_preds/bottomleft.csv')
results = results.append(results2)
results2 = pd.read_csv('../data/new_results/triple_preds/bottomright.csv')
results = results.append(results2)
results.columns = ['row_id', 'place_id']

**Make sure we have enough predictions**
test = pd.read_csv('../../test.csv')
a = results.row_id.ravel()
b = test.row_id.ravel()
print(len(a))
print(len(b))
diff = set(b) - set(a)
alist = list(diff)
for i in range(0,len(alist)):
    f = alist[i]
    a = pd.DataFrame([[f, '8129178888 8129178888 8129178888']])
    a.columns = ['row_id','place_id']
    results = results.append(a)

**Compare Various Results**
results_orig.place_id.value_counts()[0:20]
results.place_id.value_counts()[0:20]


### Thoughts on Attributes of the Problem
First off, we are dealing with a highly multiclass classification problem. In this case,
(without breaking it up into subproblems) we would have one vs MANY classification (one vs x million).

#### Breaking Down the Problem
So one of our first challenges is to make this more manageable by breaking it down in a meaningful way.
**Grid**
The first, and perhaps most intuitive way, it to break down the problem as a grid. This is because we understand that x,y coordinates are somewhat of a definitive cutoff for certain locations. Although, certain place_ids, may appear in multiple locations (maybe due to multiple locations or just mistakes in checking in). But I'm going to go out on a limb here and say that initially we can fairly assume that location plays a very important part in where you have checked into.
**Time**
We can perhaps further subset the problem by time. Day of the week, hour etc.
**Accuracy**
Again, we may find that problems of accuracy less than a certain cutoff require an entirely different model than those with higher accuracy.
**Number of Occurrences**
We have filtered out any places that have appeared 3 or less times from the dataset of 15 million.
Perhaps a different model can be trained on these infrequent places. Otherwise we are resigned to getting these wrong.
-- Nevermind this is stupid, the test set obviously can't capture this information.

Now, in terms of decision trees and random forests, we see that our classifier is kind of doing this anyway. So once our problem is broken down to a manageable size, it may not be necessary to subset the problem further, since this information may be captured by our model.

What it isn't doing, however, is training an entirely different model. We may find this useful and will investigate further upon analyzing misclassified results.

**Outcome**
We opted initially for the grid approach.

#### Variable Creation

**X**
**Y**
**Accuracy**
**Time**
   - **Hr**
   - **Day**
**Block Density**
**FUCK**



#### Analyzing Misclassified Results

**Method**
We will run a quick training model on < 150,000 rows (at certain grid locations) to get an idea on which rows we are misclassifying.

From there, we will store the _row_id_ and the _place_id_. That's really all the information we need to do a lookup from the original dataframe
(We actually don't need the place_id but it's nice to have for a quick initial diagnostic if we don't want to look up every misclassified row.)
