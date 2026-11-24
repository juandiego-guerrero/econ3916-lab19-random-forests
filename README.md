# econ3916-lab19-random-forests
Lab 19 submission

# Tree-Based Models: Random Forests

## Objective
I compared a single decision tree, OLS, Ridge and a random forest on California Housing to see whether the forest's extra complexity improves predictions, which features it relies on, and how much tuning helps.

## Methodology
- Split California Housing (20,640 block groups, 8 features, median house value in $100,000s) into 80% training and 20% test rows.
- Fitted four models on the same training rows: an unlimited-depth decision tree, OLS, Ridge behind a `StandardScaler`, and a 100-tree random forest with out-of-bag scoring. I scored each on RMSE and R² for both sets.
- Compared two measures of feature importance for the forest: impurity-based (MDI), computed on the training rows, and permutation importance, the fall in test R² when one column is shuffled (10 repeats).
- Tuned `max_depth` (10 or no limit) and `max_features` ('sqrt' or 0.5) with 3-fold `GridSearchCV`, then scored the best forest once on the test rows.
- Wrote my own prompt asking an AI for a function that computes permutation importance and charts the top k features, then checked its output against my lab's numbers on the same forest and on an unseen tree.

## Key Findings
- The single tree fit its training rows perfectly (15,860 leaves for 16,512 rows) but reached a test R² of only 0.6228, because it memorised noise. Ridge reached 0.5758, the same as OLS, since with 8 features and over 16,000 rows the penalty had nothing to remove. The forest reached 0.8046, and its out-of-bag R² of 0.8070 landed close to that without using any test rows.
- The forest's 0.23 lead over Ridge with the same features means house values depend on the features in non-linear, interacting ways that a linear model cannot capture.
- Latitude scored 0.0890 by MDI but 0.4427 by permutation. MDI credits splits by how many training rows reach them, so it undervalues location, which the trees build from many deep splits shared between latitude and longitude. It also overrates noisy continuous features such as AveOccup.
- Tuning raised the test R² to 0.8094. Removing the depth limit mattered most in the grid; compared with my first forest, the only change was letting each split choose from 4 features instead of 8.
- The AI's function took 2 rounds. Every check line matched my permutation numbers at k = 8 and k = 3, no forest was refitted, and it matched `permutation_importance` run directly on an unseen tree. Adding `n_jobs=-1` in round 2 changed no number.
