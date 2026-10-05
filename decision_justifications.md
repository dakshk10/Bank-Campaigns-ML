# Decision Justifications --- Reference Notebooks

This explains every non-obvious choice made in `01_eda`,
`02_supervised_learning`, and `03_clustering`. It mirrors the
assignment's own evidence framework --- each item is tagged:

-   **OBSERVED** --- a fact read directly off the data.
-   **ASSUMPTION** --- a methodological choice made for a stated reason,
    where a defensible alternative exists.
-   **LIMITATION** --- something the data/method cannot establish,
    flagged rather than hidden.


------------------------------------------------------------------------

## 1. EDA (`01_eda`)

### `duration` is dropped

**OBSERVED.** Call duration is only known once the call has ended.
**ASSUMPTION → decision.** Since the decision point is "who to call
*before* dialling," any feature only available after the call is
leakage. UCI's own dataset documentation flags this column explicitly
for realistic models. There's no defensible alternative here --- this is
a hard exclusion, not a judgment call.

### Economic indicators (`emp.var.rate`, `cons.price.idx`,

`cons.conf.idx`, `euribor3m`, `nr.employed`) are kept, with a caveat
**OBSERVED.** These are published macro figures, technically known
before any given call. **LIMITATION.** Because the dataset is
date-ordered, these five columns are near-constant within any given
month and change abruptly between months. In a temporal split, they can
act as a proxy for "which period is this row from" rather than genuine
client signal. Kept in because they are legitimately available, but
flagged as a reason the model's apparent skill may partly reflect period
rather than client quality.

### `pdays == 999` → `was_contacted_before` flag

**OBSERVED.** 999 is a sentinel for "never contacted," not a real day
count --- the real `pdays` values for previously-contacted clients range
0--27 days. **ASSUMPTION.** Feeding 999 directly into a distance- or
coefficient-based model (logistic regression, LDA/QDA) would imply
"contacted roughly three years ago," which is false. Adding a binary
flag and leaving `pdays` otherwise untouched is one defensible fix;
bucketing or capping `pdays` is an equally defensible alternative. Pick
one and justify it --- don't leave 999 raw without comment.

### `unknown` is treated as its own category, not imputed

**ASSUMPTION.** Imputing `unknown` (e.g., to the mode) would discard
information and risks being read as fabricating a value. Keeping it as
an explicit category lets the model use "client declined to answer" as a
signal if it's predictive, and is the simpler, more defensible default.
`default` is \~21% unknown --- worth naming explicitly since a field
that's one-fifth missing carries limited information regardless of how
it's encoded.

### Notebook scope has been consolidated

**OBSERVED.** The updated notebook is `01_eda.ipynb` and contains both
the data-audit reasoning and the visual EDA in one workflow. The visual
EDA is explicitly target-focused: each chart asks what separates
`y = yes` from `y = no` and whether that information is available before
the bank makes a call.

**ASSUMPTION.** Keeping the audit, feature-availability checks, visual
EDA, and modelling-oriented conclusions together makes the reasoning
easier to trace from raw data → evidence → feature decisions.

### Target balance is treated as a modelling constraint

**OBSERVED.** The dataset contains 41,188 rows and the positive class
(`y = yes`) is 11.27%.

**ASSUMPTION → decision.** Accuracy is not treated as the primary
measure of usefulness because an all-negative classifier would already
achieve roughly 89% accuracy while identifying no likely subscribers.
The EDA therefore emphasizes subscription rates, association strength,
and later precision/recall, PR-AUC, or top-k lift rather than accuracy
alone.

### Visual EDA is question-driven rather than column-driven

**ASSUMPTION → decision.** The notebook contains nine main visual
analyses rather than automatically plotting every variable. Each
visualization is tied to a specific question and is intended to support
a downstream modelling decision.

The visual sections are: 1. Numeric distributions and skew/outlier
behavior. 2. Numeric variables versus subscription outcome. 3. Job versus
subscription rate. 4. Previous campaign outcome versus subscription
rate. 5. Age versus subscription rate. 6. Contact effort (`campaign`)
versus subscription rate. 7. Subscription rate by month. 8. `euribor3m`
by month to test whether macro variables encode calendar period. 9.
Spearman correlation among numeric features to identify
redundancy/collinearity.

A Cramér's V table is also used to rank categorical associations with
the target.

### Numeric distributions use log-scaled counts

**ASSUMPTION.** `campaign` and `previous` have long right tails, so the
distribution plots use a logarithmic y-axis. This prevents the large
mass near the lower values from making the tail effectively invisible.

**LIMITATION.** A log-scaled count axis improves visualization but does
not change the underlying distribution or remove outliers.

### Outliers are shown, not silently removed

**OBSERVED.** `campaign`, `previous`, and `duration` have long tails.
The notebook reports IQR-based outlier shares and keeps those
observations in the analysis.

**ASSUMPTION → decision.** Outliers are not automatically deleted
because extreme contact counts and call durations are real observations
and may contain business information. Removing them without a domain
justification could discard signal.

### `duration` is visualized only as a leakage demonstration

**OBSERVED.** The EDA shows a strong separation in `duration` between
subscribers and non-subscribers; its standalone AUC is approximately
0.82, with median duration of 449 seconds for subscribers versus 164
seconds for non-subscribers.

**LIMITATION → decision.** This is not treated as a usable predictive
feature. Call duration is known only after the call has ended, so its
strong association is outcome-related leakage rather than pre-call
predictive signal. It remains excluded from supervised modelling.

### Subscription rate is preferred to category counts for categorical comparisons

**ASSUMPTION → decision.** For categorical variables such as `job` and
`poutcome`, the notebook plots the proportion subscribing within each
group rather than raw category counts. Raw counts mainly describe how
common a group is; subscription rate answers the business question of
how promising that group historically was.

**LIMITATION.** Confidence intervals and group sample sizes are shown
because a high rate in a small category can be unstable.

### `poutcome` is treated as the strongest categorical pre-call signal

**OBSERVED.** `poutcome` has the strongest categorical association with
`y` in the current EDA, with Cramér's V ≈ 0.32. Subscription is
approximately 65.1% after a previous successful campaign, 14.2% after a
previous failure, and 8.8% for clients with no previous campaign
outcome.

**ASSUMPTION → decision.** Previous campaign outcome is therefore
treated as an important candidate pre-call signal.

**LIMITATION.** This is an association, not evidence that a previous
successful campaign causes a later subscription.

### `pdays`, `previous`, and `poutcome` encode overlapping campaign history

**OBSERVED.** `pdays != 999`, `poutcome != nonexistent`, and
`previous > 0` identify the same previously-contacted population in the
audit.

**ASSUMPTION → decision.** The representation should be handled
deliberately rather than blindly including all three as independent
signals. The notebook recommends keeping one meaningful representation
of previous-contact status/history to avoid unnecessary duplication.

### Job is a meaningful categorical segmentation

**OBSERVED.** Job has Cramér's V ≈ 0.15 with the target. Students and
retirees have substantially higher historical subscription rates than
the overall base rate, while blue-collar workers are lower.

**ASSUMPTION → decision.** `job` remains a useful candidate feature for
supervised modelling and a useful variable for business segmentation.

### Age requires a nonlinear interpretation

**OBSERVED.** The age-band visualization shows a U-shaped relationship
rather than a simple monotonic trend. Subscription is higher among the
youngest and oldest groups and lower around middle age; the oldest
groups have relatively small sample sizes.

**ASSUMPTION → decision.** A purely linear age effect may be inadequate.
Binning or a spline/nonlinear representation should be considered during
modelling.

**LIMITATION.** The high subscription rates among the oldest age bands
have wider uncertainty because those groups contain relatively few
observations.

### Contact effort shows diminishing returns / negative association

**OBSERVED.** Subscription rate declines across the `campaign` buckets,
from approximately 13.0% for one contact to approximately 3.1% for 11+
contacts.

**ASSUMPTION → decision.** `campaign` should be considered carefully as
a feature, potentially with a transformation or buckets because of its
long tail and nonlinear relationship.

**LIMITATION.** The pattern is associative. It does not prove that
making more calls causes a client to become less likely to subscribe;
clients who are harder to convert may simply receive more calls.

### Month is informative but potentially confounded

**OBSERVED.** Subscription rates vary substantially by month. March,
September, October, and December show high rates, while May has a much
lower rate despite being the largest month by call volume.

**LIMITATION.** Several high-rate months have relatively small sample
sizes, so their estimates have wider uncertainty.

**ASSUMPTION → decision.** `month` may carry predictive information, but
it should not automatically be interpreted as a stable client
characteristic.

### Economic indicators are treated as possible calendar proxies

**OBSERVED.** The EDA shows that `euribor3m` varies strongly by month,
while the economic indicators are also highly correlated with one
another. In particular, `emp.var.rate`, `euribor3m`, and `nr.employed`
have Spearman correlations around 0.93--0.94.

**ASSUMPTION → decision.** These variables are retained as technically
available pre-call information but flagged as potentially encoding the
collection period rather than independent client-level signal.

**LIMITATION.** With month and macroeconomic variables moving together,
the EDA cannot cleanly separate client propensity from period effects.
Model performance should therefore be evaluated with and without these
variables and interpreted cautiously.

### Spearman correlation is used for numeric redundancy

**ASSUMPTION.** Spearman correlation is used rather than relying only on
Pearson correlation because the goal is to detect monotonic feature
redundancy without assuming linear relationships.

**OBSERVED.** The macroeconomic variables exhibit strong mutual
correlation, supporting the concern that several of them contain
overlapping information.

**ASSUMPTION.** `pdays = 999` is converted to missing for this
correlation analysis so that the artificial sentinel value does not
create a misleading numeric relationship.

### Cramér's V is used to rank categorical associations

**ASSUMPTION.** Cramér's V is used alongside the categorical association
checks because with 41k observations, chi-square p-values alone are not
useful for ranking practical strength: very small associations can
become statistically significant.

**OBSERVED.** `poutcome` is strongest at approximately 0.32, followed by
`month` at approximately 0.27 and `job` at approximately 0.15. Variables
such as `day_of_week`, `housing`, and `loan` have very weak associations
in the current EDA.

### Weak variables are not given unnecessary visualizations

**ASSUMPTION → decision.** A dedicated day-of-week visualization was not
added because its Cramér's V is only about 0.03. Similarly, `housing`
and `loan` have very weak associations (about 0.01 and 0.005
respectively).

This is intentional: the EDA prioritizes visualizations that can answer
meaningful questions rather than generating one chart per column.

### Target balance and `unknown` bars are not duplicated

**ASSUMPTION → decision.** Separate visualizations for target balance
and `unknown` percentages were not added because the exact figures are
already printed in the audit sections. Repeating them visually would add
little analytical value.

### Pairplot and pie charts are intentionally omitted

**ASSUMPTION → decision.** A full pairplot is avoided because the
dataset has too many features for a readable all-variable pairplot. Pie
charts are also avoided because the EDA does not contain a key question
requiring a simple part-to-whole composition.

### Duplicate rows are flagged, not automatically deleted

**OBSERVED.** The current EDA identifies 12 exact duplicate rows across
all 21 columns, approximately 0.03% of the dataset.

**LIMITATION.** There is no client identifier, so these rows cannot be
proven to represent erroneous duplicates rather than repeated records
with identical observed attributes.

**ASSUMPTION → decision.** They are therefore retained and explicitly
documented rather than silently removed.

### Current EDA modelling implications

**OBSERVED / ASSUMPTION → decision.** - **Strong candidates:**
`poutcome` or an appropriate previous-contact representation, `job`,
`contact`, nonlinear `age`, and `campaign`. - **Exclude:** `duration`
because it is leakage. - **Use cautiously:** `month` and the five
macroeconomic indicators because of their shared calendar signal. -
**Weak standalone signals:** `day_of_week`, `housing`, and `loan`.

These are EDA-driven candidate decisions, not final claims about model
feature importance. Final selection must be validated using the
supervised-learning evaluation procedure.

### EDA evidence standard

**ASSUMPTION → decision.** The notebook explicitly distinguishes
observations from causal claims. A visualization can establish a pattern
or association in this dataset, but it cannot establish that a variable
causes a client to subscribe.

The EDA should therefore be read as: **data evidence → defensible
feature decision → modelling hypothesis**, not as proof of causal
business drivers.

------------------------------------------------------------------------

## 2. Supervised learning (`02_supervised_learning`)

### Temporal 80/20 split, not a random shuffle

**OBSERVED.** The dataset is ordered by date; a random split would let
the model train on rows that are chronologically *after* some of its
"test" rows, which the bank could never do in practice. **ASSUMPTION.**
80/20 is a starting point, not a rule. **OBSERVED, important:** at this
cut, the positive rate moves from **6.4% in train to 30.8% in the
holdout** --- the later period was a different, more successful
campaign. This is a real property of the data, not a bug. It means base
rate and lift are specific to the holdout period, and the 80/20 cut point
itself deserves sensitivity testing rather than being taken as fixed.

### Preprocessing fit on train only

**OBSERVED / standard practice.** `ColumnTransformer` is fit inside the
pipeline on `X_train` only; the holdout only ever calls `.transform()`.
This is non-negotiable for a valid holdout --- fitting a scaler or
encoder on data that includes the holdout leaks its distribution into
training.

### `OneHotEncoder(handle_unknown="ignore")`

**ASSUMPTION.** Any category value that appears in the holdout but not
in train gets an all-zero encoding instead of crashing the pipeline.
This is a reasonable default for a temporal split, since new category
values can genuinely appear later in time.

### Imbalance handling: `class_weight="balanced"`, compared against unweighted

**ASSUMPTION, tested not assumed.** Class weighting was chosen over
resampling (e.g. SMOTE) because it's simpler, requires no synthetic
data, and avoids the categorical-representation problems SMOTE raises
on one-hot features. Both weighted and unweighted logistic regression
are run side by side in the comparison table specifically so this choice
isn't taken on faith.

### Model choices: Logistic Regression, Random Forest (bagging), XGBoost (boosting), LDA, QDA

**ASSUMPTION.** These satisfy the assignment's required coverage (one
bagging method, one boosting method, LDA, QDA, plus a simple baseline)
without adding unjustified extra models. Hyperparameters are modest,
fixed values chosen to run quickly and avoid overfitting a small tuning
budget --- not the result of a tuning search.

### `QuadraticDiscriminantAnalysis(reg_param=0.1)`

**OBSERVED.** Without regularization, QDA fails outright --- the one-hot
categorical features make each class's covariance matrix singular
(`LinAlgError: not full rank`). **ASSUMPTION.** Adding `reg_param=0.1`
shrinks the covariance estimate enough to let QDA run, so there's a
number to compare rather than a crash. The underlying failure is still
reported, not hidden.

### Evaluation metric: `precision@500` chosen over accuracy as the primary business metric

**OBSERVED.** With an 11--31% positive rate depending on period, accuracy
rewards predicting the majority class and is not informative for a fixed
call budget. **ASSUMPTION.** `precision@500` and `lift@500` were chosen
because they directly answer the business question: of the top 500 calls
the model would recommend, how many actually subscribed historically.
ROC-AUC and PR-AUC are reported alongside for completeness but are not
the selection criterion.

### Final model selected by `precision@500`, not by ROC-AUC

**OBSERVED, and worth double-checking.** In the captured run, QDA had
the highest precision@500 despite a middling ROC-AUC. **LIMITATION.** This
result should be treated with suspicion and re-run with a different split
point or cross-validated top-k estimate before trusting that QDA genuinely
beats Random Forest and XGBoost.

### 0.5-threshold confusion matrix shown for completeness only

**ASSUMPTION.** The actual operating rule recommended to the bank is
"call the top 500 by score," not a 0.5 probability cutoff. The confusion
matrix at 0.5 is included as a standard evaluation artifact.

### Error analysis grouped by `poutcome`

**ASSUMPTION.** `poutcome` was chosen as the first error-analysis cut
because it's the strongest prior-history signal available pre-call. The
notebook reports the false-positive/false-negative rates by group without
pre-deciding what pattern would be found.

------------------------------------------------------------------------

## 3. Clustering (`03_clustering`)

### Clustering unit: contact records, not "clients"

**OBSERVED.** The dataset has no client identifier, and the same person
could plausibly appear more than once with different attribute snapshots.
**LIMITATION, stated explicitly.** Claiming "one row = one unique client"
is not supportable from this data, so all conclusions below are about
*contact records*, not verified individual people.

### Clustering fit on the train period only

**ASSUMPTION.** Uses the identical 80/20 temporal split as Task 1, so
that evaluating holdout subscription rate by cluster afterward is a
genuine out-of-sample check, not something the clustering could have seen
during fitting.

### Economic indicators excluded from clustering features

**ASSUMPTION.** Same reasoning as in the data audit: these columns mostly
encode time period rather than client characteristics, and the assignment
specifically calls out that risk for clustering. Excluded from the main
run; adding them back and comparing profiles is a valid follow-up test.

### Silhouette scored on a fixed 4,000-row sample, not the full 33k rows

**ASSUMPTION.** Silhouette score is O(n²) and too slow to compute
repeatedly on the full training set for multiple values of k. Sampling
4,000 rows with a fixed seed (`RANDOM_STATE`) makes this reproducible.

### K selected by sampled silhouette (k=3 in this run)

**OBSERVED.** Silhouette peaked at k=3 and dropped sharply for k≥4.
**LIMITATION.** An absolute silhouette around 0.32 is moderate, not
strong --- treat "3 clusters" as a reasonable starting point, not proof
of a strong natural grouping.

### Agglomerative clustering run on the same sample, for comparison

**ASSUMPTION.** `AgglomerativeClustering` doesn't scale to 33k rows by
default and has no `.predict()` for new data, so it's compared against
K-means on the identical 4,000-row sample. The Adjusted Rand Index between
the two methods' k=3 solutions is reported to assess agreement.

### Stability check across random seeds

**OBSERVED, important finding.** Refitting K-means with a different seed
produced an Adjusted Rand Index of **≈0.01** against the original
solution --- essentially no agreement. Combined with one cluster being
only ~0.5% of the training data, this indicates that the k=3 solution is
not robust.

**LIMITATION.** This instability should weigh directly on any verdict
about whether segmentation is useful.

### PCA plot captioned with its explained variance

**ASSUMPTION.** The 2D projection explains roughly 30% of total variance
in this run. The caption states that number explicitly so the plot isn't
mistaken for proof of separation in the full feature space.

### Cluster → holdout subscription rate as the Task 1 link

**ASSUMPTION.** K-means, rather than agglomerative clustering, was used
for this step because it supports `.predict()` on new/holdout rows without
refitting. The resulting table shows variation in subscription rate
across clusters in this run, but given the stability finding, treat that
variation as suggestive rather than confirmed.
