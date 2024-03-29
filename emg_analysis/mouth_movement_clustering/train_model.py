############################################################
# Classifier comparison on gapes 
############################################################
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#wanted_event_types = ['gape','tongue protrusion','lateral tongue protrusion',]
wanted_event_types = ['gape','tongue protrusion',]

# Plot examples of all wanted events
plot_gape_frame = scored_gape_frame.loc[scored_gape_frame.event_type.isin(wanted_event_types)]

fig,ax = plt.subplots(len(wanted_event_types), 1, 
                      sharex=True, sharey=True,
                      figsize = (5,10))
for this_event, this_ax in zip(wanted_event_types, ax):
    this_dat = plot_gape_frame.loc[plot_gape_frame.event_type == this_event]
    this_plot_dat = this_dat.segment_raw.values.T
    for this_seg in this_plot_dat:
        this_ax.plot(this_seg, color='k', alpha=0.1)
    this_ax.set_title(this_event)
fig.savefig(os.path.join(plot_dir, 'wanted_event_examples'), dpi = 150,
            bbox_inches='tight')
plt.close(fig)

# Get count of each type of event
event_counts = scored_gape_frame.event_type.value_counts()
event_counts = event_counts.loc[wanted_event_types]

#classes = scored_gape_frame['event_type'].astype('category').cat.codes

n_cv = 500

############################################################
# One vs All

xgb_accuracy_list = []
xgb_confusion_list = []

for this_event_type in wanted_event_types:
    #this_event_type = wanted_event_types[0]

    # Train new classifier on data
    # And calculate cross-validation accuracy score
    X = np.stack(scored_gape_frame['features'].values)
    y = np.array(scored_gape_frame['event_type'].values == this_event_type) 
    # Pull out classes for stratified sampling

    if this_event_type == 'gape':
        JL_accuracy = accuracy_score(y, scored_gape_frame.classifier)
        JL_confusion = confusion_matrix(y, scored_gape_frame.classifier,
                                        normalize = 'all')

    xgb_accuracy = []
    xgb_confusion = []
    for i in tqdm(range(n_cv)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.3,
                                                            #stratify=classes
                                                            )
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cv_accuracy = accuracy_score(y_test, y_pred)
        cv_confusion = confusion_matrix(y_test, y_pred, normalize = 'all')
        xgb_accuracy.append(cv_accuracy)
        xgb_confusion.append(cv_confusion)

    xgb_accuracy = np.array(xgb_accuracy)
    xgb_confusion = np.array(xgb_confusion)

    xgb_accuracy_list.append(xgb_accuracy)
    xgb_confusion_list.append(xgb_confusion)

xgb_accuracy_list = np.stack(xgb_accuracy_list)
np.save(os.path.join(code_dir, 'data', 'xgb_one_vs_all_accuracy_list.npy'), xgb_accuracy_list)

# Histograms of accuracy per event type
cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(1,1)
for i, this_event_type in enumerate(wanted_event_types):
    this_accuracy = xgb_accuracy_list[i]
    ax.hist(this_accuracy, label=this_event_type.title(), 
            alpha=0.5, bins = np.linspace(0,1), color = cmap(i))
    ax.hist(this_accuracy, 
            bins = np.linspace(0,1), histtype = 'step',
            color = cmap(i))
ax.axvline(JL_accuracy, color = 'k', linewidth = 5, alpha = 0.7,
           linestyle = '--', label = 'JL Classifier Gapes')
ax.legend()
ax.set_xlabel('Cross-validated Accuracy')
ax.set_ylabel('Count')
ax.set_title('Classification of Mouth Movements (One vs All)')
fig.savefig(os.path.join(plot_dir, 'classification_accuracy.svg'), 
                         bbox_inches='tight')
plt.close(fig)

############################################################
# Multiclass 

# Train new classifier on data
# And calculate cross-validation accuracy score
X = np.stack(scored_gape_frame['features'].values)
y = scored_gape_frame['event_type']
y_bool = [x in wanted_event_types for x in y] 
X = X[y_bool]
y = y[y_bool]
y_labels = y.astype('category').cat.categories.values
y = y.astype('category').cat.codes

xgb_accuracy = []
xgb_confusion = []
y_test_list = []
y_pred_list = []
for i in tqdm(range(n_cv)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3,
                                                        )
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_accuracy = accuracy_score(y_test, y_pred)
    cv_confusion = confusion_matrix(y_test, y_pred, normalize = 'all')
    xgb_accuracy.append(cv_accuracy)
    xgb_confusion.append(cv_confusion)
    y_test_list.append(y_test)
    y_pred_list.append(y_pred)

xgb_accuracy = np.array(xgb_accuracy)
xgb_confusion = np.array(xgb_confusion)
y_test_list = np.array(y_test_list)
y_pred_list = np.array(y_pred_list)

# Average confusion matrix
# Only take cases with all 3 labels
label_len = len(wanted_event_types)
wanted_xgb_confusion = [x for x in xgb_confusion if x.shape == (label_len,label_len)]
avg_confusion = np.mean(wanted_xgb_confusion, axis = 0)
std_confusion = np.std(wanted_xgb_confusion, axis = 0)

# Normalize over predicted
norm_avg_confusion = avg_confusion / avg_confusion.sum(axis=-1)[:,None]
norm_std_confusion = std_confusion / avg_confusion.sum(axis=-1)[:,None]

plt.matshow(norm_avg_confusion, vmin = 0, vmax = 1)
plt.xticks(range(label_len), y_labels, rotation = 45)
plt.yticks(range(label_len), y_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Average Confusion Matrix')
plt.colorbar(label = 'Fraction of Predictions')
# Also plot text in each square
for i in range(label_len):
    for j in range(label_len):
        plt.text(j, i, '{:.2f}'.format(norm_avg_confusion[i,j]) + '\n' + '± {:.2f}'.format(norm_std_confusion[i,j]), 
                 horizontalalignment="center", 
                 verticalalignment="center",
                 color="white" if norm_avg_confusion[i,j] < 0.5 else "black")
plt.savefig(os.path.join(plot_dir, 'average_confusion_matrix.svg'),
            bbox_inches='tight')
plt.close()

# Plot average accuracy
fig, ax = plt.subplots(1,1)
ax.hist(xgb_accuracy, bins = np.linspace(0,1))
ax.set_xlabel('Cross-validated Accuracy')
ax.set_ylabel('Count')
ax.set_title('Classification of Mouth Movements (Multiclass)')
fig.savefig(os.path.join(plot_dir, 'classification_accuracy_multiclass.svg'),
            bbox_inches='tight')
plt.close(fig)

############################################################
# Assess differentiability of lateral tongue protrusions
############################################################
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

# Train NCA
nca_model = NCA(n_components=2)
nca_model.fit(X, y)
nca_out = nca_model.transform(X)

plot_c_list = [cmap(i) for i in [0,2]]
fig,ax = plt.subplots()
for i, this_event_type in enumerate(y_labels):
    this_nca_out = nca_out[y == i]
    ax.scatter(this_nca_out[:,0], this_nca_out[:,1], 
               label = this_event_type.title(), alpha = 0.7,
               c = plot_c_list[i])
## Plot 3D scatter 
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for i, this_event_type in enumerate(y_labels):
#    this_nca_out = nca_out[y == i]
#    if not this_event_type == 'lateral tongue protrusion':
#        ax.scatter(this_nca_out[:,0], this_nca_out[:,1], this_nca_out[:,2],
#                   label = this_event_type.title(), alpha = 0.7)
#    else:
#        ax.scatter(this_nca_out[:,0], this_nca_out[:,1], this_nca_out[:,2],
#                   label = this_event_type.title(), s = 50,
#                   color = 'k')
ax.legend()
ax.set_xlabel('NCA 1')
ax.set_ylabel('NCA 2')
ax.set_aspect('equal')
#ax.set_zlabel('NCA 3')
#ax.set_title('NCA of Mouth Movements')
#plt.show()
fig.savefig(os.path.join(plot_dir, 'nca.svg'),
            bbox_inches='tight')
plt.close(fig)

# Train NCA pairwise
combs = list(itertools.combinations(range(3), 2))

trans_x = []
trans_y = []
for this_comb in combs:
    this_y_bool = np.array([x in this_comb for x in y])
    this_y = y[this_y_bool]
    this_x = X[this_y_bool]

    nca_model = NCA(n_components=2)
    nca_model.fit(this_x, this_y)
    nca_out = nca_model.transform(this_x)

    trans_x.append(nca_out)
    trans_y.append(this_y)

# Plot scatter plots for each NCA
fig, ax = plt.subplots(1, len(combs), figsize = (15, 5))
for i in range(len(combs)):
    this_comb = combs[i]
    this_titles = [y_labels[x] for x in this_comb]
    ax[i].scatter(*trans_x[i].T, c = trans_y[i])
    ax[i].set_xlabel('NCA 1')
    ax[i].set_ylabel('NCA 2')
plt.show()

# Cross-validate XGBoost on balanced samples vs lateral tongue protrusion
code_inds = {i: np.where(y.values == i)[0] for i in y.unique()}
min_count = min([len(x) for x in code_inds.values()])

n_repeats = 500
# For each repeat, draw equal samples for each label 
# Train classifier with cross-validations and measure accuracy
xgb_accuracy = []
xgb_confusion = []
for i in tqdm(range(n_repeats)):
    wanted_y_inds = np.concatenate(
            [np.random.choice(x, min_count, replace = False) \
                    for x in code_inds.values()])

    this_y = y.values[wanted_y_inds]
    this_x = X[wanted_y_inds]

    X_train, X_test, y_train, y_test = \
            train_test_split(this_x, this_y, 
                            test_size=0.3,
                                )
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_accuracy = accuracy_score(y_test, y_pred)
    cv_confusion = confusion_matrix(y_test, y_pred)
    xgb_accuracy.append(cv_accuracy)
    xgb_confusion.append(cv_confusion)

xgb_accuracy = np.array(xgb_accuracy)
xgb_confusion = np.array(xgb_confusion)

# Average confusion matrix
# Only take cases with all 3 labels
wanted_xgb_confusion = [x for x in xgb_confusion if x.shape == (3,3)]
avg_confusion = np.mean(wanted_xgb_confusion, axis = 0)
std_confusion = np.std(wanted_xgb_confusion, axis = 0)

# Normalize over predicted
norm_avg_confusion = avg_confusion / avg_confusion.sum(axis=-1)[:,None]
norm_std_confusion = std_confusion / avg_confusion.sum(axis=-1)[:,None]

plt.matshow(norm_avg_confusion)
plt.xticks(range(3), y_labels, rotation = 45)
plt.yticks(range(3), y_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Average Confusion Matrix')
plt.colorbar(label = 'Fraction of Predictions')
# Also plot text in each square
vmin, vmax = norm_avg_confusion.min(), norm_avg_confusion.max()
v_mid = (vmin + vmax) / 2
for i in range(3):
    for j in range(3):
        plt.text(j, i, '{:.2f}'.format(norm_avg_confusion[i,j]) + '\n' + '± {:.2f}'.format(norm_std_confusion[i,j]), 
                 horizontalalignment="center", 
                 verticalalignment="center",
                 color="white" if norm_avg_confusion[i,j] < v_mid else "black")
plt.savefig(os.path.join(plot_dir, 'average_confusion_matrix_balanced.svg'),
            bbox_inches='tight')
plt.close()

############################################################
# Train model on full dataset to get SHAP values
############################################################

X_frame = pd.DataFrame(data = X, columns = feature_names)

model = XGBClassifier().fit(X_frame, y) 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_frame)

shap.summary_plot(shap_values, X_frame, show=False)
plt.savefig(os.path.join(plot_dir, 'shap_summary_plot.svg'),
            bbox_inches='tight')
plt.close()

fig = sns.pairplot(X_frame)
fig.savefig(os.path.join(plot_dir, 'feature_pairplot'),
            bbox_inches='tight')
plt.close(fig)

# Spearman correlation for all features
corr = X_frame.corr(method = 'spearman')

fig = plt.matshow(np.abs(corr))
plt.xticks(range(len(feature_names)), feature_names, rotation = 45)
plt.yticks(range(len(feature_names)), feature_names)
plt.title('Spearman Correlation')
plt.colorbar(label = 'Correlation')
plt.savefig(os.path.join(plot_dir, 'feature_correlation.svg'),
            bbox_inches='tight')
plt.close()

# PCA of features
# Keep 95% of variance explained
pca = PCA()
pca.fit(X_frame)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cum_var > 0.95)[0][0] + 1

plt.plot(cum_var, '-x');plt.show()

