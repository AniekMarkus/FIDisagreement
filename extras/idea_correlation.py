# Modules
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from matplotlib.backends.backend_pdf import PdfPages


# Get functions in other Python scripts
from help_data import *
from models import *
from feature_importance import *
from help_functions import *



X, y, coef_dgp, binary_outcome = load_data()

# correlogram:
pdf = PdfPages('idea_correlation.pdf')
correlogram(X, pdf)

# PCA:
# num_components = 10
# from sklearn.decomposition import PCA
#
# # define transform
# pca = PCA()
#
# # prepare transform on dataset
# components = pca.fit(X)
# # components = pd.DataFrame(components)
#
# # apply transform to dataset
# transformed = pca.transform(X)
#
# X_PCA = transformed[:, 1:num_components]

# ...
# total = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(total)
# plt.show()

# we can even visualize a biplot to understand the influence of features on principal components


# K means:
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from matplotlib import pyplot

# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

X_transpose = X.transpose()

# define the model
model = KMeans(n_clusters=10)
# fit the model
model.fit(X_transpose)
# assign a cluster to each example
yhat = model.predict(X_transpose)
# retrieve unique clusters
clusters = unique(yhat)

res = {i:yhat.tolist().count(i) for i in yhat.tolist()}
print(res)

# create scatter plot for samples from each cluster
# for cluster in clusters:
#     # get row indexes for samples with this cluster
#     row_ix = where(yhat == cluster)
#     # create scatter of these samples
#     pyplot.scatter(X_transpose[row_ix, 0], X_transpose[row_ix, 1])
# # show the plot
# pyplot.show()

# correlation each var and outcome:
corr1 = [correlation(X.iloc[:,i], y) for i in range(0, X.shape[1])]

plt.scatter(yhat, corr1) # todo: order features by cluster
plt.show()

# correlation each var and other outcomes:
corr2 = X.corr()

# distance each var and other outcomes:
# todo: what makes distance different from correlation?