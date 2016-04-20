using ScikitLearnBase

## 1. Generate synthetic data from two distinct Gaussians: n_samples_A and
##    n_samples_B data points
## 2. Fit the GMM to it
## 3. Count how many points were classified into each Gaussian, test that
##    it's either n_samples_A or n_samples_B

n_samples_A = 300
n_samples_B = 600

# generate spherical data centered on (20, 20)
srand(42)
shifted_gaussian = randn(n_samples_A, 2) .+ [20, 20]'

# generate twice as many points from zero centered stretched Gaussian data
C = [0. -0.7
     3.5 .7]
stretched_gaussian = randn(n_samples_B, 2) * C

# concatenate the two datasets into the final training set
X_train = vcat(shifted_gaussian, stretched_gaussian)

# fit a Gaussian Mixture Model with two components
gmm = fit!(GMM(n_components=2, kind=:full), X_train)

# Check that the training points are correctly classified
@assert sum(predict(gmm, X_train) .== 1) in [n_samples_A, n_samples_B]
