## Density Estimation

**Definition**: Density estimation is an unsupervised learning problem that aims to model the underlying probability density function that governs the distribution of data points in a given dataset. It involves estimating the probability of observing data points within specific regions of the attribute space.

*   **Probability densities** are models that describe the underlying true distribution of our data. They allow us to quantify:
    
    *   The probability of extracting a sample within a region of the space.
    *   The fraction of population samples that lie within a region of the space.
    
*   **Divide and count**

    Partitioning the space into smaller regions and counting the number of samples within each region is a simple way of describing the observed distribution.

    It also gives an indication of what to expect if we extract more samples from the same population, i.e. the true distribution. Counts will not give useful quantitative answers, but we can transform them into rates.

*   **Non-parametric methods**

    *   **Definition:** Non-parametric methods for density estimation do not assume any specific shape for the probability density.

    *   **The histogram** is the simplest and best known non-parametric method for density estimation. A histogram is built by dividing the feature space into equal-sized regions called **bins**. The density is approximated by the fraction of samples that fall within each bin. If bins are small, the estimated density will be **spiky**. If they are big, the estimated density will be **flat** and the underlying structure will be lost.
    *   **Kernel methods** proceed by building an individual density around each sample first and then combining all the densities together. Individual densities have the same shape (the kernel), for instance a Gaussian.
*   **Parametric density estimation**

    *   **Definition:** Parametric approaches **specify the shape** of the probability density. The problem of density estimation consists of **estimating its parameters**.

    *   The **Gaussian distribution**, usually denoted by $\mathcal{N}(\mu, \Sigma)$.
    *   The Gaussian or normal distribution $\mathcal{N}(\mu, \sigma)$ is defined by two parameters $\mu$ and $\sigma$ describing its **location** and **width**. Its mathematical expression in a 1D attribute spaces is:
        $$
        p(x_1) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x_1-\mu}{\sigma})^2}
        $$
        $\mu$ is known as the **mean** and $\sigma$ is the **standard deviation**. The value $\sigma^2$ is known as the **variance**.
    *   The **Central Limit Theorem (CLT)** states that the **sum** of a large number of independent, random quantities has a **Gaussian** distribution. The CLT is perhaps the **main reason** why the Gaussian distribution is one of our favourite density models.
    *   The Gaussian distribution can be extended to 2D, 3D... attribute spaces:
        $$
        p(x) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
        $$
        Where $x = [x_1,...,x_P]^T$ contains all the attributes. The mean $\mu$ and covariance matrix $\Sigma$ describe the **position** and **shape** of the distribution.
    *   Given a Gaussian distribution $p(x_1, x_2)$, the following statements are equivalent:
        *   Attributes $x_1$ and $x_2$ are **independent** (e.g. we cannot predict the value of one based on the other).
        *   The covariance matrix $\Sigma$ is **diagonal**.
        *   $p(x_1, x_2)$ can be obtained as the **product** of the **marginal densities** $p(x_1)$ and $p(x_2)$, which are themselves Gaussian.
    *   If we are given a dataset consisting of $N$ samples $x_i$, the parameters of a Gaussian distribution can be estimated using **maximum likelihood** approaches.
        *   In 1D attribute spaces, the parameters $\mu$ and $\sigma^2$ can be estimated as:
            $$
            \hat{\mu} = \frac{1}{N}\sum_i x_i,\text{ and } \hat{\sigma}^2 = \frac{1}{N}\sum_i (x_i - \hat{\mu})^2
            $$
        *   In higher dimensional spaces, $\mu$ and $\Sigma$ are estimated as:
            $$
            \hat{\mu} = \frac{1}{N}\sum_i x_i,\text{ and } \hat{\Sigma} = \frac{1}{N}\sum_i (x_i - \hat{\mu})(x_i - \hat{\mu})^T
            $$
    *   **Mixture models**: Datasets can exhibit more than one mode (clumps) for which single Gaussian densities are not suitable. In such cases, mixture densities such as **Gaussian Mixture Models (GMM)** constitute a convenient choice. A GMM probability density is formulated as a **combination** of Gaussian densities $g_m(x)$ with their own mean $\mu_m$ and covariance matrix $\Sigma_m$:
        $$
        p(x) = \sum_m g_m(x)\pi_m
        $$
        where $\pi_m$ are the mixing coefficients.
*   **Applications**

    *   **Noise and outliers**: Samples extracted from the same population will always exhibit some level of randomness and deviate from the underlying pattern. Such deviations are known as noise. Sometimes a sample can be so different that we doubt its deviation is just due to noise. We call these samples outliers or anomalies. Outliers are samples that belong to a different population altogether.
    *   **Basic anomaly detection algorithm**: The main idea behind an anomaly detection algorithm is to quantify the probability of observing samples some distance away from the general pattern. If this probability is low, the sample is an anomaly. The main pattern is described by a probability density $p(x)$. If $p(x)$ is a multivariate Gaussian distribution, we proceed as follows:
        *   Estimate $\mu$ and $\Sigma$ from the dataset.
        *   Agree on a threshold value $T$.
        *   If $p(x_i) < T$, $x_i$ is an anomaly.
    *   **Classification: Estimating class densities**: Classifiers that apply Bayes rule turn posterior probabilities into priors and class densities. A class density $p(x|C)$ describes the distribution of samples in the predictor space for each class $C$. Class densities are obtained using density estimation methods. We need to fit a probability distribution for each class separately.
    *   **Naive Bayes classifiers**: Gaussian distributions are the most popular choice for class densities. In high-dimensional scenarios, the total number of parameters is very large: for $P$ predictors, we have $P$ (mean) + $P^2$ (covariance) parameters. Naive Bayes classifiers make the (naive) assumption that predictors are independent, hence a $P$-dimensional Gaussian distribution can be expressed as the product of its $P$ marginal distributions. As a result of this additional constraint, we need to obtain $P$ (means) + $P$ (variances) parameters, which reduces the risk of overfitting.
    *   **Clustering**: K-means can be seen as a version of GMM fitting, where the Gaussian distributions have the same diagonal covariance matrix and hence clusters tend to be spherical. GMM can be used as a clustering method that produces ellipsoidal clusters. First we fit $K$ Gaussian densities and then we assign each sample to the most likely density.
*   **Summary**
    *   Probability densities are models that allow us to calculate the probability of finding a sample in a region of the attribute space.
    *   Non-parametric methods do not assume any particular shape for the probability density, whereas parametric methods do.
    *   The Central Limit Theorem and other mathematical properties, make the Gaussian distribution one of the most popular choices.
    *   Probability densities can be used in many machine learning problems, such as anomaly detection, classification and clustering.