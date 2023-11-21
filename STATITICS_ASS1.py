'''Q1'''
'''The Probability Density Function (PDF) is a concept in probability theory and statistics. It is a mathematical function that describes the likelihood of a continuous random variable falling within a particular range of values. In other words, the PDF provides a way to model the probability distribution of a continuous random variable.

For a continuous random variable \(X\), the probability density function is denoted as \(f(x)\), and it has the property that the probability of \(X\) lying in an interval \((a, b)\) is given by the integral of \(f(x)\) over that interval:

\[ P(a < X < b) = \int_{a}^{b} f(x) \,dx \]

Key properties of a probability density function:

1. **Non-Negativity:** The PDF is non-negative for all values of \(x\): \(f(x) \geq 0\).

2. **Total Area under the Curve:** The total area under the PDF curve over the entire range of possible values is equal to 1:

\[ \int_{-\infty}^{\infty} f(x) \,dx = 1 \]

3. **Probability at a Point:** The probability of a continuous random variable taking on any specific value is technically zero:

\[ P(X = x) = 0 \]

4. **Probability within an Interval:** The probability of \(X\) falling within a given interval is the integral of the PDF over that interval.

### Example:

Let's take an example of a standard normal distribution, which has a PDF denoted as \(f(x)\) for a random variable \(X\). The standard normal distribution has a mean (\(\mu\)) of 0 and a standard deviation (\(\sigma\)) of 1.

\[ f(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}} \]

In this case:
- The curve represents the PDF of the standard normal distribution.
- The area under the curve between two points on the x-axis represents the probability of the random variable falling within that range.

Understanding the PDF is fundamental in probability and statistics, especially when working with continuous random variables and conducting statistical analyses.'''

'''Q2'''
'''There are several types of probability distributions, and they can be broadly categorized into two main classes: discrete probability distributions and continuous probability distributions. Each class has its own specific distributions that model different types of random variables.

### Discrete Probability Distributions:

1. **Bernoulli Distribution:**
   - Models a single trial with two possible outcomes, usually labeled as success and failure. It is characterized by a probability parameter \(p\), representing the probability of success.

2. **Binomial Distribution:**
   - Describes the number of successes in a fixed number of independent Bernoulli trials. It is characterized by two parameters: the number of trials (\(n\)) and the probability of success (\(p\)).

3. **Poisson Distribution:**
   - Models the number of events occurring in a fixed interval of time or space. It is characterized by a single parameter (\(\lambda\)), which represents the average rate of occurrence.

### Continuous Probability Distributions:

1. **Uniform Distribution:**
   - All outcomes in a given range are equally likely. The probability density function is constant within the range.

2. **Normal Distribution (Gaussian Distribution):**
   - Characterized by a bell-shaped curve. Many natural phenomena follow a normal distribution. It is defined by two parameters: mean (\(\mu\)) and standard deviation (\(\sigma\)).

3. **Exponential Distribution:**
   - Models the time between events in a Poisson process. It is characterized by a rate parameter (\(\lambda\)).

4. **Gamma Distribution:**
   - Generalizes the exponential distribution and includes the exponential distribution as a special case. It is characterized by two parameters: shape (\(k\)) and scale (\(\theta\)).

5. **Beta Distribution:**
   - Often used to model the distribution of random variables that have values between 0 and 1. It is characterized by two shape parameters (\(\alpha\) and \(\beta\)).

6. **Cauchy Distribution:**
   - Has heavy tails and no defined mean or variance. It is characterized by a location parameter (\(x_0\)) and a scale parameter (\(\gamma\)).

These are just a few examples, and there are many other probability distributions that are used to model different types of random variables in various fields of study, including statistics, finance, engineering, and physics. The choice of the distribution depends on the characteristics of the data and the nature of the random variable being studied.'''

'''Q3'''
'''Certainly! You can use the probability density function (PDF) formula for the normal distribution to create a Python function. The PDF of the normal distribution with mean \(\mu\) and standard deviation \(\sigma\) at a given point \(x\) is given by:

\[ f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \]

Here's a Python function that calculates the PDF of a normal distribution:

```python
import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of a normal distribution at a given point.

    Parameters:
    - x: The point at which to evaluate the PDF.
    - mean: The mean of the normal distribution.
    - std_dev: The standard deviation of the normal distribution.

    Returns:
    - PDF value at the given point.
    """
    coefficient = 1 / (math.sqrt(2 * math.pi) * std_dev)
    exponent = -((x - mean)**2) / (2 * std_dev**2)
    pdf_value = coefficient * math.exp(exponent)
    return pdf_value

# Example usage:
mean_value = 0
std_dev_value = 1
x_value = 1.5

pdf_result = normal_pdf(x_value, mean_value, std_dev_value)
print(f'The PDF at x={x_value} is {pdf_result}')
```

In this example, the `normal_pdf` function takes three parameters: the point at which to evaluate the PDF (`x`), the mean of the normal distribution (`mean`), and the standard deviation of the normal distribution (`std_dev`). It then calculates and returns the PDF value at the given point using the normal distribution formula.'''

'''Q4'''
'''The Binomial distribution is a discrete probability distribution that describes the number of successes in a fixed number of independent Bernoulli trials, where each trial has only two possible outcomes: success or failure. The Binomial distribution has several key properties:

1. **Fixed Number of Trials (\(n\)):** The distribution is defined for a fixed number of trials (\(n\)), where each trial is independent of the others.

2. **Two Possible Outcomes:** Each trial results in one of two outcomes, often labeled as success (S) and failure (F).

3. **Constant Probability of Success (\(p\)):** The probability of success (\(p\)) remains constant across all trials.

4. **Discrete:** The random variable representing the number of successes (\(X\)) is discrete and takes on integer values from 0 to \(n\).

5. **Probability Mass Function (PMF):** The probability mass function of the Binomial distribution is given by the formula:

   \[ P(X = k) = \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k} \]

   where \(\binom{n}{k}\) is the binomial coefficient, representing the number of ways to choose \(k\) successes out of \(n\) trials.

6. **Mean and Variance:** The mean (\(\mu\)) and variance (\(\sigma^2\)) of the Binomial distribution are given by:

   \[ \mu = np \]

   \[ \sigma^2 = np(1-p) \]

### Examples of Events Modeled by Binomial Distribution:

1. **Coin Flipping:**
   - Example: Consider flipping a fair coin (where the probability of heads, \(p\), is 0.5) 10 times. The number of heads obtained in these 10 flips follows a Binomial distribution with parameters \(n = 10\) and \(p = 0.5\).

2. **Product Defects:**
   - Example: A manufacturing process produces light bulbs, and each bulb has a 5% chance of being defective. If we randomly select 20 bulbs, the number of defective bulbs among them follows a Binomial distribution with parameters \(n = 20\) and \(p = 0.05\).

Binomial distribution is commonly used in various fields to model the number of successes in a fixed number of independent trials with two possible outcomes.'''

'''Q5'''
import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Generate a random sample of size 1000 from a binomial distribution with p=0.4
sample_size = 1000
probability_of_success = 0.4

# Generate the random sample
random_sample = np.random.binomial(1, probability_of_success, size=sample_size)

# Plot the histogram
plt.hist(random_sample, bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7)
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.title(f'Binomial Distribution (n=1, p={probability_of_success}) - Random Sample')
plt.xticks([0, 1], ['Failure (0)', 'Success (1)'])
plt.show()

'''Q6'''
import math

def poisson_cdf(k, mean):
    """
    Calculate the cumulative distribution function (CDF) of a Poisson distribution at a given point.

    Parameters:
    - k: The point at which to evaluate the CDF.
    - mean: The mean of the Poisson distribution.

    Returns:
    - CDF value at the given point.
    """
    cdf_value = 0.0
    for i in range(int(k) + 1):
        cdf_value += math.exp(-mean) * (mean ** i) / math.factorial(i)
    return cdf_value

# Example usage:
mean_value = 2.5
point = 3

cdf_result = poisson_cdf(point, mean_value)
print(f'The CDF at {point} is {cdf_result}')

'''Q7'''
'''The Binomial and Poisson distributions are both probability distributions in statistics, but they are used to model different types of random phenomena. Here are the key differences between the Binomial and Poisson distributions:

### 1. Nature of the Random Phenomenon:

- **Binomial Distribution:**
  - The Binomial distribution is used to model the number of successes in a fixed number of independent Bernoulli trials (experiments with only two possible outcomes: success or failure).
  - The trials are independent, meaning the outcome of one trial does not affect the outcome of another.

- **Poisson Distribution:**
  - The Poisson distribution is used to model the number of events that occur in a fixed interval of time or space.
  - It is often applied when the events are rare, and the probability of an event occurring in a very small time or space interval is constant.

### 2. Number of Trials:

- **Binomial Distribution:**
  - Requires a fixed number of trials (\(n\)).

- **Poisson Distribution:**
  - Does not have a fixed number of trials; it is used in situations where the number of events is measured in a continuous manner.

### 3. Assumptions:

- **Binomial Distribution:**
  - Assumes a fixed number of trials (\(n\)).
  - Assumes the trials are independent.
  - Assumes a constant probability of success (\(p\)) for each trial.

- **Poisson Distribution:**
  - Assumes events occur independently.
  - Assumes the average rate of events (\(\lambda\)) is constant over time or space.

### 4. Probability Mass Function (PMF):

- **Binomial Distribution:**
  - The probability mass function is given by:
    \[ P(X = k) = \binom{n}{k} \cdot p^k \cdot (1 - p)^{n - k} \]

- **Poisson Distribution:**
  - The probability mass function is given by:
    \[ P(X = k) = \frac{e^{-\lambda} \cdot \lambda^k}{k!} \]

### 5. Application Examples:

- **Binomial Distribution:**
  - Modeling the number of successes in a fixed number of coin flips, where each flip is independent.
  - Counting the number of defective items in a sample of fixed size from a production line.

- **Poisson Distribution:**
  - Modeling the number of phone calls received at a call center in a fixed time interval.
  - Counting the number of emails received in a fixed time period.

### 6. Limiting Behavior:

- **Binomial Distribution:**
  - As the number of trials (\(n\)) increases and the probability of success (\(p\)) decreases in such a way that \(np\) remains constant, the Binomial distribution approaches the Poisson distribution.

- **Poisson Distribution:**
  - Is often used as an approximation for the Binomial distribution in situations where \(n\) is large and \(p\) is small.

In summary, while both the Binomial and Poisson distributions involve counting events, they are applied to different types of scenarios and have distinct assumptions and probability mass functions. The choice between them depends on the nature of the random phenomenon being modeled.'''

'''Q8'''
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate a random sample of size 1000 from a Poisson distribution with mean 5
sample_size = 1000
mean_value = 5
poisson_sample = np.random.poisson(mean_value, size=sample_size)

# Calculate the sample mean and variance
sample_mean = np.mean(poisson_sample)
sample_variance = np.var(poisson_sample, ddof=1)  # ddof=1 for sample variance

# Print the results
print(f'Sample Mean: {sample_mean}')
print(f'Sample Variance: {sample_variance}')

'''Q9'''
'''In both the Binomial and Poisson distributions, the mean (\(\mu\)) and variance (\(\sigma^2\)) are related, but the specific formulas differ due to the nature of each distribution.

### Binomial Distribution:

For a Binomial distribution with parameters \(n\) (number of trials) and \(p\) (probability of success in a single trial), the mean and variance are related as follows:

- **Mean (\(\mu\)):**
  \[ \mu = np \]

- **Variance (\(\sigma^2\)):**
  \[ \sigma^2 = np(1 - p) \]

In the Binomial distribution, the variance is directly influenced by the number of trials (\(n\)) and the probability of success (\(p\)). As the number of trials or the probability of success increases, the variance tends to increase as well.

### Poisson Distribution:

For a Poisson distribution with parameter \(\lambda\) (average rate of events occurring in a fixed interval), the mean and variance are related as follows:

- **Mean (\(\mu\)):**
  \[ \mu = \lambda \]

- **Variance (\(\sigma^2)):**
  \[ \sigma^2 = \lambda \]

In the Poisson distribution, the variance is equal to the mean (\(\lambda\)). This characteristic is a consequence of the Poisson distribution being specifically suited for situations where events occur independently in time or space, and the average rate of occurrence remains constant.

### Relationship Summary:

- **Binomial Distribution:**
  - \(\sigma^2\) is influenced by both \(n\) and \(p\).
  - \(\sigma^2\) can be larger or smaller than \(\mu\), depending on the values of \(n\) and \(p\).

- **Poisson Distribution:**
  - \(\sigma^2\) is equal to \(\mu\) (\(\sigma^2 = \lambda\)).
  - The variance is fixed and solely determined by the average rate of events (\(\lambda\)).

In summary, while both distributions exhibit a relationship between mean and variance, the specific formulas and characteristics differ. The Binomial distribution allows for more flexibility in adjusting the variance through manipulation of \(n\) and \(p\), while the Poisson distribution inherently links the variance to the mean.'''

'''Q10'''
'''In a normal distribution, the least frequent data points appear in the tails of the distribution, farthest away from the mean. The normal distribution, also known as the Gaussian distribution or bell curve, is symmetric, and the majority of the data is concentrated around the mean.

- **Mean:** The mean is the central point of a normal distribution. It represents the average or expected value of the dataset.

- **Tails:** The tails of the distribution are the regions on both sides, away from the mean. These tails extend infinitely in both directions.

- **Least Frequent Data:** The data points in the tails are least frequent because, as you move away from the mean, the probability density decreases. In a standard normal distribution (with a mean of 0 and standard deviation of 1), the tails are often referred to as the "tails beyond 1 standard deviation," "tails beyond 2 standard deviations," and so on.

In a normal distribution, the probability density is highest around the mean, and it gradually decreases as you move away from the mean in either direction. Data points in the tails are less likely to occur, and extreme values are even less likely.

This characteristic of the normal distribution is a key aspect of statistical inference and hypothesis testing, where critical regions in the tails are often used to make decisions about hypotheses or to establish confidence intervals.'''