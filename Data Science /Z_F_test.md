## Z-Test and F-Test in Data Science

### Z-Test

**Purpose**: Used to determine if there is a significant difference between sample means or between a sample mean and a population mean. Commonly applied when the sample size is large (n > 30) and the population variance is known.

**Types**:
1. **One-Sample Z-Test**: Tests if the sample mean differs from a known population mean.
2. **Two-Sample Z-Test**: Compares the means of two independent samples.
3. **Z-Test for Proportions**: Compares sample proportions to a known proportion or between two sample proportions.

**Assumptions**:
- Random sampling
- Known population variance
- Data is normally distributed or sample size is large

**Formula**:
\[ Z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \]
where \(\bar{X}\) is the sample mean, \(\mu\) is the population mean, \(\sigma\) is the population standard deviation, and \(n\) is the sample size.

### F-Test

**Purpose**: Used to compare variances between two or more groups. It is frequently used in ANOVA to test if there are significant differences between group means based on their variances.

**Types**:
1. **Two-Sample F-Test**: Compares variances of two independent samples.
2. **ANOVA (Analysis of Variance)**: Generalizes the F-test to compare means across multiple groups.

**Assumptions**:
- Random sampling
- Independent samples
- Data in each group is normally distributed
- Equal variances among groups (homogeneity of variance)

**Formula**:
For comparing two variances:
\[ F = \frac{s_1^2}{s_2^2} \]
where \(s_1^2\) and \(s_2^2\) are the sample variances of the two groups.

**Key Differences**:
- **Z-Test**: Compares means (one or two samples), assumes known population variance.
- **F-Test**: Compares variances (between two or more groups), often used in ANOVA.
