Iris Recognition System Optimization using Robust Radius Estimation

This Project focuses on enhancing the accuracy and robustness of iris recognition systems based on the Daugman Rubber Sheet Model, which normalizes the iris texture into a rectangular strip. A major challenge in this process is the unstable estimation of the outer iris radius, often caused by non-circular iris boundaries and image noise. Conventional approaches, which calculate the mean radius across limited angular samples, are highly sensitive to outliers. This leads to inconsistencies in the estimated radius of the same iris across different captures, ultimately degrading recognition performance.

To overcome these limitations, the research systematically optimizes the radius estimation process using two strategies: robust statistical estimators and angular sampling density adjustments. Four estimators—Mean, Trimmed Mean, Huber, and Midmean—were evaluated across 6-angle and 8-angle sampling setups using the MMU iris dataset. The standard deviation of intra-class comparisons was used to evaluate radius stability, while entropy and contrast metrics assessed the quality of the normalized iris image.

The findings show that the optimal estimator depends on the sampling density and noise characteristics of the dataset. For sparse 6-angle configurations, the Trimmed Mean offers superior stability compared to the traditional Mean. When increasing the sampling density to 8 angles, the Midmean estimator achieves the best overall performance, balancing accuracy and computational efficiency. In scenarios with significant outliers, the Huber estimator proves most robust under both configurations.

Increasing the angular sampling from 6 to 8 angles consistently improved stability and enhanced image quality, with entropy and contrast increasing by approximately 20%. The study concludes that the 8-angle sampling with the Midmean estimator offers the best general-purpose configuration, effectively optimizing the iris normalization process and improving subsequent feature extraction and matching accuracy.


Optimized iris recognition using robust estimators (Mean, Trimmed Mean, Huber, Midmean) and varied angular sampling (6 &amp; 8 angles). Midmean at 8 angles gave best stability, 20% quality gain. Huber handled outliers best. Improves iris normalization and recognition accuracy.
