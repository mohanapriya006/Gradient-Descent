**Gradient-Descent**

**Data Generation**
 -> We first create a synthetic dataset using a linear function:  y=m1⋅x + m2 + noise
Here, m1 is set to 5, m2 is set to 7, and random noise from a normal distribution is added to simulate imperfections. The values of m1 and m2 can be changed.

**Brute Force Search**
-> Brute force checks multiple m1 values within a range and calculates the MSE for each one. The best m1 is chosen based on the lowest MSE. A graph of MSE vs. m1 is plotted to visualize how error changes with different slopes.

**Gradient Descent**
-> Gradient Descent iteratively updates m1 using its derivative (gradient) to minimize MSE. Starting with a random m1, we adjust it using a learning rate until convergence. A plot of m1 values over epochs shows how the algorithm approaches the optimal value step by step.

**MSE Graphs for Slope and Intercept**
-> we generate two graphs:
MSE vs. Slope (m1): Shows a parabolic relationship where the optimal m1 has the lowest MSE.
MSE vs. Intercept (m2): Helps analyze how MSE behaves with different intercept values.

**Performance Comparison**
-> Finally, we compare both methods based on MSE values:
Brute Force: Finds the best m1 but requires checking multiple values manually.
Gradient Descent: Learns the best m1 efficiently using optimization techniques.
We calculate how much more efficient Gradient Descent is compared to Brute Force based on MSE reduction.
