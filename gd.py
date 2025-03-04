import numpy as np
import matplotlib.pyplot as plt

# For generating the noisy data
def generate_data():
    np.random.seed(42) 
    m1val,m2val = 5,7  # values set
    x=np.linspace(0,10,100) 
    noise=np.random.normal(0, 1, x.shape)  # normal distribution
    y=m1val*x + m2val + noise  # noisy linear equation (given)
    return x,y,m1val,m2val

x, y, m1val, m2val = generate_data()


#  Scatter plot of noisy data
plt.scatter(x, y, color='blue', label="Noisy Data")
plt.plot(x, m1val * x + m2val, color='red', label="True Line (m1=5, m2=7)")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Generated Data with Noise")
plt.legend()
plt.show()


# Mean Squared Error function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Brute-force to find best m1
def bruteforce(x, y, m2_fixed):
    m1_values = np.linspace(0, 9, 50)  # Trying for 50 values
    errors = [mse(y,m1*x + m2_fixed) for m1 in m1_values]
    
    best_m1 = m1_values[np.argmin(errors)]  # Pick the best m1 among them
    print(f"Brute Force Best m1: {best_m1:.3f}")

    
    plt.plot(m1_values, errors, color='blue', label="MSE vs m1")
    plt.axvline(best_m1, color='red', linestyle='dashed', label=f"Best m1: {best_m1:.3f}")
    plt.xlabel("Slope (m1)")
    plt.ylabel("MSE")
    plt.title("Brute Force: MSE vs m1")
    plt.legend()
    plt.show()

    return best_m1

bf_bestm1 =bruteforce(x,y,m2val)


# Gradient Descent for finding best m1
def gradientdescent(x, y, m2_fixed, lr=0.01, epochs=20):
    m1 = np.random.rand()  # work with a random slope
    n = len(y)
    mse_values = []  # Store MSE 
    m1_values = []  # Store m1 

    for i in range(epochs):
        gradient= (-2/n) * np.sum(x*(y-(m1*x+m2_fixed)))    # Compute gradient
        m1 -= lr*gradient     # m1 is updated using learning rate

        mse_values.append(mse(y, m1 *x+m2_fixed))  # Store current MSE value
        m1_values.append(m1)  
        
        # log every few steps in terminal
        if i % 5 == 0:
            print(f"Epoch {i}: m1={m1:.3f}, MSE={mse_values[-1]:.3f}")

    print(f"Gradient Descent Best m1: {m1:.3f}")

    #Gradient Descent Convergence
    plt.plot(range(epochs), m1_values, marker='o', linestyle='-', color='blue', label="m1 Updates")
    plt.axhline(m1val, color='red', linestyle='dashed', label=f"True m1 = {m1val}")
    plt.axhline(m1, color='green', linestyle='dashed', label=f"Best GD m1 = {m1:.3f}")  # NEW LINE
    plt.xlabel("Epochs")
    plt.ylabel("m1 Value")
    plt.title("Gradient Descent: Convergence of m1")
    plt.legend()
    plt.grid()
    plt.show()

    return m1

gd_bestm1 = gradientdescent(x, y, m2val)

#function to obtain MSE values for m1 and m2, respectively
def msegraph(x, y, m1val, m2val):
    m1_range = np.linspace(0,8,50)  
    m2_range = np.linspace(0,10,50)  

    mse_m1 = [mse(y, m1 * x+m2val) for m1 in m1_range] #finding MSE for the m1 values
    mse_m2 = [mse(y, m1val * x+m2) for m2 in m2_range] #finding MSE for the m2 values

    plt.figure(figsize=(12, 5))

    # MSE vs Slope graph
    plt.subplot(1, 2, 1)
    plt.plot(m1_range, mse_m1, color='blue', label="MSE vs Slope (m1)")
    plt.axvline(m1val, color='red', linestyle='dashed', label=f"True m1: {m1val}")
    plt.xlabel("Slope (m1)")
    plt.ylabel("MSE")
    plt.title("Parabolic MSE vs Slope (m1)")
    plt.legend()
    plt.grid()
    
    #MSE vs intercept graph
    plt.subplot(1, 2, 2)
    plt.plot(m2_range, mse_m2, color='green', label="MSE vs Intercept (m2)")
    plt.axvline(m2val, color='red', linestyle='dashed', label=f"True m2: {m2val}")
    plt.xlabel("Intercept (m2)")
    plt.ylabel("MSE")
    plt.title("Parabolic MSE vs Intercept (m2)")
    plt.legend()
    plt.grid()

    plt.show()

msegraph(x, y, m1val, m2val)



# finding the MSE value of best m1 in both approaches
bf_error = mse(y, bf_bestm1 * x + m2val)
gd_error = mse(y, gd_bestm1 * x + m2val)

percents = ((bf_error - gd_error)/ bf_error)*100 #calculating efficiency rate based on MSE

print("\n  Comparison of Methods:")
print(f"Brute Force Best m1: {bf_bestm1:.3f}, MSE value: {bf_error:.3f}")
print(f"Gradient Descent Best m1: {gd_bestm1:.3f}, MSE value: {gd_error:.3f}")
print(f"\n Gradient Descent is{percents:.2f}% efficient compared to brute force")
