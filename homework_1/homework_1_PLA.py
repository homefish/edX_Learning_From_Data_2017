import numpy as np
import matplotlib.pyplot as plt

def rnd(n): 
    return np.random.uniform(-1, 1, size = n)

RUNS = 1000
iterations_total = 0
ratio_mismatch_total = 0

#--------------------------------------------------------------

for run in range(RUNS):

    # choose two random points A, B in [-1,1] x [-1,1]
    A = rnd(2)
    B = rnd(2)

    # the line can be described by y = m*x + b where m is the slope
    m = (B[1] - A[1]) / (B[0] - A[0])
    b = B[1] - m * B[0]  
    w_f = np.array([b, m, -1])

    #-----------------------

    # Create N data points (x, y) from the target function
    N = 100
    X = np.transpose(np.array([np.ones(N), rnd(N), rnd(N)]))           # input
    y_f = np.sign(np.dot(X, w_f))                                      # output
    
    #----------------------------------------

    # choose hypothesis h

    w_h = np.zeros(3)                       # initialize weight vector for hypothesis h
    t = 0                                   # count number of iterations in PLA
    
    while True:
        # Start PLA
        y_h = np.sign(np.dot(X, w_h))       # classification by hypothesis
        comp = (y_h != y_f)                 # compare classification with actual data from target function
        wrong = np.where(comp)[0]           # indices of points with wrong classification by hypothesis h

        if wrong.size == 0:
            break
        
        rnd_choice = np.random.choice(wrong)        # pick a random misclassified point

        # update weight vector (new hypothesis):
        w_h = w_h +  y_f[rnd_choice] * np.transpose(X[rnd_choice])
        t += 1

    iterations_total += t
    
    # ------------------------------------------

    # Calculate error
    # Create data "outside" of training data

    N_outside = 1000
    test_x0 = np.random.uniform(-1,1,N_outside)
    test_x1 = np.random.uniform(-1,1,N_outside)

    X = np.array([np.ones(N_outside), test_x0, test_x1]).T

    y_target = np.sign(X.dot(w_f))
    y_hypothesis = np.sign(X.dot(w_h))
    
    ratio_mismatch = ((y_target != y_hypothesis).sum()) / N_outside
    ratio_mismatch_total += ratio_mismatch

#------------------------------------------------
    
print("Size of training data: N = ", N, "points")
    
iterations_avg = iterations_total / RUNS
print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)

ratio_mismatch_avg = ratio_mismatch_total / RUNS
print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
print("P(f(x)!=h(x)) = ", ratio_mismatch_avg)
