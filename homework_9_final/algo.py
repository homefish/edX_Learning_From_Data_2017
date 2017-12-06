import numpy as np

#print("class kmeans_RBF loaded")

class kmeans_RBF:

    def __init__(self, num_clusters = None, gamma = None):
        '''
        - Takes number of clusters K
        - returns weight vector learned by regular RBF model,
        i.e. using Lloyd's algorithm + pseudo inverse
        '''
        self.K = num_clusters
        self.cluster_centers = None
        self.cluster_index_of_x = None
        self.w = None
        self.gamma = gamma

    def fit(self, X, y):
        N = X.shape[0]
        
        '''
        - Takes points X (numpy array)
        - Calculates final cluster centers
        - Calculates cluster index of each point x
        - Returns None
        '''
        while True:
            empty_cluster_detected = False
            in_sample_error_nonzero = False

            # We repeat the experiment until we get a case where all
            # clusters are non-empty

            # initialize centers by picking random points
            mu_list = np.random.uniform(-1,1,(self.K,2))
            
            #print("\ninitial centers: mu_list = ")
            #print(mu_list)
            
            #------------

            # cluster_of_x stores for each point x its cluster
            cluster_of_x = [-1 for _ in range(N)]
            old_cluster_of_x = [-1 for _ in range(N)]


            MAX_ITERATIONS = 10**6

            for i in range(MAX_ITERATIONS):

                # initialize clusters
                S = [[] for _ in range(self.K)]

                # assign each point to a cluster
                for point_index, x in enumerate(X):
                    # determine for each point its nearest cluster
                    min_distance = 2**64
                    min_cluster = None
                    for index, mu in enumerate(mu_list):
                        distance = np.linalg.norm(x - mu)
                        if distance < min_distance:
                            min_distance = distance
                            min_cluster = index
                    S[min_cluster].append(x)
                    cluster_of_x[point_index] = min_cluster

                # check if there is an empty cluster
                for cluster in S:
                    if not cluster:
                        #print("\nEmpty cluster detected, discarding run")
                        empty_cluster_detected = True

                if empty_cluster_detected:
                    break

                #----------------------------------

                # stop if nothing changes, i.e. points are in the same clusters as in previous iteration
                if cluster_of_x == old_cluster_of_x:
                    #print("Cluster have not changed, stopping for loop...")
                    break

                #------------------------------------------------------------------

                # make a copy
                old_cluster_of_x = [cluster_index for cluster_index in cluster_of_x]

                # calculate the new centers mu
                for index, cluster in enumerate(S):
                    mu = sum(cluster) / len(cluster)   # compute center of gravity
                    mu_list[index] = mu

            #print("\nfinal centers: mu_list = ")
            #print(mu_list)
            


            #if discard_run == False:
            #    break
            if (empty_cluster_detected == True):
                #print("\nEmpty cluster detected, discarding run")
                continue
            
                
            # setting attributes
            self.cluster_centers = mu_list
            self.cluster_index_of_x = cluster_of_x


            # calculate w via linear regression
            def matrix_phi_entry(x, mu, gamma):
                return np.exp(-gamma * np.linalg.norm(x - mu)**2)

            # initialize phi
            phi = np.zeros((N, self.K))

            # fill matrix phi
            for i in range(N):
                for k in range(self.K):
                    phi[i,k] = matrix_phi_entry(X[i], mu_list[k], self.gamma)

            phi = np.c_[np.ones(N), phi]

            #print("\nphi for training points = ")
            #print(phi)

            self.w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), y)   # coefficients w
            #print("\nw = ", self.w)
            
            
            #if (in_sample_error_nonzero == False):
            #    break
            break
                
        #-----------------------------------------------------------------------


    def predict(self, X_test):
        '''
        - Takes points X
        - Returns predicted y
        '''
        
        def matrix_phi_entry(x, mu, gamma):
            return np.exp(-self.gamma * np.linalg.norm(x - mu)**2)

        # initialize phi
        N_test = X_test.shape[0]
        phi = np.zeros((N_test, self.K))
        #print("shape of phi: ", phi.shape)

        # fill matrix phi
        for i in range(N_test):
            for k in range(self.K):
                phi[i,k] = matrix_phi_entry(X_test[i], self.cluster_centers[k], self.gamma)


        phi = np.c_[np.ones(N_test), phi]
        
        #print("\nphi matrix:")
        #print(phi)
        
        y_predicted = np.sign(np.dot(phi, self.w))
        return y_predicted

        
