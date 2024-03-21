import numpy as np
from sklearn.cluster import KMeans

def get_arg_max_real(x):
    index = 0
    M = -np.inf
    for i in range(len(x)):
        if np.real(x[i])> M:
            M = np.real(x[i])
            index=i
    return index

def get_arg_max_modulus(x):
    index = 0
    M = -np.inf
    for i in range(len(x)):
        if x[i]>= M:
            M = x[i]
            index=i
    return index

def get_leading_eigenvector(B):
    [vals,vecs] = np.linalg.eig(B)
    # If no positive eigenvalues, to within round off error
    if all(v < 1e-4 for v in list(vals)): 
        ones = np.ones((1,len(B))) / np.sqrt(len(B)) # Normalized vector of all 1's
        max_vec = ones[0,:]
    else:
        arg_max = get_arg_max_real(vals)
        max_vec = np.real(vecs[:,arg_max])
    return max_vec

def get_principal_eigenvector(A):
    [vals,vecs] = np.linalg.eig(A)
    arg_max = get_arg_max_modulus(vals)
    max_vec = vecs[:,arg_max]
    return max_vec

def get_shores_from_eigenvector(nodes,x):
    shore1 = set()
    shore2 = set()
    for node in nodes:
        if x[node] < 0: shore1.add(node)
        else: shore2.add(node)
    return [shore1, shore2]

def get_two_fiedler_eigenvectors(L):
    [evals, evecs] = np.linalg.eig(L)
    sorted_indices = np.argsort(evals)
    two_smallest_nonzero = []
    for i in sorted_indices:
        if evals[i] ==0: continue
        two_smallest_nonzero.append(i)
        if len(two_smallest_nonzero) == 2: break
    print(two_smallest_nonzero)
    fiedler1 = evecs[:,two_smallest_nonzero[0]]
    fiedler2 = evecs[:,two_smallest_nonzero[1]]
    return fiedler1, fiedler2

def get_largest_vectors(A):
    [evals, evecs] = np.linalg.eig(A)
    evals = [0 if np.abs(v) < 1e-5 else np.round(np.abs(v),1) for v in evals]
    sorted_indices = np.argsort(evals)
    two_largest_nonzero = []
    for i in np.flip(sorted_indices):
        two_largest_nonzero.append(i)
        if len(two_largest_nonzero) == 2: break
    big1 = np.real(evecs[:,two_largest_nonzero[0]])
    big2 = np.real(evecs[:,two_largest_nonzero[1]])
    return big1, big2

def form_encoding(vector1, vector2):
    z = np.zeros((len(vector1),2))
    for i in range(len(vector1)):
        z[i,0] = vector1[i]
        z[i,1] = vector2[i]
    return z

def get_clusters(embedding, num_clusters = 4):
    kmeans = KMeans(
        init="random",
        n_clusters=num_clusters,
        n_init=10,
        random_state=1234
        )
    kmeans.fit(embedding)
    return kmeans

def get_colors_from_clusters(embedding, num_clusters = 4):
    kmeans = get_clusters(embedding, num_clusters=num_clusters)
    labels = kmeans.labels_
    color_template = ['y', 'c', 'm', 'k', 'red', 'green', 'lightblue']
    color = [color_template[x] for x in list(labels) ]
    return color
