from scipy import sparse


csr_arr = sparse.random(1, 10, density=0.2, format='csr')
arr = csr_arr.toarray()

