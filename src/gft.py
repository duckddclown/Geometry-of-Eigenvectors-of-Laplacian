# This file looks into different graphs and how they differ.
import numpy as np
import pygsp as gsp
import matplotlib.pyplot as plt
import scipy.optimize as opt
from mpl_toolkits import mplot3d
from DimensionReduction import PCA
from sklearn.manifold import MDS
from scipy.stats import wasserstein_distance


# One dimensional path. Undirected case.
class path:
    def __init__(self, n) -> None:
        self.n = n
        self._createAdj()
        self.graph = gsp.graphs.Graph(self.adj)
        self._setCoordinate()
        self.graph.plot_signal(self.graph.U[:,2])
        plt.show()

    def _createAdj(self):
        n = self.n
        adj = np.zeros((n, n))
        for i in range(n):
            if i-1 >= 0:
                adj[i,i-1] = 1
            if i+1 < n:
                adj[i,i+1] = 1

        self.adj = adj

    def _setCoordinate(self):
        n = self.n
        self.coordinate = np.zeros((n,2))
        self.coordinate[:,0] = range(n)
        self.graph.set_coordinates(self.coordinate)

class TwoDGrid:
    def __init__(self, nx, ny) -> None:
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self._createAdj()
        self._createInc()
        self.graph = gsp.graphs.Graph(self.adj)
        self._setCoordinate()
        self._computeGridEigenvalue()
        # self._computeGridEigenvector()
        # self.graph.plot_signal(self.graph.U[:,3])
        self._computeEffResistance()
        self._computeProdResistance()
        # plt.show()
        
    def _createAdj(self):
        nx = self.nx
        ny = self.ny
        n = self.n
        self.num_edge = 0
        adj = np.zeros((n, n))
        for i in range(ny):
            for j in range(nx):
                currIndex = i * nx + j
                if j-1 >= 0:
                    index = currIndex-1
                    adj[currIndex, index] = 1
                    self.num_edge += 1
                if j+1 < nx:
                    index = currIndex+1
                    adj[currIndex, index] = 1
                    self.num_edge += 1
                if i-1 >= 0:
                    index = currIndex-nx
                    adj[currIndex, index] = 1
                    self.num_edge += 1
                if i+1 < ny:
                    index = currIndex+nx
                    adj[currIndex, index] = 1
                    self.num_edge += 1
        
        self.adj = adj

    def _createInc(self):
        n = self.n
        num_edge = self.num_edge
        adj = self.adj
        inc = np.zeros((n,num_edge))
        edge_index = 0
        for i in range(n):
            j = i
            while j < n:
                if adj[i,j] != 0:
                    inc[i,edge_index] = 1
                    inc[j,edge_index] = -1
                    edge_index += 1
                j += 1

        self.inc = inc
        self.weight = np.identity(self.num_edge)

    def _setCoordinate(self):
        n = self.n
        nx = self.nx
        ny = self.ny
        coordinate = np.zeros((ny, nx, 2))
        for i in range(nx):
            coordinate[:,i,0] = i

        for i in range(ny):
            coordinate[i,:,1] = i
        
        coordinate = coordinate.reshape((n,2))
        self.coordinate = coordinate
        self.graph.set_coordinates(self.coordinate)

    def _xdimEigenvalue(self, k):
        nx = self.nx
        return 4 * np.sin(np.pi*k/2/nx) ** 2
    
    def _ydimEigenvalue(self, k):
        ny = self.ny
        return 4 * np.sin(np.pi*k/2/ny) ** 2

    def _computeGridEigenvalue(self):
        gridEigenvalue = np.zeros((self.ny,self.nx))
        for x in range(self.nx):
            for y in range(self.ny):
                xEigenvalue = self._xdimEigenvalue(x)
                yEigenvalue = self._ydimEigenvalue(y)
                gridEigenvalue[y,x] = xEigenvalue + yEigenvalue

        self.grid_eigenvalue = gridEigenvalue

    def _computeGridEigenvector(self):
        eigenfunctions = self.graph.U.copy()
        gridEigenfunction = np.zeros((self.ny, self.nx, self.n))
        for x in range(self.nx):
            for y in range(self.ny):
                eigenvalue = self.grid_eigenvalue[y,x]
                index = np.where(np.isclose(self.graph.e, eigenvalue))
                gridEigenfunction[y,x,:] = eigenfunctions[:,index].reshape(self.n)
        
        self.grid_eigenfunction = gridEigenfunction

    def _computeEffResistance(self):
        laplacian = self.graph.L.toarray()
        sudoInverseLaplacian = np.linalg.pinv(laplacian)
        eigenfunctions = self.graph.U.copy()
        # for i in range(self.n):
        #     norm_eigenfunctions = np.e ** eigenfunctions[:,i]
        #     eigenfunctions[:,i] = norm_eigenfunctions / norm_eigenfunctions.sum()
        eigenfunctions = eigenfunctions ** 2
        self.diff_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                f1 = eigenfunctions[:,i]
                f2 = eigenfunctions[:,j]
                dist = EffResistance(f1=f1, L=sudoInverseLaplacian, f2=f2)
                self.diff_matrix[i,j] = dist ** 0.5

    def _computeProdResistance(self):
        laplacian = self.graph.L.toarray()
        sudoInverseLaplacian = np.linalg.pinv(laplacian)
        eigenfunctions = self.graph.U.copy()
        # for i in range(self.n):
        #     norm_eigenfunctions = np.e ** eigenfunctions[:,i]
        #     eigenfunctions[:,i] = norm_eigenfunctions / norm_eigenfunctions.sum()
        eigenfunctions = eigenfunctions ** 2
        self.prod_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                f1 = eigenfunctions[:,i]
                f2 = eigenfunctions[:,j]
                dist = ProdResistance(f1=f1, L=sudoInverseLaplacian, f2=f2)
                self.prod_matrix[i,j] = dist ** 0.5

    def computeDiffBeckmann(self, p):
        eigenfunctions = self.graph.U.copy()
        for i in range(self.n):
            norm_eigenfunctions = np.e ** eigenfunctions[:,i]
            eigenfunctions[:,i] = norm_eigenfunctions / norm_eigenfunctions.sum()
        diff_Beckmann_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                f1 = eigenfunctions[:,i]
                f2 = eigenfunctions[:,j]
                dist = BeckmannDiff(p = p, B = self.inc, f1 = f1, f2 = f2, W = self.weight, num_edge = self.num_edge)
                diff_Beckmann_matrix[i,j] = dist ** 2
        
        return diff_Beckmann_matrix
    
    def computeProdBeckmann(self, p):
        eigenfunctions = self.graph.U.copy()
        for i in range(self.n):
            norm_eigenfunctions = np.e ** eigenfunctions[:,i]
            eigenfunctions[:,i] = norm_eigenfunctions / norm_eigenfunctions.sum()
        prod_Beckmann_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                f1 = eigenfunctions[:,i]
                f2 = eigenfunctions[:,j]
                dist = BeckmannProd(p = p, B = self.inc, f1 = f1, f2 = f2, W = self.weight, num_edge = self.num_edge)
                prod_Beckmann_matrix[i,j] = dist ** 2
        
        return prod_Beckmann_matrix

    def orderByDiffBeckmann(self):
        order = [0]
        curr_index = 0
        for i in range(self.n-1):
            diff_vector = self.diff_matrix[:,curr_index].copy()
            diff_index = []
            for j, diff in enumerate(diff_vector):
                diff_index.append((diff,j))
            diff_index_in_order = sorted(diff_index, key=lambda x:x[0])
            for j in range(self.n):
                if diff_index_in_order[j][1] in order:
                    continue
                order.append(diff_index_in_order[j][1])
                curr_index = diff_index_in_order[j][1]
                break

        print(order)

    def orderByProdBeckmann(self):
        order = [0]
        curr_index = 0
        for i in range(self.n-1):
            diff_vector = self.prod_matrix[:,curr_index].copy()
            diff_index = []
            for j, diff in enumerate(diff_vector):
                diff_index.append((diff,j))
            diff_index_in_order = sorted(diff_index, key=lambda x:x[0])
            for j in range(self.n):
                if diff_index_in_order[j][1] in order:
                    continue
                order.append(diff_index_in_order[j][1])
                curr_index = diff_index_in_order[j][1]
                break

        print(order)

def wpDistance(J, p, W):
    return (np.matmul(W.T, J ** p)) ** (1/p)

def wpDistanceDer(J, p, W):
    return W * J ** (p-1) * np.matmul(W.T, J ** p) ** (1/p-1)

def wpDistanceHess(J, p, W):
    n = J.shape[0]
    hess = np.zeros((n,n))
    if p != 1:
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = (1-p) * (np.matmul(W.T, J ** p)) ** (1/p-2) * W[i] * W[j] * (J[i] * J[j]) ** (p-1)
                    hess[i,j] = val
                else:
                    val = W[i] * (p-1) * J[i] ** (p-2) * np.matmul(W.T, J ** p) ** (1/p-1) + (1-p) * np.matmul(W.T, J ** p) ** (1/p-2) * (W[i] * J[i] ** (p-1)) ** 2
                    hess[i,j] = val
    return hess
                
def BeckmannDiff(p, B, f1, f2, W, num_edge):
    diff = f1 - f2
    linear_constraint = opt.LinearConstraint(B, diff, diff)
    weight_vec = np.diagonal(W)
    res = opt.minimize(wpDistance, x0 = np.ones(num_edge), args = (p, weight_vec), method = "trust-constr", jac = wpDistanceDer, hess = wpDistanceHess, constraints = [linear_constraint])
    return res.fun

def BeckmannProd(p, B, f1, f2, W, num_edge):
    diff = f1 * f2
    linear_constraint = opt.LinearConstraint(B, diff, diff)
    weight_vec = np.diagonal(W)
    res = opt.minimize(wpDistance, x0 = np.ones(num_edge), args = (p, weight_vec), method = "trust-constr", jac = wpDistanceDer, hess = wpDistanceHess, constraints = [linear_constraint])
    return res.fun

def EffResistance(f1, L, f2):
    diff = f1 - f2
    return np.matmul(diff.T, np.matmul(L, diff))

def ProdResistance(f1, L , f2):
    diff = f1 * f2
    return np.matmul(diff.T, np.matmul(L, diff))

st = TwoDGrid(7,3)
# f1 = st.grid_eigenfunction[0,1]
# distance_matrix = np.zeros((3,7))
# laplacian = st.graph.L.toarray()
# sudoInverseLaplacian = np.linalg.pinv(laplacian)
# for x in range(st.nx):
#     for y in range(st.ny):
#         f2 = st.grid_eigenfunction[y,x]
#         distance_matrix[y,x] = wasserstein_distance(f1,f2)
# print(distance_matrix)
# print(st.graph.e)
# diff_beck1 = st.computeDiffBeckmann(1)
# print("diff done")
# prod_beck2 = st.computeProdBeckmann(2)

# diff_eigenvalue, diff_eigenvector = np.linalg.eig(st.diff_matrix)
# prod_eigenvalue, prod_eigenvector = np.linalg.eig(st.prod_matrix)

# P, Y = PCA(st.diff_matrix)
# data_x = Y[0,:]
# data_y = Y[1,:]
# data_z = Y[2,:]

n_components = 2
mds = MDS(n_components=n_components)
X_reduced = mds.fit_transform(st.prod_matrix)

fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(data_x, data_y, data_z)

plt.scatter(X_reduced[:,0], X_reduced[:,1])
plt.show()

# print(data_x)
# plt.scatter(data_x, data_y)
# ax.scatter2D(data_x, data_y, cmap='Greens')
# plt.show()

# print(st.prod_matrix)
# print(st.graph.e)
# print(st.diff_matrix[:,2])
# st.orderByProdBeckmann()
# L = st.graph.L.toarray()
# f1 = np.array([1,2,3,4])
# f2 = np.array([4,3,2,1])
# st = TwoDGrid(7,3)
# print(np.linalg.norm(st.graph.U[:,2]))

# print(Beckmann_2(f1, L, f2))


