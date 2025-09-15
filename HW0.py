import numpy as np
from numpy import linalg as nla
from scipy import linalg as sla

a = np.array([[ 1,  2,  3],
              [ 4,  5,  6],
              [ 7,  8,  9],
              [10, 11, 12],
              [13, 14, 15]])

b = np.array([[1,3,4],
              [7,7,7],
              [8,8,8],
              [9,9,9],
              [3,3,3]])

c = np.array([[1,2,1],
              [1,3,1],
              [1,4,1],
              [1,5,1],
              [1,6,1]])

d = np.array([[2,3,2],
              [2,4,2],
              [2,5,2],
              [2,6,2],
              [2,7,2]])

e = np.block([[a, b],
              [c, d]])

numOfDim = a.ndim
sizeOfArray = a.size
shapeOfArray = a.shape
numElementsof2dim = a.shape[1]

lastElement = a[-1]
firstRowsecondColEle = a[0, 1]
secondRow = a[1]
first5row = a[0:5]
last3row = a[-3:]
firstthirdrowandfirstthirdcol = a[0:3, 0:3]
row245andcol13 = a[np.ix_([1,3,4], [0,2])]
otherrowstartingwith3 = a[2:5:2, :]
otherrowstartingwith1 = a[::2, :]
reverseorder = a[::-1, :]
# Fixed: np.r_ should be used to concatenate ranges, not with len()
copyof1toend = a[np.r_[:len(a)]]

transpose = a.T
conjugateT = a.conj().T

aMultb = a @ b.T
elementMult = a * b
elementDiv = a / b
elementExpo = a**3

largerthanhalf = (a > 0.5)
indiceslargerthanhalf = np.nonzero(a > 0.5)

v = np.array([1, 2, 3])
# Fixed: v > 0.5 returns boolean array, need to handle properly
selected_Col_v = a[:, np.nonzero(v > 0.5)[0]]
selected_Col_cv = a[:, v > 0.5]

a_thr = a.copy()
a_thr[a_thr < 0.5] = 0
delete_smallerhalf2 = a * (a > 0.5)

a_copy = a.copy()
a_copy[:] = 3

x = np.array([1,2,3])
y = x.copy()

y2 = a[1, :].copy()
y3 = x.flatten()

increasev = np.arange(1., 11.)
increasev2 = np.arange(10.)
colv = np.arange(1., 11.)[:, np.newaxis]
array2Dofzeros = np.zeros((3,4))
array3Dofzeros = np.zeros((3,4,5))
array2Dofones = np.ones((3,4))
array3Dofones = np.ones((3,4,5))
identity = np.eye(3)

returnDiagV = np.diag(a)
returnSqDiag = np.diag(v, 0)

from numpy.random import default_rng
rng = default_rng(42)
random3x4array = rng.random((3,4))
foursamplesbetween1n3 = np.linspace(1, 3, 4)
xy2D = np.mgrid[0:9., 0:6.]

copiesofa = np.tile(a, (3, 5))
concateColAB = np.hstack((a, b))

maxa = a.max()
maxElementofCol = a.max(0)
maxElementofRow = a.max(1)
compareAB = np.maximum(a, b)

L2norm = np.sqrt(v @ v)

C = a.T @ a

# Fixed: C is singular, using pseudo-inverse instead
try:
    inverseofSquare = nla.inv(C)
except nla.LinAlgError:
    inverseofSquare = "Singular matrix - cannot compute inverse"

pinverse = nla.pinv(a)
rankofarray = nla.matrix_rank(a)

a1 = np.array([[1,1,1],
               [1,1,1],
               [1,1,1]], dtype=float)
b1 = np.array([[2,3,2],
               [3,4,3],
               [1,2,1]], dtype=float)

solution_ls, *_ = nla.lstsq(a1, b1, rcond=None)

S = C
try:
    cholesky = nla.cholesky(S)
except nla.LinAlgError:
    cholesky = "Matrix not positive definite"

D, V = nla.eig(C)

Q, R = nla.qr(a, mode='reduced')
P, L, U = sla.lu(a)

fouriertransform = np.fft.fft(a)
inversefourier = np.fft.ifft(a)

vectoruniq = np.unique(a)
singleton = np.array([[42]]).squeeze()

# Print statements
print("a @ b.T:\n", aMultb)
print("elementMult:\n", elementMult)
print("elementDiv:\n", elementDiv)
print("elementExpo:\n", elementExpo)
print("indiceslargerthanhalf:\n", indiceslargerthanhalf)
print("selected_Col_v:\n", selected_Col_v)
print("selected_Col_cv:\n", selected_Col_cv)
print("a_thr (<0.5 set to 0):\n", a_thr)
print("delete_smallerhalf2:\n", delete_smallerhalf2)
print("a_copy[:] = 3:\n", a_copy)
print("x:\n", x)
print("y (copy of x):\n", y)
print("y2 (row copy):\n", y2)
print("y3 (flattened x):\n", y3)
print("increasev:\n", increasev)
print("increasev2:\n", increasev2)
print("colv:\n", colv)
print("array2Dofzeros:\n", array2Dofzeros)
print("array3Dofzeros shape:", array3Dofzeros.shape)
print("array2Dofones:\n", array2Dofones)
print("array3Dofones shape:", array3Dofones.shape)
print("identity:\n", identity)
print("returnDiagV:\n", returnDiagV)
print("returnSqDiag:\n", returnSqDiag)
print("random3x4array:\n", random3x4array)
print("foursamplesbetween1n3:\n", foursamplesbetween1n3)
print("xy2D shape:", xy2D.shape)
print("copiesofa shape:", copiesofa.shape)
print("concateColAB:\n", concateColAB)
print("maxa:", maxa)
print("maxElementofCol:", maxElementofCol)
print("maxElementofRow:", maxElementofRow)
print("compareAB:\n", compareAB)
print("L2norm:", L2norm)
print("C (a.T@a):\n", C)
print("inverseofSquare:\n", inverseofSquare)
print("pinverse:\n", pinverse)
print("rankofarray:", rankofarray)
print("solution least squares:\n", solution_ls)
print("cholesky(S):\n", cholesky)
print("eigvals(C):", D)
print("eigvecs(C):\n", V)
print("Q (QR):\n", Q)
print("R (QR):\n", R)
print("P (LU):\n", P)
print("L (LU):\n", L)
print("U (LU):\n", U)
print("fouriertransform shape:", fouriertransform.shape)
print("inversefourier shape:", inversefourier.shape)
print("vectoruniq:\n", vectoruniq)
print("singleton squeeze:", singleton)

import matplotlib.pyplot as plt
plt.plot ([1,2,3,4],[1,2,7,14])
plt.axis([0,6,0,20])
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,10,100)
y=np.sin(x)
plt.figure(figsize=(8,5))
plt.plot(x,y,'b-',linewidth=2,label='sin(x)')
plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.title('Sin plot')
plt.grid(True,alpha=0.3)
plt.legend()
plt.show()