from machine import UART
from pyb import LED
uart = UART(2, baudrate=115200)     # 初始化串口 波特率设置为115200 TX是B12 RX是B13
import sensor, image, time
import math
DEBUG = 1
#--------------------umatrix---------------------------#
import sys

stypes = [bool, int]
ddtype = int
estypes = []
flt_eps = 1.19E-7


class matrix(object):

    def __init__(self, data, cstride=0, rstride=0, dtype=None):
        ''' Builds a matrix representation of 'data'.
            'data' can be a list (columns) of lists (rows)
            [[1,2,3],[4,5,6]] or
            a simple list organized as determined by rstride and cstride:
            [1,2,3,4,5,6] cstride=1, rstride=3.
            Elements will be of highest type included in 'data' or
            'dtype' can be used to force the type.
        '''
        if cstride != 0:
            if cstride == 1:
                self.n = rstride
                self.m = int(len(data) / self.n)
            else:
                self.m = cstride
                self.n = int(len(data) / self.m)
            self.cstride = cstride
            self.rstride = rstride
            self.data = data
        else:
            # else determine shape from list passed in
            self.n = 1
            if type(data) == int:
                self.m = 1
            else:  # it is a list
                self.m = len(data)
                # is data[0] a list
                if (type(data[0]) == list):
                    self.n = len(data[0])
            self.data = [data[i][j]
                         for i in range(self.m) for j in range(self.n)]
            self.cstride = 1
            self.rstride = self.n
        # ensure all elements are of the same type
        if dtype is None:
            self.dtype = stypes[max([stypes.index(type(i)) for i in self.data])]
        else:
            if dtype in stypes:
                self.dtype = dtype
            else:
                raise TypeError('unsupported type', dtype)
        self.data = [self.dtype(i) for i in self.data]

    def __len__(self):
        return self.m

    def __eq__(self, other):
        if self.shape == other.shape:
            res = all([self.data[i] == other.data[i] for i in range(self.size())])
            return res and (self.shape == other.shape)
        else:
            raise ValueError('shapes not equal')

    def __ne__(self, other):
        return not __eq__(other)

    def __iter__(self):
        self.cur = 0
        # determine proper axis
        if self.m == 1:
            self.cnt_lim = self.n
        else:
            self.cnt_lim = self.m
        return self

    def __next__(self):
        '''
        Returns a matrix if m > 1
        else the next numeric element of the vector.
        (Numpy returns vectors if selected via slice)
        '''
        if self.cur >= self.cnt_lim:
            raise StopIteration
        self.cur = self.cur + 1
        if self.m == 1:
            return self.data[self.cur - 1]
        else:
            return self[self.cur - 1]

    def slice_to_offset(self, r0, r1, c0, c1):
        # check values and limit them
        nd = [self.data[i * self.rstride + j * self.cstride]
              for i in range(r0, r1) for j in range(c0, c1)]
        return matrix(nd, cstride=1, rstride=(c1 - c0))

    def slice_indices(self, index, axis=0):
        # handles the unsupported slice.indices() method in uPy.
        # If implemented:
        #     midx = index.indices(self.m)
        # should work.
        if isinstance(index.start, type(None)):
            s0 = 0
        else:
            s0 = min(int(index.start), self.shape[axis])
        if isinstance(index.stop, type(None)):
            p0 = self.shape[axis]
        else:
            p0 = min(int(index.stop), self.shape[axis])
        return (s0, p0)

    def __getitem__(self, index):
        if type(index) == tuple:
            # int and int
            # int and slice
            # slice and int
            # slice and slice
            if isinstance(index[0], int):
                s0 = index[0]
                p0 = s0 + 1
            else:  # row slice
                s0, p0 = self.slice_indices(index[0], 0)
            if isinstance(index[1], int):
                s1 = index[1]
                p1 = s1 + 1
            else:  # column slice
                s1, p1 = self.slice_indices(index[1], 1)
        elif type(index) == list:
            # list of indices etc
            raise NotImplementedError('Fancy indexing')
        else:
            # type is int? This will default to returning a row
            s0 = index
            p0 = s0 + 1
            s1 = 0
            p1 = self.n
        # resultant matrix
        z = self.slice_to_offset(s0, p0, s1, p1)
        # if it's a single entry then return that entry as int, float etc.
        if (p0 == s0 + 1) and (p1 == s1 + 1):
            return z.data[0]
        else:
            return z

    def __setitem__(self, index, val):
        if type(index) != tuple:
            # need to make it a slice without the slice function
            raise NotImplementedError('Need to use the slice [1,:] format.')
        # int and int => single entry gets changed
        # combinations of int and slice => row and columns take on elements from val
        if isinstance(index[0], int):
            s0 = index[0]
            p0 = s0 + 1
        else:  # slice
            s0, p0 = self.slice_indices(index[0], 0)
        if isinstance(index[1], int):
            s1 = index[1]
            p1 = s1 + 1
        else:  # slice
            s1, p1 = self.slice_indices(index[1], 1)
        if type(val) == matrix:
            val = val.data
        elif type(val) not in [list, tuple]:
            val = [val]
        if not all([type(i) in stypes for i in val]):
            raise ValueError('Non numeric entry')
        else:
            # assign list values wrapping as necessary to fill destination
            k = 0
            for i in range(s0, p0):
                for j in range(s1, p1):
                    self.data[i * self.rstride + j * self.cstride] = (self.dtype(val[k]))
                    k = (k + 1) % len(val)

    # there is also __delitem__

    # def __str__(self):
    def __repr__(self):
        # things that use __str__ will fallback to __repr__
        # find max string field size for formatting
        l = 0
        for i in self.data:
            l = max(l, len(repr(i)))
        s = 'mat(['
        r = 0
        for i in range(self.m):
            c = 0
            s = s + '['
            for j in range(self.n):
                s1 = repr(self.data[r + c])
                s = s + s1 + ' ' * (l - len(s1))
                if (j < (self.n - 1)):
                    s = s + ', '
                c = c + self.cstride
            if (i < (self.m - 1)):
                s = s + '],\n     '
            else:
                s = s + ']'
            r = r + self.rstride
        s = s + '])'
        return s

    # Reflected operations are not yet implemented in MicroPython
    # __rmul__ for example will not be invoked

    def __neg__(self):
        ndat =[self.data[i] * (-1) for i in range(len(self.data))]
        return matrix(ndat, cstride=self.cstride, rstride=self.rstride)

    def __do_op__(self, a, b, op):
        if op == '+':
            return (a + b)
        elif op == '-':
            return (a - b)
        elif op == '*':
            return (a * b)
        elif op == '**':
            return (a ** b)
        elif op == '/':
            try:
                return (a / b)
            except ZeroDivisionError:
                raise ZeroDivisionError('division by zero')
        elif op == '//':
            try:
                return (a // b)
            except ZeroDivisionError:
                raise ZeroDivisionError('division by zero')
        else:
            raise NotImplementedError('Unknown operator ', op)

    def __OP__(self, a, op):
        if type(a) in stypes:
            # matrix - scaler elementwise operation
            ndat = [self.__do_op__(self.data[i], a, op) for i in range(len(self.data))]
            return matrix(ndat, cstride=self.cstride, rstride=self.rstride)
        elif (type(a) == list):
            # matrix - list elementwise operation
            # hack - convert list to matrix and resubmit then it gets handled below
            # if self.n = 1 try transpose otherwise broadcast error to match numpy
            if (self.n == 1) and (len(a) == self.m):
                return self.__OP__(matrix([a]).T, op)
            elif len(a) == self.n:
                return self.__OP__(matrix([a]), op)
            else:
                raise ValueError('could not be broadcast')
        elif (type(a) == matrix):
            if (self.m == a.m) and (self.n == a.n):
                # matrix - matrix elementwise operation
                # use matrix indices to handle views
                ndat = [self.__do_op__(self[i, j], a[i, j], op) for i in range(self.m) for j in range(self.n)]
                return matrix(ndat, cstride=1, rstride=self.n)
            # generalize the following two elif for > 2 dimensions?
            elif (self.m == a.m):
                # m==m n!=n => column-wise row operation
                Y = self.copy()
                for i in range(self.n):
                    # this call _OP_ once for each row and __do_op__ for each element
                    for j in range(self.m):
                        Y[j, i] = self.__do_op__(Y[j, i], a[j, 0], op)
                return Y
            elif (self.n == a.n):
                # m!=m n==n => row-wise col operation
                Y = self.copy()
                for i in range(self.m):
                    # this call _OP_ once for each col and __do_op__ for each element
                    for j in range(self.n):
                        Y[i, j] = self.__do_op__(Y[i, j], a[0, j], op)
                return Y
            else:
                raise ValueError('could not be broadcast')
        raise NotImplementedError('__OP__ matrix + ', type(a))

    def __add__(self, a):
        ''' matrix - scaler elementwise addition'''
        return self.__OP__(a, '+')

    def __radd__(self, a):
        ''' scaler - matrix elementwise addition'''
        ''' commutative '''
        return self.__add__(a)

    def __sub__(self, a):
        ''' matrix - scaler elementwise subtraction '''
        if type(a) in estypes:
            return self.__add__(-a)
        raise NotImplementedError('__sub__ matrix -', type(a))

    def __rsub__(self, a):
        ''' scaler - matrix elementwise subtraction '''
        self = -self
        return self.__add__(a)

    def __mul__(self, a):
        ''' matrix scaler elementwise multiplication '''
        return self.__OP__(a, '*')

    def __rmul__(self, a):
        ''' scaler * matrix elementwise multiplication
            commutative
        '''
        return self.__mul__(a)

    def __truediv__(self, a):
        ''' matrix / scaler elementwise division '''
        return self.__OP__(a, '/')

    def __rtruediv__(self, a):
        ''' scaler / matrix elementwise division '''
        return self.__OP__(a, '/')

    def __floordiv__(self, a):
        ''' matrix // scaler elementwise integer division '''
        return self.__OP__(a, '//')

    def __rfloordiv__(self, a):
        ''' scaler // matrix elementwise integer division '''
        return self.__OP__(a, '//')

    def __pow__(self, a):
        ''' matrix ** scaler elementwise power '''
        return self.__OP__(a, '**')

    def __rpow__(self, a):
        ''' scaler ** matrix elementwise power '''
        return self.__OP__(a, '**')

    def copy(self):
        """ Return a copy of matrix, not just a view """
        return matrix([i for i in self.data],
                      cstride=self.cstride, rstride=self.rstride)

    def size(self, axis=0):
        """ 0 entries
            1 rows
            2 columns
        """
        return [self.m * self.n, self.m, self.n][axis]

    @property
    def shape(self):
        return (self.m, self.n)

    @shape.setter
    def shape(self, nshape):
        """ check for proper length """
        if (nshape[0] * nshape[1]) == self.size():
            self.m, self.n = nshape
            self.cstride = 1
            self.rstride = self.n
        else:
            raise ValueError('total size of new matrix must be unchanged')
        return self

    @property
    def is_square(self):
        return self.m == self.n

    def reshape(self, nshape):
        """ check for proper length """
        X = self.copy()
        X.shape = nshape
        return X

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        """ Return a view """
        X = matrix(self.data, cstride=self.rstride, rstride=self.cstride)
        if self.cstride == self.rstride:
            # handle column vector
            X.shape = (self.n, self.m)
        return X

    def reciprocal(self, n=1):
        return matrix([n / i for i in self.data], cstride=self.cstride, rstride=self.rstride)

    def apply(self, func, *args, **kwargs):
        """ call a scalar function on each element, returns a new matrix
        passes *args and **kwargs to func unmodified
        note: this is not useful for matrix-matrix operations
        e.g.
            y = x.apply(math.sin)
            y = x.apply(lambda a,b: a>b, 5) # equivalent to y = x > 5
            y = x.apply(operators.gt, 5)    # equivalent to y = x > 5 (not in micropython)
        """
        return matrix([func(i, *args, **kwargs) for i in self.data],
                      cstride=self.cstride, rstride=self.rstride)

def matrix_isclose(x, y, rtol=1E-05, atol=flt_eps):
    ''' Returns a matrix indicating equal elements within tol'''
    for i in range(x.size()):
        try:
            data = [abs(x.data[i] - y.data[i]) <= atol+rtol*abs(y.data[i]) for i in range(len(x.data))]
        except (AttributeError, IndexError):
            data = [False for i in range(len(x.data))]
    return matrix(data, cstride=x.cstride, rstride=x.rstride, dtype=bool)


def matrix_equal(x, y, tol=0):
    ''' Matrix equality test with tolerance same shape'''
    res = False
    if type(y) == matrix:
        if x.shape == y.shape:
            res = all([abs(x.data[i] - y.data[i]) <= tol for i in range(x.size())])
    return res


def matrix_equiv(x, y):
    ''' Returns a boolean indicating if X and Y share the same data and are broadcastable'''
    res = False
    if type(y) == matrix:
        if x.size() == y.size():
            res = all([x.data[i] == y.data[i] for i in range(len(x.data))])
    return res

def fp_eps():
    ''' Determine floating point resolution '''
    e = 1
    while 1 + e > 1:
        e = e / 2
    return 2 * e

flt_eps = fp_eps()
try:
    if sys.implementation.name == 'micropython' and sys.platform == 'linux':
        # force this as there seems to be some interaction with
        # some operations done using the C library with a smaller epsilon (doubles)
        flt_eps = 1.19E-7   # single precision IEEE 2**-23  double 2.22E-16 == 2**-52
except:
    pass
# Determine supported types
try:
    stypes.append(float)
    ddtype = float
except:
    pass
try:
    stypes.append(complex)
except:
    pass
# extended types
estypes = [matrix]
estypes.extend(stypes)
#-----------------------------------------------#
#------------------------ulinalg-----------------------#

def zeros(m, n, dtype=ddtype):
    return matrix([[0 for i in range(n)] for j in range(m)], dtype=dtype)


def ones(m, n, dtype=ddtype):
    return zeros(m, n, dtype) + 1


def eye(m, dtype=ddtype):
    Z = zeros(m, m, dtype=dtype)
    for i in range(m):
        Z[i, i] = 1
    return Z

def det_inv(x):
    ''' Return (det(x) and inv(x))

        Operates on a copy of x
        Using elementary row operations convert X to an upper matrix
        the product of the diagonal = det(X)
        Continue to convert X to the identity matrix
        All the operation carried out on the original identity matrix
        makes it the inverse of X
    '''
    if not x.is_square:
        raise ValueError('Matrix must be square')
    else:
        # divide each row element by [0] to give a one in the first position
        # (may have to find a row to switch with if first element is 0)
        x = x.copy()
        inverse = eye(len(x), dtype=float)
        sign = 1
        factors = []
        p = 0
        while p < len(x):
            d = x[p, p]
            if abs(d) < flt_eps:
                # pivot == 0 need to swap a row
                # check if swap row also has a zero at the same position
                np = 1
                while (p + np) < len(x) and abs(x[p + np, p]) < flt_eps:
                    np += 1
                if (p + np) == len(x):
                    # singular
                    return [0, []]
                # swap rows
                z = x[p + np]
                x[p + np, :] = x[p]
                x[p, :] = z
                # do identity
                z = inverse[p + np]
                inverse[p + np, :] = inverse[p]
                inverse[p, :] = z
                # change sign of det
                sign = -sign
                continue
            factors.append(d)
            # change target row
            for n in range(p, len(x)):
                x[p, n] = x[p, n] / d
            # need to do the entire row for the inverse
            for n in range(len(x)):
                inverse[p, n] = inverse[p, n] / d
            # eliminate position in the following rows
            for i in range(p + 1, len(x)):
                # multiplier is that column entry
                t = x[i, p]
                for j in range(p, len(x)):
                    x[i, j] = x[i, j] - (t * x[p, j])
                for j in range(len(x)):
                    inverse[i, j] = inverse[i, j] - (t * inverse[p, j])
            p = p + 1
        s = sign
        for i in factors:
            s = s * i  # determinant
        # travel through the rows eliminating upper diagonal non-zero values
        for i in range(len(x) - 1):
            # final row should already be all zeros
            # except for the final position
            for p in range(i + 1, len(x)):
                # multiplier is that column entry
                t = x[i, p]
                for j in range(i + 1, len(x)):
                    x[i, j] = x[i, j] - (t * x[p, j])
                for j in range(len(x)):
                    inverse[i, j] = inverse[i, j] - (t * inverse[p, j])
        return (s, inverse)


def inverse_matrix(m):
    # 获取矩阵的大小
    n = len(m)
    # 构造增广矩阵
    aug = zeros(n,2*n,dtype=float)
    for i in range(n):
        for j in range(n):
            aug[i,j] = m[i,j]
        aug[i,i + n] = 1
    # 对增广矩阵进行高斯消元法
    for i in range(n):
        # 如果主对角线上的元素为0，则交换该列的不为0的行到主对角线上
        if aug[i,i] == 0:
            for j in range(i + 1, n):
                if aug[j,i] != 0:
                    #交换两行
                    tep = aug[i,:]
                    aug[i,:]=aug[j,:]
                    aug[j,:]=tep
                    break
            else:
                return None # 如果没有找到不为0的元素，则说明没有逆矩阵
        # 将该行除以主元，使主对角线上的元素为1
        aug[i,:]=aug[i,:]/aug[i,i]
        # 将其他行加上该行乘以相应的系数，使其他列在该行的位置为0
        for j in range(n):
            if j != i:
                aug[j,:] = aug[j,:]-aug[j,i]*aug[i,:]
    # 返回右半部分作为逆矩阵
    return aug[:,n:2*n]


def pinv(X):
    ''' Calculates the pseudo inverse Adagger = (A'A)^-1.A' '''
    Xt = X.transpose()
    d, Z = det_inv(dot(Xt, X))
    return dot(Z, Xt)


def dot(X, Y):
    ''' Dot product '''
    if X.size(2) == Y.size(1):
        Z = []
        for k in range(X.size(1)):
            for j in range(Y.size(2)):
                Z.append(sum([X[k, i] * Y[i, j] for i in range(Y.size(1))]))
        return matrix(Z, cstride=1, rstride=Y.size(2))
    else:
        raise ValueError('shapes not aligned')


def cross(X, Y, axis=1):
    ''' Cross product
        axis=1 Numpy default
        axis=0 MATLAB, Octave, SciLab default
    '''
    if axis == 0:
        X = X.T
        Y = Y.T
    if (X.n in (2, 3)) and (Y.n in (2, 3)):
        if X.m == Y.m:
            Z = []
            for k in range(min(X.m, Y.m)):
                z = X[k, 0] * Y[k, 1] - X[k, 1] * Y[k, 0]
                if (X.n == 3) and (Y.n == 3):
                    Z.append([X[k, 1] * Y[k, 2] - X[k, 2] * Y[k, 1],
                              X[k, 2] * Y[k, 0] - X[k, 0] * Y[k, 2], z])
                else:
                    Z.append([z])
            if axis == 0:
                return matrix(Z).T
            else:
                return matrix(Z)
        else:
            raise ValueError('shape mismatch')
    else:
        raise ValueError('incompatible dimensions for cross product'
                         ' (must be 2 or 3)')

def eps(x = 0):
    # ref. numpy.spacing(), Octave/MATLAB eps() function
    if x:
        return 2**(math.floor(math.log(abs(x))/math.log(2)))*flt_eps
    else:
        return flt_eps

#-----------------------------------------------#

#-----------------------------------------------#
#找特定长宽比和大小的矩形
def Find_rec(img,w_h_min,w_h_max,size_min,size_max=640*480):
    for r in img.find_rects(threshold=8000):
        #r是一个矩形对象，直接获得的w和h是bbox的属性，并不是矩形的,corners左下起逆时针
        corners = list(r.corners())
        x1,y1 = corners[0]
        x2,y2 = corners[1]
        len_x = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        x1,y1 = corners[2]
        len_y = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        if len_y < 0.01:
            continue
        w_h = len_x/len_y
        bbox_size = r.w()*r.h()
        if w_h >w_h_min and w_h <w_h_max and bbox_size > size_min and bbox_size < size_max:
            return 1,r
    return 0,None

#A4识别模块
tar=matrix([[0,0],[140,0],[140,100],[0,100]])
def getPerspectMat(ls ,tar_ls):
    pre_trans = []
    aft_trans = []
    H=[]
    for i in range(4):
        pre_trans.append([ls[i,0],ls[i,1],1,0,0,0,-tar_ls[i,0]*ls[i,0],-tar_ls[i,0]*ls[i,1],
               0,0,0,ls[i,0],ls[i,1],1,-tar_ls[i,1]*ls[i,0],-tar_ls[i,1]*ls[i,1]])
        aft_trans.append([tar_ls[i,0],
               tar_ls[i,1]])
    pre_trans = matrix(pre_trans).reshape([8,8])
    #check if the pre_trans inversible
    det,x = det_inv(pre_trans)
    if det < flt_eps or det > 1.19E7:
        y = inverse_matrix(pre_trans*10)*10
    else:
        y = inverse_matrix(pre_trans)
    aft_trans = matrix(aft_trans).reshape([8,1])
    tep = dot(y,aft_trans)
    for i in range(8):
        H.append([tep[i]])
    H.append([1])
    H = matrix(H,dtype=float).reshape([3,3])
    return H
def map_recog(img,tar_ls):
    #灰度图
    pre_point=[]
    rect_coord=[]
    flag,r = Find_rec(img,1.35,1.45,8000)
    if flag:
        for p in r.corners():
            rect_coord.append([p[0], p[1]])
        rect_coord = matrix(rect_coord)
        H = getPerspectMat(rect_coord, tar_ls)
        if type(H) != matrix:
            pass
        for c in img.find_circles(roi=r.rect(), threshold=1500, x_margin=10, y_margin=10,
                                  r_margin=10, r_min=2,
                                  r_max=6, r_step=1):
            if c.y() - r.y() > 5 and c.y() - r.y() - r.h() < -5:
                pre_point.append([c[0], c[1], 1])
        pre_point = matrix(pre_point)
        aft_point = pre_point
        aft_point = dot(H, pre_point.T)
        for i in range(aft_point.n):
            aft_point[0, i] = aft_point[0, i] * 5 / aft_point[2, i]
            aft_point[1, i] = aft_point[1, i] * 5 / aft_point[2, i]
        aft_point = aft_point // 20 + 1
        return aft_point[0:2, :]

def recognize():
    tep = []
    if DEBUG:
        img = sensor.snapshot()
        point = map_recog(img,tar)
        if type(point) == matrix:
            tep = point

    tep = matrix([[6.0 , 20.0, 24.0, 8.0 , 33.0, 5.0 , 28.0, 11.0, 18.0, 14.0, 14.0, 28.0, 10.0, 4.0 , 18.0, 14.0, 3.0 , 25.0],
     [22.0, 20.0, 13.0, 9.0 , 9.0 , 6.0 , 5.0 , 4.0 , 4.0 , 22.0, 17.0, 17.0, 17.0, 14.0, 13.0, 10.0, 10.0, 9.0 ]])
    print("recognize")
    return tep
#-----------------------------------------------#
#-----------------------------------------------#
#图像分类模块
def classify():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)  # we run out of memory if the resolution is much bigger...
    sensor.set_brightness(2000)
    sensor.skip_frames(time=20)
    sensor.set_auto_gain(False)  # must turn this off to prevent image washout...
    sensor.set_auto_whitebal(False, (0, 0x80, 0))  # must turn this off to prevent image washout...
    net_path = "mymodel.tflite"  # 定义模型的路径
    labels = [line.rstrip() for line in open("/sd/labels_animal_fruits.txt")]  # 加载标签
    net = tf.load(net_path, load_to_fb=True)  # 加载模型
    img = sensor.snapshot().binary([(0, 100, -128, 127, -128, 0)])  # (0, 100, -128, 127, -128, 0)蓝色
    flag, r = Find_rec(img, 0.95, 1.05, 3600)
    if flag:
        img.draw_rectangle(r.rect(), color=(255, 0, 0))  # 绘制矩形外框，便于在IDE上查看识别到的矩形位置
        img1 = sensor.snapshot().copy(1, 1, r.rect())  # 拷贝矩形框内的图像
        # 将矩形框内的图像使用训练好的模型进行分类
        # tf.classify()将在图像的roi上运行网络(如果没有指定roi，则在整个图像上运行)
        # 将为每个位置生成一个分类得分输出向量。
        # 在每个比例下，检测窗口都以x_overlap（0-1）和y_overlap（0-1）为指导在ROI中移动。
        # 如果将重叠设置为0.5，那么每个检测窗口将与前一个窗口重叠50%。
        # 请注意，重叠越多，计算工作量就越大。因为每搜索/滑动一次都会运行一下模型。
        # 最后，对于在网络沿x/y方向滑动后的多尺度匹配，检测窗口将由scale_mul（0-1）缩小到min_scale（0-1）。
        # 下降到min_scale(0-1)。例如，如果scale_mul为0.5，则检测窗口将缩小50%。
        # 请注意，如果x_overlap和y_overlap较小，则在较小的比例下可以搜索更多区域...

        # 默认设置只是进行一次检测...更改它们以搜索图像...
        for obj in tf.classify(net, img1):
            print("**********\nTop 1 Detections at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
            sorted_list = sorted(zip(labels, obj.output()), key=lambda x: x[1], reverse=True)
            # 打印准确率最高的结果
            for i in range(1):
                print("%s = %f" % (sorted_list[i][0], sorted_list[i][1]))
                return labels.index(sorted_list[i][0])

#-----------------------------------------------#
#-----------------------------------------------#
#通信模块
import struct
#struct将字节串解读为打包的二进制数据
def Send_start(uart):
    uart.write("m")
def Send_end(uart):
    uart.write("l")
def Send_float(uart,bytes):
    uart.write(struct.pack("<f",bytes))
#要在主循环中轮询。
def Send_loc(uart,point_ls:matrix,i):
    if i < 2 * point_ls.n:
        Send_float(uart, point_ls[i % 2, i // 2])
        return i + 1
    else:
        Send_float(uart,100.0)
        return 0

def Read_line(uart,flag):
    tep = uart.readline().decode().strip().split(",")
    if tep != [""] :
        if tep == ["M"]:
            #A4坐标
            flag = 1
        elif tep == ["C"]:
            #图片分类
            flag = 2
        elif tep == ["T"]:
            #发送坐标点
            flag = 3
        elif tep == ["T"]:
            #发送图片识别结果
            flag = 4
    return tep,flag




#-----------------------------------------------#
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000)
flag = 0
times = 0
point = []
cls = -1
while(True):
    tep,flag = Read_line(uart,flag)
    if flag == 1:
        print(tep)
        recognize()
        flag = 0
    elif flag == 2:
        print(tep)
        cls = classify()
        flag = 0
    elif flag == 3:
        print(tep)
        times = Send_loc(uart,point,times)
        flag = 0
    elif flag == 4:
        print(tep)
        Send_float(uart,cls)
        flag = 0




