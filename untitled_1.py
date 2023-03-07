# import sensor, image, time,tf
# from machine import UART
# import time
# import pyb
# from pyb import LED
# from machine import Pin

import math
class array:
    def __init__(self,M:list):
        self.M= M
        self.shape = self.get_shape()
        self.ndim = len(self.shape)
    def __len__(self):
        return len(self.M)
    def __getitem__(self, *args):
        if isinstance(args[0],tuple):
            assert len(args[0]) <= self.ndim, 'out'
            indexs =list(args[0])
            def get_value(a,num):
                if len(indexs)-1==num:
                    return a[indexs[num]]
                return get_value(a[indexs[num]],num+1)
            return get_value(self.M,0)
        elif isinstance(args[0],int):
            return self.M[args[0]]
    def get_shape(self):
        shape = []
        def get_len(a):
            try:
                shape.append(len(a))
                get_len(a[0])
            except:
                pass
        get_len(self.M)
        return tuple(shape)

    def __add__(self, other):
        assert self.ndim == 2 and other.ndim == 2 and self.shape == other.shape
        r, w = self.shape
        return array([[self[i][j] + other[i][j] for j in range(w)] for i in range(r)])
    def __sub__(self, other):
        assert self.ndim == 2 and other.ndim == 2 and self.shape == other.shape
        r, w = self.shape
        return array([[self[i][j] - other[i][j] for j in range(w)] for i in range(r)])
    def __mul__(self, other):
        if isinstance(other,(int,float)):
            other = eye(self.shape[1],other)
        assert self.ndim == 2 and other.ndim == 2
        r_a, w_a = self.shape
        r_b, w_b = other.shape
        assert w_a == r_b, '无法相乘'
        def l(i, j):
            return sum([self[i][t] * other[t][j] for t in range(w_a)])
        return array([[l(i, j) for j in range(w_b)] for i in range(r_a)])

    @property
    def T(self):
        assert self.ndim==2
        r, w = self.shape
        B = array([[self[j][i] for j in range(r)] for i in range(w)])
        return B
    def det(self):
        shape = self.shape
        assert self.ndim==2 and shape[0] == shape[1], '非方阵'
        r, c = shape
        m = [[self.M[i][j] for j in range(c)] for i in range(r)]
        ans=1
        for col in range(c):
            v = [math.fabs(row[col]) for row in m[col:]]
            pivot = max(v)
            if pivot==0:
                return 0
            pivot_index = v.index(pivot)+col
            pivot = m[pivot_index][col]
            pivot_row = [x / pivot for x in m[pivot_index]]
            ans=ans*pivot
            if pivot_index!=col:
                temp = m[col]
                m[col]=pivot_row
                m[pivot_index] =temp
                ans*=-1#互换行列式两行，需要变号
            else:
                m[col] = pivot_row

            for i in range(col+1,r):
                k = m[i][col]
                m[i] = [ m[i][j]-k*pivot_row[j] for j in range(c)]
        return ans
    def inv(self):
        shape = self.shape
        assert self.det()!=0,'方阵不可逆'
        r, c = shape
        m = [[self.M[i][j] for j in range(c)] for i in range(r)]
        I = [[1 if i==j else 0 for i in range(r)] for j in range(r)]
        for col in range(c):
            v = [math.fabs(row[col]) for row in m[col:]] #选出从第col行开始的第col列
            pivot = max(v)
            if pivot==0:
                return 0
            pivot_index = v.index(pivot)+col
            pivot = m[pivot_index][col]
            pivot_row = [x/pivot for x in m[pivot_index]]
            I_pivot_row = [x/pivot for x in I[pivot_index]]
            if pivot_index!=col:#行互换
                temp = m[col]
                m[col]=pivot_row
                m[pivot_index] =temp

                I_temp = I[col]
                I[col] = I_pivot_row
                I[pivot_index]=I_temp
            else:
                m[col] = pivot_row
                I[col] = I_pivot_row
            for i in range(r):
                if i!=col:
                    k = m[i][col]
                    #对应行相减
                    m[i] = [ m[i][j]-k*pivot_row[j] for j in range(c)]
                    I[i] = [I[i][j]-k*I[col][j]for j in range(c)]
        return array(I)
    @staticmethod
    def A_yu(A, I, J):
        r = len(A[0])
        M = []
        for i in range(r):
            if i != I:
                row = []
                for j in range(r):
                    if j != J:
                        row.append(A[i][j])
                M.append(row)
        return array(M)
    # def det(self):#递归太慢，弃用
    #     assert len(self.shape) == 2 and self.shape[0] == self.shape[1], '非方阵'
    #     if len(self) == 1:
    #         return self[0][0]
    #     ans = 0
    #     r = len(self[0])
    #     for t in range(r):
    #         M = self.A_yu(self.M, 0, t)
    #         ans += (-1) ** (1 + t + 1) * self[0][t] * M.det()
    #     return ans
    # def inv(self):
    #     assert len(self.shape) == 2 and self.shape[0] == self.shape[1], '非方阵'
    #     assert self.det()!=0,'方阵不可逆'
    #     A_star = []
    #     r, w = self.shape
    #     c = self.det()
    #     for i in range(r):
    #         row = []
    #         for j in range(w):
    #             row.append((-1) ** (i + j) * self.A_yu(self, i, j).det()/c)
    #         A_star.append(row)
    #     A_inv = array(A_star).T
    #     return A_inv
    def __str__(self):
        return str(self.M)



def eye(size,value=1):
    M = [[value if i==j else 0 for i in range(size)] for j in range(size)]
    return array(M)

def full(shape:tuple,value):
    def add(m,index):
        if index<0:
            return m
        M=[]
        for _ in range(shape[index]):
            M.append(m)
        return add(M,index-1)

    M = add([value for _ in range(shape[-1])],len(shape)-1-1)
    return array(M)

def zeros(shape:tuple):
    return full(shape,0)

def ones(shape:tuple):
    return full(shape,1)

#解线性方程组
def solve(A:array,B:array)->array:
    if A.det()==0:
        raise ValueError("无解")
    assert B.ndim==2 and B.shape[0]==A.shape[0] and B.shape[1]==1
    r, c = A.shape
    m = [[A.M[i][j] for j in range(c)] for i in range(r)]
    b = [[B.M[i][0]] for i in range(r)]
    for col in range(c):
        v = [math.fabs(row[col]) for row in m[col:]] #选出从第col行开始的第col列
        pivot = max(v)
        if pivot==0:
            return 0
        pivot_index = v.index(pivot)+col
        pivot = m[pivot_index][col]
        pivot_row = [x/pivot for x in m[pivot_index]]
        b_pivot_row = [x/pivot for x in b[pivot_index]]
        if pivot_index!=col:#行互换
            temp = m[col]
            m[col]=pivot_row
            m[pivot_index] =temp

            b_temp = b[col]
            b[col] = b_pivot_row
            b[pivot_index]=b_temp

        else:
            m[col] = pivot_row
            b[col] = b_pivot_row
        for i in range(r):
            if i!=col:
                k = m[i][col]
                #对应行相减
                m[i] = [ m[i][j]-k*pivot_row[j] for j in range(c)]
                b[i] = [b[i][0]-k*b[col][0]]
    return array(b)

#
# sensor.reset()
# sensor.set_pixformat(sensor.RGB565)
# sensor.set_framesize(sensor.QVGA)
# sensor.set_brightness(2000)
# sensor.skip_frames(time = 100)
# sensor.set_auto_gain(True)
# sensor.set_auto_whitebal(True)
# sensor.set_auto_exposure(False,100)
#


world_coordinates = [[40,140],
                     [180,140],
                     [180,40],
                     [40,40]]

#返回透视矩阵
#XY为世界坐标，UV为相机坐标
def cal_mtx(UV:array,XY:array)->array:
    A = []
    B =[]
    for i in range(4):
        a = [[UV[i][0],UV[i][1],1,0,0,0,-XY[i][0]*UV[i][0],-XY[i][0]*UV[i][1]],
             [0,0,0,UV[i][0],UV[i][1],1,-XY[i][1]*UV[i][0],-XY[i][1]*UV[i][1]]]
        B+= [[XY[i][0]],
             [XY[i][1]]]
        A+=a

    A = array(A)
    B = array(B)

    x= solve(A,B)

    H = [[x[0][0], x[1][0], x[2][0]],
         [x[3][0], x[4][0], x[5][0]],
         [x[6][0], x[7][0], 1]]

    return array(H)

def map(img):

    for r in img.find_rects(threshold = 10000):
        #img.draw_rectangle(r.rect(), color = (255, 0, 0))
        point_num=0
        if r.w()>=200 and r.h()>=100 and r.w()<=300 and r.h()<=300:
            img.draw_rectangle(r.rect(), color = (255, 0, 0))
            print(r.rect())
            img_coordinate=[]

            points =[]
            for c in img.find_circles(roi =r.rect(),threshold = 1500, x_margin = 10, y_margin = 10, r_margin = 10,r_min = 2, r_max =6, r_step = 1):


                c_roi=(c.x()-c.r(),c.y()-c.r(),2*c.r(),2*c.r())
                threshold = (0, 58, -87, 127,-128, 127)
                blobs = img.find_blobs([threshold],roi = c_roi, pixels_threshold=int(0.2*4*c.r()*c.r()), area_threshold=1,
                                        merge=True, margin=10, invert=False)
                img.draw_circle(c.x(), c.y(),2, color = (0, 255, 0),thickness = 4)
                if len(blobs):
                    points.append([c[0], c[1], 1])

            if show:
                for p in r.corners():
                    img.draw_circle(p[0], p[1], 2, color = (0, 0, 255))
                    img_coordinate.append([p[0], p[1]])

            img_coordinate[0][1]-=3
            img_coordinate[1][1]-=3
            point_num += 1
            return  img_coordinate,points
    return None,None


tep = array([[60.0    , 48.0    , 1.0     , 0.0     , 0.0     , 0.0     , -0.0    , -0.0    ],
     [0.0     , 0.0     , 0.0     , 60.0    , 48.0    , 1.0     , -0.0    , -0.0    ],
     [288.0   , 48.0    , 1.0     , 0.0     , 0.0     , 0.0     , -40320.0, -6720.0 ],
     [0.0     , 0.0     , 0.0     , 288.0   , 48.0    , 1.0     , -0.0    , -0.0    ],
     [288.0   , 206.0   , 1.0     , 0.0     , 0.0     , 0.0     , -40320.0, -28840.0],
     [0.0     , 0.0     , 0.0     , 288.0   , 206.0   , 1.0     , -28800.0, -20600.0],
     [60.0    , 206.0   , 1.0     , 0.0     , 0.0     , 0.0     , -0.0    , -0.0    ],
     [0.0     , 0.0     , 0.0     , 60.0    , 206.0   , 1.0     , -6000.0 , -20600.0]])
print(tep.inv())







show =True
Debug = 0
if Debug:
    while(True):
        img = sensor.snapshot()
        #img = img.lens_corr(strength = 0.8, zoom = 1.0).histeq(adaptive=True, clip_limit=3)

        img_coordinate,points =map(img)#地图识别

        if img_coordinate==None:
            continue

        img_coordinate =array(img_coordinate)
        world_coordinates =array(world_coordinates)

        H= cal_mtx(img_coordinate,world_coordinates)


        real_point = []
        img.draw_rectangle(40,40,140,100, color = (0, 0, 255))

        show_points=[]
        for c in points:

            coord = array([[p] for p in c])#升维
            point = H*coord#透视变换

            x,y = point[0][0]/point[2][0],point[1][0]/point[2][0]
            #点不会太靠近边界
            if x>=40+5 and x<180-5 and y>=40+5 and y<=140-5:
                if show:
                    img.draw_circle(c[0], c[1], 3, color=(255, 0, 0), thickness=4)

                img.draw_circle(int(x), int(y),2, color = (0, 0, 255),thickness = 4)
                x,y = x-40,100-y+(40)
                x,y = (x*5),(y*5)
                show_points.append([x//20+1,y//20+1])
                x = (x//20)*20+10
                y = (y//20)*20+10
                real_point.append([int(x),int(y)])
                print(real_point)
