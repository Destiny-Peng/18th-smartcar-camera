#-----------------------------------------------#
#通信模块
import struct
import umatrix,ulinalg
#struct将字节串解读为打包的二进制数据
def Send_start(uart):
    uart.write("ST\r\n")
def Send_end(uart):
    uart.write("END\r\n")
def Send_float(uart,bytes):
    uart.write(struct.pack("<f",bytes))
def Send_tuple(uart,tup):
    x,y=tup
    x = float(x)
    y = float(y)
    uart.write("tup\r\n")
    Send_float(uart,x)
    Send_float(uart,y)
#要在主循环中轮询。
def Read_line(uart,flag):
    #返回值为数字
    tep = uart.readline().decode().strip().split(",")
    if tep != [""]:
        if tep == ["M"] :
            #Matrix
            flag = 1
        elif tep == ["S"] :
            #Str
            flag = 2
    return tep


def Request_Trans_Mat(uart,ls,tar_ls):
    Send_start(uart)
    for i in range(0,len(ls)):
        Send_tuple(uart,ls[i])
    for i in range(0,len(tar_ls)):
        Send_tuple(uart,tar_ls[i])
    Send_end(uart)
def Receive_Trans_Mat(uart,mat:umatrix.matrix):
    if mat.shape == (3,3) :
        tep = Read_line(uart)
        for i in range(0,2):
            for j in range(0,2):
                mat[i,j]=tep[i*3+j]
        return True
    else:
        return False
def getPerspectMat(ls ,tar_ls):
    pre_trans = []
    aft_trans = []
    H=[]
    for i in range(4):
        pre_trans.append([ls[i,0],ls[i,1],1,0,0,0,-tar_ls[i,0]*ls[i,0],-tar_ls[i,0]*ls[i,1],
               0,0,0,ls[i,0],ls[i,1],1,-tar_ls[i,1]*ls[i,0],-tar_ls[i,1]*ls[i,1]])
        aft_trans.append([tar_ls[i,0],
               tar_ls[i,1]])
    pre_trans = umatrix.matrix(pre_trans).reshape([8,8])
    #check if the pre_trans inversible
    det,x = ulinalg.det_inv(pre_trans)
    if det < umatrix.flt_eps or det > 1.19E7:
        y = ulinalg.inverse_matrix(pre_trans*10)*10
    else:
        y = ulinalg.inverse_matrix(pre_trans)
    aft_trans = umatrix.matrix(aft_trans).reshape([8,1])
    tep = ulinalg.dot(y,aft_trans)
    for i in range(8):
        H.append([tep[i]])
    H.append([1])
    H = umatrix.matrix(H,dtype=float).reshape([3,3])
    return H
def map_recog(img,tar_ls):
    #灰度图
    pre_point=[]
    rect_coord=[]
    for r in img.find_rects(threshold=10000):
        w_h = r.w()/r.h()
        if r.w() >= 200 and w_h >1.3 and w_h <1.45:
            for p in r.corners():
                rect_coord.append([p[0], p[1]])
            rect_coord = umatrix.matrix(rect_coord)
            H = getPerspectMat(rect_coord, tar_ls)
            if H == None:
                continue
            for c in img.find_circles(roi=r.rect(), threshold=1500, x_margin=10, y_margin=10,
                                      r_margin=10, r_min=2,
                                      r_max=6, r_step=1):
                if abs(c.y()-r.y())>5:
                    pre_point.append([c[0], c[1], 1])
            pre_point = umatrix.matrix(pre_point)
            aft_point = pre_point
            aft_point = ulinalg.dot(H,pre_point.T)
            aft_point[0, :] = aft_point[0, :] * 5
            aft_point[1, :] = 500 - aft_point[1, :] * 5
            aft_point = aft_point // 20 + 1
            return aft_point[0:2,:]


def debug(uart,ls,tar_ls):
    Request_Trans_Mat(uart,ls,tar_ls)

    #to be continued...

tar_ls=[[0,0],[140,0],[140,100],[0,100]]
'''
[[102.1776], [61.99528], [0.9682666]]
[[147.8109], [65.02446], [0.9582563]]
[[80.99836], [77.88336], [0.969917]]
[[98.84212], [86.14917], [0.9647761]]
[[72.30547], [89.87259], [0.9696473]]
[[86.50702], [104.7733], [0.9641143]]
(74, 34, 207, 150)
'''
testls=[[60,48],[288,48],[288,206],[60,206]]
testls = umatrix.matrix(testls)
tar_ls = umatrix.matrix(tar_ls)
H = getPerspectMat(testls,tar_ls)
#print(H)
#print(y)
point = umatrix.matrix([[97, 69, 1], [186, 81, 1], [212, 126, 1], [111, 150, 1],
                  [269, 151, 1], [92, 174, 1], [237, 176, 1], [131, 182, 1],
                  [171, 184, 1], [147, 68, 1], [146, 99, 1], [238, 99, 1],
                  [119, 104, 1], [81, 123, 1], [172, 126, 1], [145, 144, 1],
                  [75, 148, 1], [219, 152, 1]])
tep = ulinalg.dot(H,point.T)
tep[0,:] = tep[0,:]*5
tep[1,:] = 500-tep[1,:]*5
tep = tep//20+1
x=tep[0:2,:]
print(x)



