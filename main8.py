import random
import math
import numpy as np
from matplotlib import pyplot as plt

class source: #класс источника
    def __init__(self,x0,y0,z0):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

def ray(): #функция задающая произвольные параметры l,n,m для прямой
    while True:
        l = random.uniform(-1, 1)
        n = random.uniform(-1, 1)
        m = random.uniform(- 1, 1)
        length = (l ** 2 + n ** 2 + m ** 2) ** 0.5
        if length < 1:
            break
    l = l / length
    n = n / length
    m = m / length
    return l,n,m

class cl_plane(): #класс плоскостей
    def __init__(self, P1, P2, P3 ):
        self.A = P1[1]*P2[2]+P2[1]*P3[2]+P3[1]*P1[2]-P3[1]*P2[2]-P1[1]*P3[2]-P2[1]*P1[2]
        self.B = P1[2]*P2[0]+P2[2]*P3[0]+P3[2]*P1[0]-P3[2]*P2[0]-P1[2]*P3[0]-P2[2]*P1[0]
        self.C = P1[0]*P2[1]+P2[0]*P3[1]+P3[0]*P1[1]-P3[0]*P2[1]-P1[0]*P3[1]-P2[0]*P1[1]
        self.D = -(P1[2] * P2[0] * P3[1] + P1[1] * P2[2] * P3[0] + P1[0] * P2[1] * P3[2] - P1[1] * P2[0] * P3[2] - P1[2] * P2[1] * P3[0] - P1[0] * P2[2] * P3[1])

def cylinder(s):
    count = 0
    Ech=[]
    Nch=[0]*1000 # число частиц в канале
    for i in range(1000):
        Ech.append(i/1000)#в МэВ
    N=1000
    for j in range(N):
        l,n,m = ray()#направляем произвольный луч
        E=0.665 #задаем энергию
        exit=0
        while (E>Ech[1]) and (exit==0):
            cross = []
            t_cross = []
            Ph_Na=func_sigmaPh(E, 11) # расчет длины пробега через сечение
            Ph_I=func_sigmaPh(E, 53)
            K_Na=func_sigmaK(E, 11)
            K_I=func_sigmaK(E, 53)
            sigmaPh=5/4*6.02*10**23*(Ph_Na*11/23+Ph_I*53/127)
            sigmaK = 6.02 * 10 ** 23 * (K_Na * 11 / 23 + K_I * 53 / 127)

            sigma=sigmaPh+sigmaK
            length=-1 / sigma * np.log(random.uniform(0,1))

            count_pl=0 # счетчик пересечения плоскостей
            for i in range(2):#пересекаем по очереди со всеми плоскостями
                znamen = (plane[i].A*l+plane[i].B*n+plane[i].C*m)
                if znamen != 0:
                    t = - (plane[i].A*s.x0+plane[i].B*s.y0+plane[i].C*s.z0+plane[i].D)/znamen # пересечение произвольной прямой и плоскости
                    if t > 0.:
                        x = s.x0+l*t #координаты пересечения
                        y = s.y0+n*t
                        z = s.z0+m*t
                        if (x ** 2 + y ** 2) < r ** 2:
                            count_pl+=1
                            cross.append(source(x,y,z))#точка входа/выхода луча
                            t_cross.append(t)
            discr = (l*s.x0+n*s.y0)**2-(l**2+n**2)*(s.x0**2+s.y0**2-r**2) #деленный на 4
            if discr>0.0:
                t1 = (-(l * s.x0 + n * s.y0) - discr ** 0.5) / (l ** 2 + n ** 2)
                t2 = (-(l * s.x0 + n * s.y0) + discr ** 0.5) / (l ** 2 + n ** 2)
                if t1>0.0:
                    x = s.x0 + l * t1 # координаты пересечения
                    y = s.y0 + n * t1
                    z = s.z0 + m * t1
                    if (np.abs(z)< d/2):
                        count_pl += 1
                        cross.append(source(x, y, z))
                        t_cross.append(t)
                if (t2>0.0):
                    x = s.x0 + l * t2 # координаты пересечения
                    y = s.y0 + n * t2
                    z = s.z0 + m * t2
                    if (np.abs(z)< d/2):
                        count_pl += 1
                        cross.append(source(x, y, z))
                        t_cross.append(t)

            if count_pl >0:
                count += 1
                t_cross.append(t_cross[0])#на случай одного пересечения (если пересечения 2, просто добавится третий элемент) нигде не будет учитываться
                if t_cross[0] > t_cross[1]:
                    help = cross[0]
                    cross[0] = cross[1]
                    cross[1] = help
                init = source(cross[0].x0 + length / l, cross[0].y0 + length / n, cross[0].z0 + length / m) # точка взаимодействия
                if ((d/2>init.x0**2+init.y0**2)):
                    E, l,n,m=Lottery(sigmaPh, sigmaK, E, l, n, m)
                    s = init
                else:
                    exit=1 #частица вылетает из цилиндра
            else:
                exit=1 #частица не перескает цилиндр

        Eloss=0.665-E
        k=0
        while (k<1000):#канал энергии
            if Eloss>Ech[999-k]:
                Nch[999-k]+=1
                k=1000
            else:
                k+=1
    return Ech, Nch

def func_sigmaPh(E, Z):
    return 6.651*10**(-25)*4*(2**0.5)*(Z**5)/(137**4)*(0.511/E)**(7/2)

def func_sigmaK(E,Z):
    g = E/0.511
    return 6.651*10**(-25)*3*Z/(8*g)*((1-2*(g+1)/(g**2))*math.log(2*g+1)+0.5+4/g-1/(2*(2*g+1)**2))

def Lottery(sigmaPh, sigmaK,E, l,n,m):
    help=random.uniform(0, sigmaPh+sigmaK )
    if help <= sigmaPh:
        E = 0
        l2, n2, m2 = [0, 0, 0]
    else:
        l2, n2, m2 = ray()
    cos=(l*l2+n*n2+m*m2)/((l**2+n**2+m**2)**0.5+(l2**2+n2**2+m2**2)**0.5)
    E_loss=E/(1+E/0.511*(1-cos))
    E-=E_loss
    return E, l2, n2, m2

def func_Eloss(cos, E): #потерянная после Комптоновского рассеяния энергия
    return 1

def interaction(Pcross, l, n, m, Length):
    x_int=Pcross[0]+Length/l
    y_int = Pcross[1]+Length/n
    z_int = Pcross[2]+Length/m
    P_int = source(x_int, y_int, z_int)
    return P_int

def in_cylinder(): # взаимодействие внутри цилиндра
    return 1

d=10 #задана длина цилиндра
r=5 #задан радиус цилиндра

P=[[0,0,d/2],[r,0,d/2],[0,r,d/2],[0,0,-d/2],[r,0,-d/2],[0,r,-d/2]]
plane = [] # задано плоскости
pl = cl_plane(P[0],P[1],P[2]) # верхняя плоскость 0
plane.append(pl)
pl = cl_plane(P[3], P[4], P[5])
plane.append(pl)
l = 10
s = source(0, 0, 1)
E,N = cylinder(s)
plt.figure(figsize=(20,10))
plt.bar(E, N, width=0.001)
plt.yscale('log')
plt.grid(True)
plt.show()