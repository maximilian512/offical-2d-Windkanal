#https://github.com/dgorissen/naca/blob/master/naca.py
#def naca, def naca4, def naca5, def rotate_points, def linspace ist aus dem github repository von dgorissen kopiert um die NACA Profile zu generieren.

import numpy as np
from math import cos, sin, tan, radians
from math import atan
from math import pi
from math import pow
from math import sqrt

def linspace(start,stop,np):
    """
    Emulate Matlab linspace
    """
    return [start+(stop-start)*i/(np-1) for i in range(np)]
def naca4(number, n, finite_TE = False, half_cosine_spacing = False):
    """
    Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string
    """

    m = float(number[0])/100.0
    p = float(number[1])/10.0
    t = float(number[2:])/100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843

    if finite_TE:
        a4 = -0.1015 # For finite thick TE
    else:
        a4 = -0.1036 # For zero thick TE

    if half_cosine_spacing:
        beta = linspace(0.0,pi,n+1)
        x = [(0.5*(1.0-cos(xx))) for xx in beta]  # Half cosine based spacing
    else:
        x = linspace(0.0,1.0,n+1)

    yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx,2)+a3*pow(xx,3)+a4*pow(xx,4)) for xx in x]

    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = [-xx for xx in yt]

        xc = xc1 + xc2
        zc = [0]*len(xc)
    else:
        yc1 = [m/pow(p,2)*xx*(2*p-xx) for xx in xc1]
        yc2 = [m/pow(1-p,2)*(1-2*p+xx)*(1-xx) for xx in xc2]
        zc = yc1 + yc2

        dyc1_dx = [m/pow(p,2)*(2*p-2*xx) for xx in xc1]
        dyc2_dx = [m/pow(1-p,2)*(2*p-2*xx) for xx in xc2]
        dyc_dx = dyc1_dx + dyc2_dx

        theta = [atan(xx) for xx in dyc_dx]

        xu = [xx - yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yu = [xx + yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

        xl = [xx + yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yl = [xx - yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

    X = xu[::-1] + xl[1:]
    Z = yu[::-1] + yl[1:]

    return X,Z

def naca5(number, n, finite_TE = False, half_cosine_spacing = False):
    """
    Returns 2*n+1 points in [0 1] for the given 5 digit NACA number string
    """

    naca1 = int(number[0])
    naca23 = int(number[1:3])
    naca45 = int(number[3:])

    cld = naca1*(3.0/2.0)/10.0
    p = 0.5*naca23/100.0
    t = naca45/100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843

    if finite_TE:
        a4 = -0.1015 # For finite thickness trailing edge
    else:
        a4 = -0.1036  # For zero thickness trailing edge

    if half_cosine_spacing:
        beta = linspace(0.0,pi,n+1)
        x = [(0.5*(1.0-cos(x))) for x in beta]  # Half cosine based spacing
    else:
        x = linspace(0.0,1.0,n+1)

    yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx,2)+a3*pow(xx,3)+a4*pow(xx,4)) for xx in x]

    P = [0.05,0.1,0.15,0.2,0.25]
    M = [0.0580,0.1260,0.2025,0.2900,0.3910]
    K = [361.4,51.64,15.957,6.643,3.230]

    m = interpolate(P,M,[p])[0]
    k1 = interpolate(M,K,[m])[0]

    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]
    xc = xc1 + xc2

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = [-x for x in yt]

        zc = [0]*len(xc)
    else:
        yc1 = [k1/6.0*(pow(xx,3)-3*m*pow(xx,2)+ pow(m,2)*(3-m)*xx) for xx in xc1]
        yc2 = [k1/6.0*pow(m,3)*(1-xx) for xx in xc2]
        zc  = [cld/0.3 * xx for xx in yc1 + yc2]

        dyc1_dx = [cld/0.3*(1.0/6.0)*k1*(3*pow(xx,2)-6*m*xx+pow(m,2)*(3-m)) for xx in xc1]
        dyc2_dx = [cld/0.3*-(1.0/6.0)*k1*pow(m,3)]*len(xc2)

        dyc_dx = dyc1_dx + dyc2_dx
        theta = [atan(xx) for xx in dyc_dx]

        xu = [xx - yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yu = [xx + yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

        xl = [xx + yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yl = [xx - yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]


    X = xu[::-1] + xl[1:]
    Z = yu[::-1] + yl[1:]

    return X,Z


def rotate_points(p, angle_deg):
    x = p[0]
    y = p[1]
    angle_deg = -angle_deg
    angle_rad = radians(angle_deg)
    center_x = sum(x) / len(x)
    center_y = sum(y) / len(y)
    
    rotated_x = []
    rotated_y = []

    for i in range(len(x)):
        translated_x = x[i] - center_x
        translated_y = y[i] - center_y
        
        rotated_x_val = translated_x * cos(angle_rad) - translated_y * sin(angle_rad)
        rotated_y_val = translated_x * sin(angle_rad) + translated_y * cos(angle_rad)
        
        final_x = rotated_x_val + center_x
        final_y = rotated_y_val + center_y
        
        rotated_x.append(final_x)
        rotated_y.append(final_y)
    
    return rotated_x, rotated_y


def naca(number, n, finite_TE = False, half_cosine_spacing = False, rotation = 0):
    if len(number) in [4, 5]:
        points = naca4(number, n, finite_TE, half_cosine_spacing) if len(number) == 4 else naca5(number, n, finite_TE, half_cosine_spacing)
        if rotation != 0:
            points = rotate_points(points, rotation)
        return points
    else:
        raise Exception




def naca_in_windkanal_einbauen(
    naca_code,
    anstellwinkel,
    nx,
    ny,
    skalierung=25,
    x_offset=0,
    y_offset=0,
    horizontal_spiegeln=False,  # <--- Neuer Parameter
):
  wand = np.ones((nx, ny))

  X_mathe, Y_mathe = naca(
      number=naca_code, n=150, half_cosine_spacing=True, rotation=anstellwinkel
  )
  X_mathe = np.array(X_mathe)
  Y_mathe = np.array(Y_mathe)

  # 🌟 Horizontal spiegeln, falls gewünscht:
  if horizontal_spiegeln:
    X_mathe = 1.0 - X_mathe

  x_gitter = (X_mathe * skalierung + nx * 0.3 + x_offset).astype(int)
  y_gitter = (Y_mathe * skalierung + ny / 2 + y_offset).astype(int)

  for xi in range(nx):
    y_treffer = y_gitter[x_gitter == xi]
    if len(y_treffer) > 0:
      y_min = np.min(y_treffer)
      y_max = np.max(y_treffer)

      for yi in range(y_min, y_max + 1):
        if 0 <= xi < nx and 0 <= yi < ny:
          wand[xi, yi] = 0.0

  return wand





