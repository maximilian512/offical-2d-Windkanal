import numpy as np
import matplotlib.pyplot as plt
from flügel import naca_in_windkanal_einbauen
from numba import njit

#gehört zu inkompressibilitaet_loesen um die Berechnungen zu beschleunigen(mit claude gemacht)
#neu auch interpolaion direkt hier eingebaut, damit es schneller geht
@njit
def _inkompr_kern(u, v, p, wand, nx, ny, cp, omega, iterations):
    for _ in range(iterations):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if wand[i, j] == 0:
                    continue
                sx0 = wand[i-1, j]
                sx1 = wand[i+1, j]
                sy0 = wand[i, j-1]
                sy1 = wand[i, j+1]
                s = sx0 + sx1 + sy0 + sy1
                if s == 0:
                    continue
                df = (u[i+1, j] - u[i, j] +
                      v[i, j+1] - v[i, j])
                p_korr = -df / s * omega
                p[i, j] += cp * p_korr
                u[i,   j] -= sx0 * p_korr
                u[i+1, j] += sx1 * p_korr
                v[i, j  ] -= sy0 * p_korr
                v[i, j+1] += sy1 * p_korr

#gehört zu advektion, um die Berechnungen zu beschleunigen(mit claude gemacht)
@njit
def _advektion_kern(u, v, wand, nx, ny, h, dt):
    u_neu = u.copy()
    v_neu = v.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if wand[i, j] == 0:
                continue
            x_alt = i * h - u[i, j] * dt
            y_alt = j * h - v[i, j] * dt
            # Gitterpunkt finden
            gi = int(x_alt / h)
            gj = int(y_alt / h)
            gi = max(0, min(gi, nx - 2))
            gj = max(0, min(gj, ny - 2))
            tx = (x_alt - gi * h) / h
            ty = (y_alt - gj * h) / h
            sx = 1.0 - tx
            sy = 1.0 - ty
            # KORREKTUR (claude): u[i,j] liegt zwischen Zelle i-1 und i, v[i,j] zwischen j-1 und j.
            # Ist die Nachbarzelle Wand, muss die Geschwindigkeit dort 0 bleiben, sonst strömt Luft durch den Flügel
            if wand[i-1, j] != 0:
                u_neu[i, j] = (sx*sy*u[gi,   gj  ] + tx*sy*u[gi+1, gj  ] +
                               tx*ty*u[gi+1, gj+1] + sx*ty*u[gi,   gj+1])
            if wand[i, j-1] != 0:
                v_neu[i, j] = (sx*sy*v[gi,   gj  ] + tx*sy*v[gi+1, gj  ] +
                               tx*ty*v[gi+1, gj+1] + sx*ty*v[gi,   gj+1])
    return u_neu, v_neu



class Windkanal:
    def __init__(self, breite, hoehe, zellgrösse):
        self.nx = breite
        self.ny = hoehe
        self.h = zellgrösse
        self.rho = 1.225
        self.dt = 0.005

        #geschwindigkeit
        self.u = np.zeros((self.nx, self.ny)) #horizontale Geschwindigkeit
        self.v = np.zeros((self.nx, self.ny)) #vertikale Geschwindigkeit

        #druck
        self.p = np.zeros((self.nx, self.ny))

        #wand(0) oder fluid(1)
        self.wand = np.ones((self.nx, self.ny))

    def setze_wand(self, eintrittgeschwindigkeit):
            
        #wand oben und unten und linke
        self.wand[:, 0] = 0
        self.wand[:, -1] = 0
        self.wand[0, :] = 0   

        #linke wand wo luft eintritt
        self.u[1, :] = eintrittgeschwindigkeit
        #rechte wand wo luft austritt
        self.u[-1, :] = self.u[-2, :]  # Neumann-Bedingung für die rechte Wand
         

    def schwerkraft(self, g):
        self.v += g * self.dt * self.wand

       
    #def divergenz(self, i, j):
        #return (self.u[i, j] - self.u[i-1, j] + self.v[i, j] - self.v[i, j-1])


    #mit claude alte version von def Advektion und def inkompressibilitaet_lösen effizienter gemacht, rechnet schneller aus    
    def inkompressibilitaet_lösen(self, iterations=100, omega=1.9):
        cp = self.rho * self.h / self.dt
        self.p = np.zeros((self.nx, self.ny))
        _inkompr_kern(self.u, self.v, self.p, self.wand,
                    self.nx, self.ny, cp, omega, iterations)
        

    def advektion(self):
        self.u, self.v = _advektion_kern(
            self.u, self.v, self.wand, self.nx, self.ny, self.h, self.dt)



    def extrapolieren(self):
        # oben und unten
        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]
        # links und rechts
        self.v[0, :] = self.v[1, :]
        self.v[-1, :] = self.v[-2, :]

    def auftrieb_berechnen(self, v_ein):
        
        nur_flügel = self.wand.copy()
        nur_flügel[:, 0] = 1 #boden  
        nur_flügel[:, -1] = 1  #decke
        nur_flügel[0, :] = 1   #links
        nur_flügel[-1, :] = 1 #rechts
        
        flügel_zellen = np.where(nur_flügel == 0)
        if len(flügel_zellen[0]) == 0:
            return 0, 0
        x_min = np.min(flügel_zellen[0])
        x_max = np.max(flügel_zellen[0])

        F_auftrieb = 0.0
        for ix in range(x_min, x_max + 1):
            y_treffer = flügel_zellen[1][flügel_zellen[0] == ix]
            if len(y_treffer) < 2:
                continue
            oben = np.max(y_treffer) + 1
            unten = np.min(y_treffer) - 1

            

            if 0 <= oben < self.ny and 0 <= unten < self.ny:
                F_auftrieb += (self.p[ix, unten] - self.p[ix, oben]) * self.h
                
            

        sehne = (x_max - x_min) * self.h
        cL = ((2 * F_auftrieb) / (self.rho * (v_ein**2) * sehne))

        return F_auftrieb, cL
    
    def zeitschritt(self, v_ein):
        self.schwerkraft(g=0.0)
     
        self.advektion()
        self.inkompressibilitaet_lösen()
        self.extrapolieren()
        
        
        self.u[self.wand == 0] = 0.0
        self.v[self.wand == 0] = 0.0

        for i in range(self.nx):
            for j in range(self.ny):
                if self.wand[i, j] == 0:
                    if i + 1 < self.nx:          # nicht über den rechten Rand hinaus
                        self.u[i+1, j] = 0.0     # rechte Kante der Wandzelle
                    if j + 1 < self.ny:          # nicht über den oberen Rand hinaus
                        self.v[i, j+1] = 0.0     # obere Kante der Wandzelle


        # Einströmung links immer aufrechterhalten
        self.u[1, :] = v_ein
        


    def visualisieren(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        p_daten = self.p.T * -1
        p_fluid = np.ma.masked_where(self.wand.T == 0, p_daten)

        im1 = ax.imshow(p_fluid, origin='lower', cmap='RdBu')

        ax.imshow(
            np.ma.masked_where(self.wand.T != 0, np.ones_like(self.wand.T)),
            origin='lower',
            cmap='gray',
            vmin=0,
            vmax=2
            
        )

        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)

        

        plt.tight_layout()
        return fig




if __name__ == "__main__":
    sim = Windkanal(breite=160, hoehe=70, zellgrösse=0.2)
    
    eintrittgeschwindigkeit = 5.0
    sim.setze_wand(eintrittgeschwindigkeit)
    sim.u[:, :] = 0
    sim.u[sim.wand == 0] = 0.0  
    sim.v[sim.wand == 0] = 0.0 

    fluegel = naca_in_windkanal_einbauen(naca_code="2412", anstellwinkel=5, nx=sim.nx, ny=sim.ny, skalierung=60)
    sim.wand = np.minimum(sim.wand, fluegel)
    
    anzahl_schritte = 150
    print("Simulation startet")

    ramp_schritte = 50
    for schritt in range(anzahl_schritte):
        if schritt < ramp_schritte:
            v_aktuell = eintrittgeschwindigkeit * (schritt / ramp_schritte)
        else:
            v_aktuell = eintrittgeschwindigkeit
        
        sim.zeitschritt(v_aktuell)
        
        
    print("Simulation fertig")
    F, cL = sim.auftrieb_berechnen(eintrittgeschwindigkeit)
    print(f"Auftriebskraft: {F} N")
    print(f"Auftriebsbeiwert cL: {cL}")
    print(f"Erwarteter Wert (NACA 2412, 5°): ~0.5")

    sim.visualisieren()