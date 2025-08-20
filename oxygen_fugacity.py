import numpy as np
import math
# constant for Kress and Carmicheal 1991
T0 = 1400  # in Celsius
TK0 = T0 + 273.15  # in kelvin
A = 0.196
B = 11492
C = -6.675
DWFEO = -1.828
DWAL2O3 = -2.243
DWCAO = 3.201
DWNA2O = 5.854
DWK2O = 6.215
E = -3.36
F = -0.000000701  # KPa - 1
G = -1.54e-10  # Pa - 1
H = 3.85e-17  # kPa - 2
R = 8.314 # J/mol

class OxygenFugacity:
    """P[MPa],Tkc[K], Initial or average melt composition: wtsio2[wt% SiO2],wttio2[wt% TiO2], wtal2o3 [wt% al2o3],
    wtfeo[wt% feo,FeO total as FeO], wtmno [wt% MnO], wtmgo[wt% MgO], wtcao[wt% CaO], wtna2o [wt% Na2O],
    wtk2o [wt% k2o], wtp2o5 [wt% p2o5]
    logfo2 of fmq buffer, can be calculated with pressure and temperature following Frost(1991)"""

    def __init__(self, P, Tkc, composition):
        # Normalize wt % of oxides
        wtsio2 = composition["SiO2"]
        wttio2 = composition["TiO2"]
        wtal2o3 = composition["Al2O3"]
        wtfeo = composition["FeOT"]
        wtmno = composition["MnO"]
        wtmgo = composition["MgO"]
        wtcao = composition["CaO"]
        wtna2o = composition["Na2O"]
        wtk2o = composition["K2O"]
        wtp2o5 = composition["P2O5"]

        oxide_tot = wtsio2 + wttio2 + wtal2o3 + wtfeo + wtmno + wtmgo + wtcao + wtna2o + wtk2o + wtp2o5
        wtsio2 = wtsio2 / oxide_tot * 100
        wttio2 = wttio2 / oxide_tot * 100
        wtal2o3 = wtal2o3 / oxide_tot * 100
        wtfeo = wtfeo / oxide_tot * 100
        wtmno = wtmno / oxide_tot * 100
        wtmgo = wtmgo / oxide_tot * 100
        wtcao = wtcao / oxide_tot * 100
        wtna2o = wtna2o / oxide_tot * 100
        wtk2o = wtk2o / oxide_tot * 100
        wtp2o5 = wtp2o5 / oxide_tot * 100

        # Convert wt % to mole fractions
        nsio2 = wtsio2 / (28.086 + 15.999 * 2)  # moles of siO2
        ntio2 = wttio2 / (47.867 + 15.999 * 2)  # moles of tio2
        nal2o3 = wtal2o3 / (26.982 * 2 + 15.999 * 3)  # moles of al2o
        nfeo = wtfeo / (55.845 + 15.999)  # moles of feo
        nmno = wtmno / (54.938 + 15.999)  # moles of mno
        nmgo = wtmgo / (24.305 + 15.999)  # moles of mgo
        ncao = wtcao / (40.078 + 15.999)  # moles of cao
        nna2o = wtna2o / (22.9898 * 2 + 15.999)  # moles of na2o
        nk2o = wtk2o / (39.098 * 2 + 15.999)  # moles of k2o
        np2o5 = wtp2o5 / (30.973 * 2 + 15.999 * 5)  # moles of p2o5
        self.ntot = (nsio2 + ntio2 + nal2o3 + nfeo + nmno + nmgo + ncao + nna2o + nk2o + np2o5)  # totalmole
        self.nctot = (nsio2 + ntio2 + 2*nal2o3 + nfeo + nmno + nmgo + ncao + 2*nna2o + 2*nk2o + 2*np2o5) # total cation mole

        self.xsio2 = nsio2 / self.ntot
        self.xtio2 = ntio2 / self.ntot
        self.xal2o3 = nal2o3 / self.ntot
        self.xfeo = nfeo / self.ntot
        self.xmno = nmno / self.ntot
        self.xmgo = nmgo / self.ntot
        self.xcao = ncao / self.ntot
        self.xna2o = nna2o / self.ntot
        self.xk2o = nk2o / self.ntot
        self.xp2o5 = np2o5 / self.ntot

        # cation fraction
        self.xmg = nmgo/self.nctot
        self.xca = ncao/self.nctot
        self.xna = 2*nna2o/ self.nctot
        self.xal = 2*nal2o3/self.nctot
        self.xk = 2*nk2o/self.nctot
        self.xp = 2*np2o5 / self.nctot
        self.xfe = nfeo / self.nctot

        self.Pp = P * 1000000  # MPa to Pa
        self.Pb = P * 10  # MPa to bar
        self.Pg = P/1000 # MPa to GPa
        self.Tkc = Tkc
        # to calculate Fe3+-fO2 using KC1991 model
        self.con = B / self.Tkc + C + (
                DWAL2O3 * self.xal2o3 + DWCAO * self.xcao + DWNA2O * self.xna2o + DWK2O * self.xk2o + DWFEO * self.xfeo) \
                   + E * (1 - TK0 / self.Tkc - np.log(self.Tkc / TK0)) + F * self.Pp / self.Tkc + \
                   G * (self.Tkc - TK0) * self.Pp / self.Tkc + H * (self.Pp ** 2) / self.Tkc
        # to calculate Fe3+-fO2 using O'Neil et al. 2006
        self.con_ON = -28144/self.Tkc + 13.95 + (3905*self.xmg -13359*self.xca -14858*self.xna-9805*self.xk + 10906*self.xal +110971*self.xp)/self.Tkc + \
                      (33122/self.Tkc-5.24)*((1+0.241*self.Pg)**0.75-1)-(39156/self.Tkc-6.17)*((1+0.132*self.Pg)**0.75-1)
        self.con_Ar = (-2248*self.xmg+7690*self.xca+8553*self.xna + 5644*self.xk-6278*self.xal)/self.Tkc

    def fe_ratio(self, o2):
        # --------------Kress & Carmichael 1991 parameter values - -------------------
        fo2_ln = np.log(np.power(10, o2))
        rfe_ln = A * fo2_ln + self.con
        rfe = 2 * np.exp(rfe_ln) / (1 + 2 * np.exp(rfe_ln))
        return rfe

    def fo2(self, rfe, choice):
        xfe2 = self.xfe * (1-rfe)
        xfe3 = self.xfe * rfe

        if choice == 1: #KC 1991 model
            rfe = (rfe / 2) / (1 - rfe)  # transfer ratio, fe3 / fet to fe3 / fe2;
            o2 = (np.log(rfe) - self.con) / A
            o2 = np.log10(np.exp(o2))
        elif choice == 2: #ONeil2006 model
            rfe = rfe/(1-rfe)
            o2 = 4*np.log10(rfe)+ self.con_ON-11952*(xfe2-xfe3)/self.Tkc
        else: # Armstrong et al. (2019) model
            rfe = rfe / (1 - rfe)
            dG = -16201/self.Tkc + 8.031 #dG/RT
            fe2_kapa0 = 37
            fe2_kapap = 8
            fe2_kapapp = -fe2_kapap/fe2_kapa0
            fe3_kapa0 = 12.6
            fe3_kapap = 1.3
            fe3_kapapp = -fe3_kapap / fe3_kapa0
            fe2_v0 = 13650 + 2.92 * (self.Tkc - 1673)
            fe3_v0 = 21070 + 4.54*(self.Tkc-1673)

            fe2_a = (1+fe2_kapap)/(1+fe2_kapap+fe2_kapa0*fe2_kapapp)
            fe2_b = fe2_kapap/fe2_kapa0 - fe2_kapapp/(1+fe2_kapap)
            fe2_c = (1+fe2_kapap+fe2_kapa0*fe2_kapapp)/(fe2_kapap**2+fe2_kapap-fe2_kapapp*fe2_kapa0)
            fe2_v = self.Pg*fe2_v0*(1-fe2_a+(fe2_a*(1-(1+fe2_b*self.Pg)**(1-fe2_c))/(fe2_b*(fe2_c-1)*self.Pg)))

            fe3_a = (1 + fe3_kapap) / (1 + fe3_kapap + fe3_kapa0 * fe3_kapapp)
            fe3_b = fe3_kapap / fe3_kapa0 - fe3_kapapp / (1 + fe3_kapap)
            fe3_c = (1 + fe3_kapap + fe3_kapa0 * fe3_kapapp) / (fe3_kapap ** 2 + fe3_kapap - fe3_kapapp * fe3_kapa0)
            fe3_v = self.Pg * fe3_v0 * (1 - fe3_a + (
                        fe3_a * (1 - (1 + fe3_b * self.Pg) ** (1 - fe3_c)) / (fe3_b * (fe3_c - 1) * self.Pg)))
            dV = (fe3_v-fe2_v)/(self.Tkc*R) #dV/RT

            o2 = 4*(np.log(rfe)+dG+dV-self.con_Ar-6880*(xfe2-xfe3)/self.Tkc)
            print(o2)
            o2 = np.log10(np.exp(o2))
        return o2

    def fmq(self):
        o2_fmq = -25096.3 / self.Tkc + 8.735 + .110 * (self.Pb - 1) / self.Tkc
        return o2_fmq

    def gas_quilibrium(self, fo2, fh2o, phiso2, phih2s):
        logK = 4.1245 - 27110/self.Tkc
        ratio = 10 ** (1.5*fo2 - logK - np.log10(fh2o)) * phih2s / phiso2
        ratio = ratio / (ratio +1)
        return ratio

    def OandM(self, rfe, o2):
        """
        rfe: xFe3+/xFeT
        o2: log10fO2
        """
        xferrous = self.xfeo*(1-rfe)
        c_sulfide = 8.77 - 23590 / self.Tkc + (1673 / self.Tkc) * (
                    6.7 * (self.xna2o + self.xk2o) + 4.9 * self.xmgo + 8.1 * self.xcao + 8.9 * (self.xfeo + self.xmno) + 5 * self.xtio2 + 1.8 * self.xal2o3
                    - 22.2 * self.xtio2 * (self.xfeo + self.xmno) + 7.2 * ((self.xfeo + self.xmno) * self.xsio2)) - 2.06 * math.erf(-7.2 * (self.xfeo + self.xmno))
        c_sulfate = (-8.02) + (
                    21100 + 44000 * self.xna2o + 18700 * self.xmgo + 4300 * self.xal2o3 + 35600 * self.xcao + 44200 * self.xk2o + 16500 * xferrous + 12600 * self.xmno) / self.Tkc
        lnk = (-55921) / self.Tkc + 25.07 - 0.6465 * np.log(self.Tkc)  # SO3/S
        lnrs = (c_sulfate - lnk - c_sulfide) + 2 * np.log(10) * o2
        rs = 1 - 1 / (1 + np.exp(lnrs))

        return rs


