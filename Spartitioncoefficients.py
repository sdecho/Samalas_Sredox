import numpy as np
import pandas as pd

class PartitionCoefficient:
    """
    P[MPa],Tkc[K], Initial or average melt composition: wtsio2[wt% SiO2], wttio2[wt% TiO2], wtal2o3 [wt% al2o3],
    wtfeo[wt% feo,FeO total as FeO], wtmno [wt% MnO], wtmgo[wt% MgO], wtcao[wt% CaO], wtna2o [wt% Na2O],
    wtk2o [wt% k2o], wtp2o5 [wt% p2o5], wth2o [wt% h2o]
    phih2o, phih2s, and phiso2 : fugacity coefficients of h2o, h2s and so2
    monte (==1) for option to return a random number within the estimated error of each kd
    logfo2 of fmq buffer, can be calculated with pressure and temperature following Frost(1991)
    """

    def __init__(self, composition):
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

        # Normalize wt % of oxides
        oxide_tot = wtsio2 + wttio2 + wtal2o3 + wtfeo + wtmno + wtmgo + wtcao + wtna2o + wtk2o + wtp2o5
        oxide_tot2 = wtsio2 + wtal2o3 + wtfeo + wtmgo + wtcao + wtna2o + wtk2o
        wtsio2 = wtsio2 / oxide_tot * 100
        wttio2 = wttio2 / oxide_tot * 100
        wtal2o3 = wtal2o3 / oxide_tot * 100
        wtfeo = wtfeo / oxide_tot * 100
        wtfeo2 = wtfeo/ oxide_tot2 *100
        wtmno = wtmno / oxide_tot * 100
        wtmgo = wtmgo / oxide_tot * 100
        wtcao = wtcao / oxide_tot * 100
        wtna2o = wtna2o / oxide_tot * 100
        wtk2o = wtk2o / oxide_tot * 100
        wtp2o5 = wtp2o5 / oxide_tot * 100

        # Convert wt % to mole fractions in a anhydrous oxide base
        nsi = wtsio2 / (28.086 + 15.999 * 2)  # moles of siO2
        nti = wttio2 / (47.867 + 15.999 * 2)  # moles of tio2
        nal = wtal2o3 / (26.982 * 2 + 15.999 * 3)  # moles of al2o3
        nfe = wtfeo / (55.845 + 15.999)  # moles of feo
        nmn = wtmno / (54.938 + 15.999)  # moles of mno
        nmg = wtmgo / (24.305 + 15.999)  # moles of mgo
        nca = wtcao / (40.078 + 15.999)  # moles of cao
        nna = wtna2o / (22.9898 * 2 + 15.999)  # moles of na2o
        nk = wtk2o / (39.098 * 2 + 15.999)  # moles of k2o
        nph = wtp2o5 / (30.973 * 2 + 15.999 * 5)  # moles of p2o5

        # Molar masses
        M_Na = 61.979
        M_K = 94.195
        M_Mg = 40.034
        M_Ca = 56.077
        M_Fe = 71.844
        M_Si = 60.084
        M_Al = 101.961


        # Numerator and Denominator
        N = 2 * (nna + nk + nmg + nca + nfe - nal)
        D = (2 * nsi + 3 * nal + nna + nk + nmg + nca + nfe)

        NBO_exp = N / D

        # Partial derivatives
        dN_dNa = 2 / M_Na
        dD_dNa = 1 / M_Na

        dN_dK = 2 / M_K
        dD_dK = 1 / M_K

        dN_dMg = 2 / M_Mg
        dD_dMg = 1 / M_Mg

        dN_dCa = 2 / M_Ca
        dD_dCa = 1 / M_Ca

        dN_dFe = 2 / M_Fe
        dD_dFe = 1 / M_Fe

        dN_dSi = 0
        dD_dSi = 2 / M_Si

        dN_dAl = 0
        dD_dAl = 3 / M_Al

        # General derivative formula
        def partial(x_sig, dN_dx, dD_dx):
            return ((dN_dx * D - N * dD_dx) / D ** 2) * x_sig

        # Total propagated uncertainty
        self.NBO_error = np.sqrt(
            partial(0.03*wtna2o, dN_dNa, dD_dNa) ** 2 +
            partial(0.02*wtk2o, dN_dK, dD_dK) ** 2 +
            partial(0.02*wtmgo, dN_dMg, dD_dMg) ** 2 +
            partial(0.02*wtcao, dN_dCa, dD_dCa) ** 2 +
            partial(0.02*wtfeo, dN_dFe, dD_dFe) ** 2 +
            partial(0.01*wtsio2, dN_dSi, dD_dSi) ** 2 +
            partial(0.01*wtal2o3, dN_dAl, dD_dAl) ** 2
        )

        self.nbo_o = 2*(nfe + nmg + nca + nna + nk - nal)/(2*nsi + 2*nti + nfe + nmg + nca + nna + nk+3*nal)  # nbo/o from Iacono_marziano
        self.feo_err = 0.02*wtfeo
        self.feo = wtfeo
        self.xfeo = nfe/(nfe + nmg + nca + nna + nk + nal+nsi)
        self.nbo_t = (2*(2*nsi + 2*nti + nfe + nmg + nca + nna + nk+3*nal)-4*(nsi+nti+2*nal))/(nsi+nti+2*nal) # nbo/t from Masotta 2015
        self.asi = nal/(nca+nna+nk)
        self.al_number = nal/(nsi+nti+nal)
        self.ca_number = nca/(nna+nk)
        # print(self.nbo_t,self.al_number, self.asi,self.ca_number)

    def kd_ox (self):
        rxn_ox = 9.2 -31.4*self.nbo_t-1.8*self.asi-29.5*self.al_number+4.2*self.ca_number
        return np.exp(rxn_ox)

    def kd_red(self, beta, cov_beta):
        X_new = [self.feo, self.nbo_o]
        X_err = [self.feo_err, self.NBO_error]

        y_pred, y_err = self.predict_with_uncertainty(X_new, X_err, beta, cov_beta)
        # Return scalars if it's a single point
        if y_pred.size == 1:

            y_err = (np.exp(float(y_pred[0])+float(y_err[0]))-np.exp(float(y_pred[0])-float(y_err[0])))/2
            y_pred = np.exp(float(y_pred[0]))
            return y_pred, y_err
        else:
            return np.exp(y_pred), (np.exp(y_pred+y_err)-np.exp(y_pred-y_err))/2

    def predict_with_uncertainty(self, X_new, X_err, beta, cov_beta):

        # Ensure input is at least 1D
        X_new = np.atleast_1d(X_new)
        X_err = np.atleast_1d(X_err)

        # Reshape to (n_features, n_points)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_err.ndim == 1:
            X_err = X_err.reshape(-1, 1)
        n_points = X_new.shape[1]
        y_pred = beta[0] + np.dot(beta[1:], X_new)


        total_var = np.zeros(n_points)

        for i in range(n_points):
            x_i = X_new[:, i]
            x_err_i = X_err[:, i]

            x_vec = np.concatenate(([1.0], x_i))
            model_var = x_vec @ cov_beta @ x_vec
            x_err_var = np.sum((beta[1:] * x_err_i) ** 2)

            total_var[i] = model_var + x_err_var

        y_err_total = np.sqrt(total_var)
        return y_pred, y_err_total