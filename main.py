import pandas as pd
import numpy as np
from oxygen_fugacity import OxygenFugacity
from fugacity import Fugacity
import matplotlib.pyplot as plt
from matplotlib import rc, cm
import matplotlib.patches as patches
import matplotlib
from matplotlib.font_manager import findfont, FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import PySulfSat as ss
from Spartitioncoefficients import PartitionCoefficient
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
import uncertainties as unc
from statistics import mean

df = pd.read_csv("meltcomp_corr.csv")
df_exp = pd.read_csv("experiments_list.csv")
df_plot = pd.read_csv("Samalas_plot.csv")
df_melts = pd.read_csv("MagmaSat_model.csv")
tk = 950+273


#################calculate SCSS of all MIs#######################
df_out2=ss.import_data('Samalas_SCSS_input.xlsx', sheet_name='Glass_input',
                       suffix="_Liq")
Smythe_CalcSulf=ss.calculate_S2017_SCSS(df=df_out2, T_K=950+273,
P_kbar=1.2, Fe_FeNiCu_Sulf=0.65, Fe3Fet_Liq=0.25)
# print(Smythe_CalcSulf.mean())
ZT2019_SCAS=ss.calculate_ZT2019_SCAS(df=df_out2, T_K=950+273)
df_ST_FixedS6=ss.calculate_S_Total_SCSS_SCAS(
    SCSS=Smythe_CalcSulf['SCSS2_ppm_ideal_Smythe2017'],
    SCAS=ZT2019_SCAS['SCAS6_ppm'], S6St_Liq=df_out2["sulfate_Liq"])
df_ST_FixedS6.to_csv("SCSS_Sulfate.csv")
#Smythe_CalcSulf.to_csv("SCSS_Smythe.csv")
scss = 50 # ppm

#calculate kdox and kdred for Samalas MIs
kd_ox_mi =[]
kd_red_mi = []
kd_red_err = []
kd_mi = []
kd_mi_err = []
dfmq_AM_max = []
dfmq_ON_max = []
dfmq_AM_min = []
dfmq_ON_min = []
fo2_AM_max = []
fo2_ON_max = []
fo2_AM_min = []
fo2_ON_min = []

dfmq_ON_avg =[]
dfmq_AM =[]
dfmq_AM_pos = []
dfmq_AM_neg = []

fh2o = []
vapor_ratio = []
phiSO2 = []
phiH2S = []
phiH2O = []
rs_OM_max = []
rs_OM_min = []
mi_feo= []
mi_nbo = []



# Load beta (parameter values)
df_beta = pd.read_csv('beta_parameters.csv')
beta = df_beta['value'].values  # 1D numpy array

# Load covariance matrix
df_cov = pd.read_csv('beta_covariance.csv', index_col=0)
cov_beta = df_cov.values  # 2D numpy array

for j in range(0, 67):
    composition_mi = {
        "SiO2": df_plot["SiO2"][j],
        "TiO2": df_plot["TiO2"][j],
        "Al2O3": df_plot["Al2O3"][j],
        "FeOT": df_plot["FeO"][j],
        "MnO": df_plot["MnO"][j],
        "MgO": df_plot["MgO"][j],
        "CaO": df_plot["CaO"][j],
        "Na2O": df_plot["Na2O"][j],
        "K2O": df_plot["K2O"][j],
        "P2O5": df_plot["P2O5"][j],
    }
    kd = PartitionCoefficient(composition=composition_mi)
    kd1, kd1_err = kd.kd_red(beta=beta, cov_beta=cov_beta)
    kd2 = kd.kd_ox()
    a = unc.ufloat(kd1, kd1_err)
    b = unc.ufloat(kd2, 0.2*kd2)
    mi_feo.append(kd.feo)
    mi_nbo.append(kd.nbo_o)


    if df_plot["S6+"][j]>0 and df_plot["S6+"][j]<0.9:
        sulf = df_plot["S6+"][j]
        c = unc.ufloat(sulf, 0.1)
        kd_u = a*(1-c)+c*b
        kd_mi.append(kd_u.nominal_value)
        kd_mi_err.append(kd_u.std_dev)
    elif df_plot["S6+"][j]==1:
        kd_u = a * (1 - df_plot["S6+"][j]) + df_plot["S6+"][j] * b
        kd_mi.append(kd_u.nominal_value)
        kd_mi_err.append(kd_u.std_dev)
    kd_ox_mi.append(kd2)
    kd_red_mi.append(kd1)
    kd_red_err.append(kd1_err)

    P = df_plot["pressure"][j]
    Tk = df_plot["temperature"][j] + 273.15
    ferric_ratio_max = (df_plot["ferric"][j] +df_plot["per"][j]) / 100
    ferric_ratio_min = (df_plot["ferric"][j] - df_plot["ner"][j]) / 100
    fo2_0 = OxygenFugacity(P, Tk, composition_mi)
    max_fo2 = fo2_0.fo2(ferric_ratio_max, 3) - fo2_0.fmq()
    dfmq_AM_max.append(max_fo2)
    oxygen_AM_max = fo2_0.fo2(ferric_ratio_max, 3)
    fo2_AM_max.append(oxygen_AM_max)
    min_fo2 = fo2_0.fo2(ferric_ratio_min, 3) - fo2_0.fmq()
    dfmq_AM_min.append(min_fo2)
    oxygen_AM_min = fo2_0.fo2(ferric_ratio_min, 3)
    fo2_AM_min.append(oxygen_AM_min)

    avg = (max_fo2 + min_fo2)/2
    dfmq_AM.append(avg)
    pos = max_fo2-avg
    neg = avg-min_fo2
    dfmq_AM_pos.append(pos)
    dfmq_AM_neg.append(neg)

    dfmq_ON_max.append(fo2_0.fo2(ferric_ratio_max, 2) - fo2_0.fmq())
    oxygen_ON_max = fo2_0.fo2(ferric_ratio_max, 2)
    fo2_ON_max.append(oxygen_ON_max)
    sulfate_OM_max = fo2_0.OandM(rfe=oxygen_ON_max, o2= oxygen_ON_max)
    rs_OM_max.append(sulfate_OM_max)

    dfmq_ON_min.append(fo2_0.fo2(ferric_ratio_min, 2) - fo2_0.fmq())
    oxygen_ON_min = fo2_0.fo2(ferric_ratio_min, 2)
    fo2_ON_min.append(oxygen_ON_min)
    sulfate_OM_min = fo2_0.OandM(rfe=oxygen_ON_min, o2=oxygen_ON_min)
    rs_OM_min.append(sulfate_OM_min)

    avg = (dfmq_ON_min[j]+dfmq_ON_max[j])/2
    dfmq_ON_avg.append(avg)


# print(dfmq_ON, dfmq_AM)
# plt.figure(1)
# plt.plot(df_plot["fo2"][0:23],fo2_ON_min[0:23],"o")
# plt.plot(df_plot["fo2"][0:23], fo2_AM_min[0:23], "d")


ACNK_exp = (df_exp["Al2O3"][0:26]/101.96)/(df_exp["CaO"][0:26]/56.077+df_exp["Na2O"][0:26]/61.98+df_exp["K2O"][0:26]/94.2)
NKA_exp = (df_exp["Na2O"][0:26]/61.98+df_exp["K2O"][0:26]/94.2)/(df_exp["Al2O3"][0:26]/101.96)
ACNK_samalas = (df_plot["Al2O3"][0:67]/101.96)/(df_plot["CaO"][0:67]/56.077+df_plot["Na2O"][0:67]/61.98+df_plot["K2O"][0:67]/94.2)
NKA_samalas = (df_plot["Na2O"][0:67]/61.98+df_plot["K2O"][0:67]/94.2)/(df_plot["Al2O3"][0:67]/101.96)
NBO_exp = 2 *(df_exp["Na2O"][0:26]/61.979+df_exp["K2O"][0:26]/94.195+df_exp["MgO"][0:26]/40.034+df_exp["CaO"][0:26]/56.077+df_exp["FeO"][0:26]/71.844-df_exp["Al2O3"][0:26]/101.961)\
          /(2*df_exp["SiO2"][0:26]/60.084+3*df_exp["Al2O3"][0:26]/101.961+df_exp["Na2O"][0:26]/61.979+df_exp["K2O"][0:26]/94.195+df_exp["MgO"][0:26]/40.034+df_exp["CaO"][0:26]/56.077+df_exp["FeO"][0:26]/71.844)
NBO_samalas = mi_nbo
feoN_samalas = mi_feo
feoN_exp = 100*df_exp["FeO"][0:26]/(df_exp["Al2O3"][0:26]+df_exp["FeO"][0:26]+df_exp["MgO"][0:26]+df_exp["CaO"][0:26]+df_exp["Na2O"][0:26]+df_exp["K2O"][0:26]+df_exp["SiO2"][0:26])

# Combine into DataFrame
df_MI_calc = pd.DataFrame({
    "ACNK_samalas": ACNK_samalas,
    "NKA_samalas": NKA_samalas,
    "NBO_samalas": NBO_samalas,
    "FeO_samalas": feoN_samalas
})

# Save to CSV
df_MI_calc.to_csv("MI_calc_samalas.csv", index=False)


# # Create the plot
# figs8, axes8 = plt.subplots(1, 2, figsize=(14, 6))
#
# # === Panel (a): FeO_norm vs. NBO ===
# axes8[0].scatter(feoN_exp, NBO_exp, color="blue", label="Experimental")
# axes8[0].scatter(feoN_samalas, NBO_samalas, color="green", label="Samalas")
#
# axes8[0].set_xlabel("FeO normalized (%)", fontsize=12)
# axes8[0].set_ylabel("NBO/T", fontsize=12)
# axes8[0].set_title("Panel (a): FeO$_{norm}$ vs. NBO/T", fontsize=13)
# axes8[0].legend()
# axes8[0].grid(True)
#
#
# axes8[1].scatter(ACNK_exp, NKA_exp, label='Experimental', color='blue')
# axes8[1].scatter(ACNK_samalas, NKA_samalas, label='Samalas', color='green')
#
# # Plot x=1 and y=1 reference lines
# axes8[1].axvline(x=1, color='gray', linestyle='--')
# axes8[1].axhline(y=1, color='gray', linestyle='--')
#
# # Annotate the geochemical fields
# axes8[1].text(0.6, 1.3, 'Peralkaline', fontsize=12, color='red')
# axes8[1].text(0.6, 0.6, 'Metaluminous', fontsize=12, color='red')
# axes8[1].text(1.1, 0.6, 'Peraluminous', fontsize=12, color='red')
#
# # Axis labels with proper chemical notation
# axes8[1].set_xlabel(r'Mol Al$_2$O$_3$ / (Na$_2$O + K$_2$O + CaO)', fontsize=12)
# axes8[1].set_ylabel(r'Mol (Na$_2$O + K$_2$O) / Al$_2$O$_3$', fontsize=12)
#
# # Add legend and grid
# axes8[1].legend()
# axes8[1].set_title('ACNK vs NKA Classification Diagram')
# axes8[1].grid(True)
# plt.tight_layout()
#


high_mgo = df.loc[df_plot["MgO"].iloc[0:len(kd_mi)]>0.85].index.to_list()

# Collect kd values and errors for high MgO
kd_vals = [kd_mi[mgo] for mgo in high_mgo]
kd_errs = [kd_mi_err[mgo] for mgo in high_mgo]

# Convert to ufloats
kd_ufloats = [unc.ufloat(val, err) for val, err in zip(kd_vals, kd_errs)]

# Extract nominal values and std_devs
vals = np.array([k.nominal_value for k in kd_ufloats])
errs = np.array([k.std_dev for k in kd_ufloats])

# Unweighted mean
mean_high = np.mean(vals)
# Propagated standard error of the mean (assuming independent errors)
mean_error_high = np.sqrt(np.sum(errs**2)) / len(vals)

# Optional: wrap as ufloat
kd_highmgo_full = unc.ufloat(mean_high, mean_error_high)
# kd_low=[]
# for mgo in low_mgo:
#     kd_low.append(kd_mi[mgo])
kd_highmgo = kd_highmgo_full.nominal_value
kd_highmgo_1sig = kd_highmgo_full.std_dev



low_mgo = df.loc[df_plot["MgO"].iloc[0:len(kd_mi)]<0.85].index.to_list()

# Collect kd values and errors for low MgO
kd_vals_low = [kd_mi[mgo] for mgo in low_mgo]
kd_errs_low = [kd_mi_err[mgo] for mgo in low_mgo]

# Convert to ufloats
kd_ufloats_low = [unc.ufloat(val, err) for val, err in zip(kd_vals_low, kd_errs_low)]

# Extract nominal values and std_devs
vals_low = np.array([k.nominal_value for k in kd_ufloats_low])
errs_low = np.array([k.std_dev for k in kd_ufloats_low])

# # Compute weighted average and its uncertainty
# weights_low = 1 / errs_low**2
# weighted_mean_low = np.sum(weights_low * vals_low) / np.sum(weights_low)
# weighted_error_low = np.sqrt(1 / np.sum(weights_low))

# Unweighted mean
mean_low = np.mean(vals_low)
# Propagated standard error of the mean (assuming independent errors)
mean_error_low = np.sqrt(np.sum(errs_low**2)) / len(vals_low)

# Wrap as ufloat and extract components
kd_lowmgo_full = unc.ufloat(mean_low, mean_error_low)
kd_lowmgo = kd_lowmgo_full.nominal_value
kd_lowmgo_1sig = kd_lowmgo_full.std_dev

# kd_lowmgo = weighted_mean_low
# kd_lowmgo_1sig = weighted_error_low
print(kd_lowmgo,kd_highmgo_1sig, kd_highmgo, kd_lowmgo_1sig )

fig1, ax1 = plt.subplots(figsize=(6, 4))  # create figure and axes

ax1.errorbar(
    x=df_plot["MgO"][:len(kd_mi)],
    y=kd_mi,
    yerr=kd_mi_err,
    fmt='o',                # marker style
    ecolor='gray',          # error bar color
    capsize=3,              # error bar cap size
    label='MI Kd values'
)

ax1.set_xlabel("MgO (wt%)")
ax1.set_ylabel("Kd")
ax1.set_title("Partition Coefficients vs MgO")
ax1.legend()
plt.tight_layout()




# Based on the kds for MIs, fit MgO vs. Kd_ox and MgO vs. Kd_red, which will be used for model S degassing
# X = MgO from MIs
X = np.array(df_plot["MgO"][0:67])

# Split the MgO and Kd_ox for MIs into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, kd_ox_mi, test_size=0.2, random_state=42)

# Evaluate the best fit among linear, polynomial and power-law for MgO vs. Kd_ox
# Define the power-law function
def power_law(x, a, b):
    return a * x**b

# Define the polynomial function
def polynomial(x, *coefficients):
    return np.polyval(coefficients, x)

# Linear regression model
linear_model = LinearRegression()

# Polynomial regression model (degree=2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Power-law regression model
def power_law_model(x, a, b):
    return a * x**b

# Fit the power-law model using curve_fit

params, covariance = curve_fit(power_law_model, x_train, y_train)

# Fit the linear and polynomial models
x_train = x_train.reshape(-1, 1)
linear_model.fit(x_train, y_train)
poly_model.fit(x_train, y_train)

# Evaluate models on the test set
linear_predictions = linear_model.predict(x_test.reshape(-1, 1))
poly_predictions = poly_model.predict(x_test.reshape(-1, 1))
power_law_predictions = power_law_model(x_test, *params)

# Calculate mean squared errors
linear_mse = mean_squared_error(y_test, linear_predictions)
poly_mse = mean_squared_error(y_test, poly_predictions)
power_law_mse = mean_squared_error(y_test, power_law_predictions)

# Print mean squared errors
print(f'Linear Regression for kd_ox MSE: {linear_mse}')
print(f'Polynomial Regression for kd_ox MSE: {poly_mse}')
print(f'Power-law Regression for kd_ox MSE: {power_law_mse}')

# Choose the best model based on the MSE
mse_results = {
                "Linear": linear_mse,
                "Polynomial": poly_mse,
                "Power-law": power_law_mse,
}
best_model = min([("Linear", linear_mse), ("Polynomial", poly_mse), ("Power-law", power_law_mse)], key=lambda x: x[1])

print(f'The best model for kd_ox is {best_model[0]} with MSE: {best_model[1]}')

# Use all data to do a final fit with the best model and apply to MgO to calculate Kd_Ox
MgO = np.arange(0.4, 1.3, 0.005) # in wt.%
MgO = np.flip(MgO)

if best_model[0] == "Linear":
    final_model_ox = LinearRegression()
    final_model_ox.fit(X.reshape(-1, 1), kd_ox_mi)
    kd_ox_pred = final_model_ox.predict(MgO.reshape(-1,1))
elif best_model[0] == "Polynomial":
    final_model_ox = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    final_model_ox.fit(X.reshape(-1, 1), kd_ox_mi)
    kd_ox_pred = final_model_ox.predict(MgO.reshape(-1, 1))
elif best_model[0] == "Power-law":
    def final_power_law_model(x, a, b):
        return a * x**b
    params, _ = curve_fit(final_power_law_model, X, kd_ox_mi)
    final_model_ox = lambda x: final_power_law_model(x, *params)
    kd_ox_pred = final_power_law_model(MgO, *params)

# Evaluate the best fit among linear, polynomial and power-law for MgO vs. Kd_red
# Split the MgO and Kd_red for MIs  into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, kd_red_mi, test_size=0.2, random_state=42)
params, covariance = curve_fit(power_law_model, x_train, y_train)

# Fit the linear and polynomial models
x_train = x_train.reshape(-1, 1)
linear_model.fit(x_train, y_train)
poly_model.fit(x_train, y_train)

# Evaluate models on the test set
linear_predictions = linear_model.predict(x_test.reshape(-1, 1))
poly_predictions = poly_model.predict(x_test.reshape(-1, 1))
power_law_predictions = power_law_model(x_test, *params)

# Calculate mean squared errors
linear_mse = mean_squared_error(y_test, linear_predictions)
poly_mse = mean_squared_error(y_test, poly_predictions)
power_law_mse = mean_squared_error(y_test, power_law_predictions)

# Print mean squared errors
print(f'Linear Regression for kd_red MSE: {linear_mse}')
print(f'Polynomial Regression for kd_red MSE: {poly_mse}')
print(f'Power-law Regression for kd_red MSE: {power_law_mse}')

# Choose the best model based on the MSE
best_model = min([("Linear", linear_mse), ("Polynomial", poly_mse), ("Power-law", power_law_mse)], key=lambda x: x[1])

print(f'The best model for kd_red is {best_model[0]} with MSE: {best_model[1]}')
if best_model[0] == "Linear":
    final_model_red = LinearRegression()
    final_model_red.fit(X.reshape(-1, 1), kd_red_mi)
    kd_red_pred = final_model_red.predict(MgO.reshape(-1,1))
elif best_model[0] == "Polynomial":
    final_model_red = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    final_model_red.fit(X.reshape(-1, 1), kd_red_mi)
    kd_red_pred = final_model_red.predict(MgO.reshape(-1, 1))
elif best_model[0] == "Power-law":
    def final_power_law_model(x, a, b):
        return a * x**b
    params, _ = curve_fit(final_power_law_model, X, kd_red_mi)
    final_model_red = lambda x: final_power_law_model(x, *params)
    kd_red_pred = final_power_law_model(MgO, *params)


# isotope fractionation factors
aso2_h2s = -0.42*(1000/tk)**3+4.367*(1000/tk)**2-0.105*(1000/tk)-0.41
as6_h2s = 6.4*1000000/tk**2
ah2s_s2 = 1.1*(1000/tk)**2-0.19
as6_s2 =7.4*1000000/tk**2-0.19
aso2_s6 = aso2_h2s-as6_h2s
aso2_s2 = aso2_s6+as6_s2

rs_f = 0.1

##################crystallization framework#######################

th_ini = -10.478 * MgO[0] + 21.963
H2O_sol = 4 # in wt.%
H2O_ini = 4 # in wt.%
H2O_ini_err = 0.3
H2O_sol_err = 0.3

S_ini = 700 #ppm
sulfate_ini = -0.9707 * MgO[0] + 1.5908


df_mc = pd.DataFrame({'MgO': MgO})
df_mc_red = pd.DataFrame({'MgO': MgO})
df_mc_ox = pd.DataFrame({'MgO': MgO})
df_mc_kdcombined = pd.DataFrame({'MgO': MgO})
df_mc_eff = pd.DataFrame({'MgO': MgO})
df_mc_eff_red = pd.DataFrame({'MgO': MgO})
df_mc_eff_ox = pd.DataFrame({'MgO': MgO})

df_mc_d34s_closed = pd.DataFrame({'MgO': MgO})
df_mc_d34s_open = pd.DataFrame({'MgO': MgO})
n = 10000 # Monte Carlo simulation for uncertainty estimate
for j in range (0, n):
    melt_fraction = [1]
    fluid_fraction = [0]
    crystal_fraction = [0]
    S_ini_sample = np.random.normal(loc=S_ini, scale=200)
    S_melt = [S_ini_sample]
    S_melt_ox = [S_ini_sample]
    S_melt_red = [S_ini_sample]

    degas_eff = [0]
    degas_ox_eff = [0]
    degas_red_eff = [0]
    sf_acc = [0]
    sf_ox_acc = [0]
    sf_red_acc = [0]
    sulfide_red = [0]

    d34s_ini = 8
    d34s_m_closed = [d34s_ini]
    d34s_m_open = [d34s_ini]
    kd_combined = [0]

    H2O_ini_sample = np.random.normal(loc=H2O_ini, scale=H2O_ini_err)
    H2O_sol_sample = np.random.normal(loc=H2O_sol, scale=H2O_sol_err)
    scss = np.random.normal(loc = 50, scale = 10)

    for i in range(1, len(MgO)):
        th = -10.478 * MgO[i] + 21.963
        melt = th_ini / th
        melt_fraction.append(melt)
        ins_fm = melt_fraction[i - 1] - melt_fraction[i]

        fluid = (melt_fraction[0] * H2O_ini_sample - melt_fraction[i] * H2O_sol_sample) * 0.01

        crystal = 1 - melt - fluid
        fluid_fraction.append(fluid)
        ins_ff = fluid_fraction[i] - fluid_fraction[i - 1]

        kd_ox = np.random.normal(loc=kd_ox_pred[i], scale=0.2 * kd_ox_pred[i])
        kd_red = np.random.normal(loc=kd_red_pred[i], scale=0.2 * kd_red_pred[i])

        if -0.9707 * MgO[i] + 1.5908 < 1:
            sulfate = -0.9707 * MgO[i] + 1.5908
            sulfate = np.random.normal(loc=sulfate, scale= 0.1*sulfate)
        else:
            sulfate = 1
            # mixed Kd

        #kd_comb = (kd_red * (1 - sulfate) + kd_ox * sulfate) * 1

        if MgO[i] >0.85:
            kd_comb= np.random.normal(loc=kd_highmgo, scale=kd_highmgo_1sig)
        else:
            kd_comb = np.random.normal(loc=kd_lowmgo, scale=kd_lowmgo_1sig)

        kd_combined.append(kd_comb)

        ms = (S_melt[i - 1] * melt_fraction[i - 1]) / (melt_fraction[i] + kd_comb * ins_ff)

        S_melt.append(ms)

        eff = 100 * (S_ini * melt_fraction[0] - ms * melt) / S_ini * melt_fraction[0]
        degas_eff.append(eff)
        fs_ins = ms * kd_comb / 10000
        fs_accum = sf_acc[i - 1] + fs_ins / 100
        sf_acc.append(fs_accum)

        # isotope fractionation
        agas_melt = rs_f * sulfate * aso2_s6 + rs_f * (1 - sulfate) * aso2_s2 - (1 - rs_f) * sulfate * as6_h2s + (
                    1 - rs_f) * (1 - sulfate) * ah2s_s2
        d34s_closed = d34s_ini + (1 - ms / S_ini) * agas_melt
        d34s_m_closed.append(d34s_closed)
        d34s_open = d34s_m_open[i - 1] - (1 - S_melt[i] / S_melt[i - 1]) * agas_melt
        d34s_m_open.append(d34s_open)

        # kd ox
        ms_ox = (S_melt_ox[i - 1] * melt_fraction[i - 1]) / (melt_fraction[i] + kd_ox * ins_ff)
        S_melt_ox.append(ms_ox)
        eff_ox = 100 * (S_ini * melt_fraction[0] - ms_ox * melt) / S_ini * melt_fraction[0]
        degas_ox_eff.append(eff_ox)
        fs_ox_ins = ms_ox * kd_ox/10000
        fs_ox_accum = sf_ox_acc[i - 1] + fs_ox_ins / 100
        sf_ox_acc.append(fs_ox_accum)

        # kd red
        ms_red = (S_melt_ox[i - 1] * melt_fraction[i - 1]) / (melt_fraction[i] + kd_red * ins_ff)
        if ms_red > scss:
            ms_red = scss
        S_melt_red.append(ms_red)

        # if MgO[i] >0.5 and MgO[i] <0.55:
        #     print(fluid, crystal)

        fs_red_ins = ms_red * kd_red / 10000

        if ins_ff>0:
            fs_red_accum = sf_red_acc[i - 1] + fs_red_ins*ins_ff / 100
        else:
            fs_red_accum = sf_red_acc[i - 1]
        sf_red_acc.append(fs_red_accum)
        eff_red = 100*fs_red_accum / (S_ini / 1000000)

        degas_red_eff.append(eff_red)
        mf_sulfide = (S_ini - ms_red * melt - fs_red_accum * 1000000) / S_ini

        sulfide_red.append(mf_sulfide)


    df_mc.insert(j + 1, j, S_melt)
    df_mc_ox.insert(j + 1, j, S_melt_ox)
    df_mc_red.insert(j+1, j, S_melt_red)
    df_mc_eff.insert(j+1, j, degas_eff)
    df_mc_eff_ox.insert(j+1, j, degas_ox_eff)
    df_mc_eff_red.insert(j+1, j, degas_red_eff)
    df_mc_d34s_closed.insert(j+1, j, d34s_m_closed)
    df_mc_d34s_open.insert(j+1, j,d34s_m_open)
    df_mc_kdcombined.insert(j+1, j, kd_combined)

def summarize_mc(df):
    """
    Computes mean and 75% confidence interval bounds (12.5%–87.5%) for each row in a Monte Carlo DataFrame.
    """
    means = df.mean(axis=1)
    lower = df.apply(lambda row: np.percentile(row, 12.5), axis=1)
    upper = df.apply(lambda row: np.percentile(row, 87.5), axis=1)
    errors = (upper - lower) / 2  # Half-width of 75% CI
    return pd.DataFrame({'mean': means, '75CI_err': errors, 'low_12.5%': lower, 'high_87.5%': upper})

summary_mc = summarize_mc(df_mc)
summary_mc_ox = summarize_mc(df_mc_ox)
summary_mc_red = summarize_mc(df_mc_red)
summary_eff = summarize_mc(df_mc_eff)
summary_eff_ox = summarize_mc(df_mc_eff_ox)
summary_eff_red = summarize_mc(df_mc_eff_red)
summary_d34s_closed = summarize_mc(df_mc_d34s_closed)
summary_d34s_open = summarize_mc(df_mc_d34s_open)
summary_kdcombined = summarize_mc(df_mc_kdcombined)

mean_s_melt = summary_mc["mean"]
low_s_melt = summary_mc["low_12.5%"]
high_s_melt = summary_mc["high_87.5%"]

mean_s_ox = summary_mc_ox["mean"]
low_s_ox = summary_mc_ox["low_12.5%"]
high_s_ox = summary_mc_ox["high_87.5%"]

mean_s_red = summary_mc_red["mean"]
low_s_red = summary_mc_red["low_12.5%"]
high_s_red = summary_mc_red["high_87.5%"]

mean_eff_ox = summary_eff_ox["mean"]
low_eff_ox = summary_eff_ox["low_12.5%"]
high_eff_ox = summary_eff_ox["high_87.5%"]

mean_eff_red = summary_eff_red["mean"]
low_eff_red = summary_eff_red["low_12.5%"]
high_eff_red = summary_eff_red["high_87.5%"]

mean_eff = summary_eff["mean"]
low_eff = summary_eff["low_12.5%"]
high_eff = summary_eff["high_87.5%"]

final_eff = df_mc_eff.iloc[:, 1:].mean(axis=1)
final_eff_ox = df_mc_eff_ox.iloc[:, 1:].mean(axis=1)
final_eff_red = df_mc_eff_red.iloc[:, 1:].mean(axis=1)



S_mass_mean = (40*(1000)**3*2500*0.087* final_eff[len(MgO)-1]/10000)*10**(-9)
S_mass_low = (40*(1000)**3*2500*0.087* low_eff[len(MgO)-1]/10000)*10**(-9)
S_mass_high = (40*(1000)**3*2500*0.087* high_eff[len(MgO)-1]/10000)*10**(-9)

S_mass_ox = (40*(1000)**3*2500*0.087 * final_eff_ox[len(MgO)-1]/10000)*10**(-9)
S_mass_ox_low = (40*(1000)**3*2500*0.087* low_eff_ox[len(MgO)-1]/10000)*10**(-9)
S_mass_ox_high = (40*(1000)**3*2500*0.087* high_eff_ox[len(MgO)-1]/10000)*10**(-9)

S_mass_red = (40*(1000)**3*2500*0.087* final_eff_red[len(MgO)-1]/10000)*10**(-9)
S_mass_red_low = (40*(1000)**3*2500*0.087* low_eff_red[len(MgO)-1]/10000)*10**(-9)
S_mass_red_high = (40*(1000)**3*2500*0.087* high_eff_red[len(MgO)-1]/10000)*10**(-9)

print(f"final sulfur {S_mass_mean}, between {S_mass_low} and {S_mass_high}; for oxidized condition {S_mass_ox},between {S_mass_ox_low} and {S_mass_ox_high}."
      f"for reduced conditions, {S_mass_red}, between {S_mass_red_low} and {S_mass_red_high}.")
# # plt.figure(1)
# # for k in range(1, n):
# #     plt.plot(MgO, df_mc.iloc[:, k])
# #matplotlib inline
# #config InlineBackend.figure_format = 'retina'
# #
# rc('font',**{'size': 20})
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
# plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
# plt.rcParams["xtick.labelsize"] = 8 # Sets size of numbers on tick marks
# plt.rcParams["ytick.labelsize"] = 8 # Sets size of numbers on tick marks
# plt.rcParams["axes.titlesize"] = 10
# plt.rcParams["axes.labelsize"] = 10 # Axes labels
#
# sz_sm = 80
# sz = 150
# # #
# # #
# # # calculate error bars for Fe2O3 measurements
# # df_plot["Fe2O3"] = 0.5*(71.844*2+15.99)*df_plot["FeO"] * df_plot["Fe3+"]/(100*71.844)
# # df_plot["Fe2O3_pe"] = 0.5*(71.844*2+15.99)*df_plot["per"]* df_plot["FeO"]/(100*71.844)
# # df_plot["Fe2O3_ne"] = 0.5*(71.844*2+15.99)*df_plot["ner"]* df_plot["FeO"]/(100*71.844)
# #
# #
#figure S7 Th and S6+ vs. MgO
th_trend = []
sulfate_trend = []
for l in range(0, len(MgO)):
    if -0.9707 * MgO[l] + 1.5908 < 1:
        sulfate = -0.9707 * MgO[l] + 1.5908
    else:
        sulfate = 1
    th = -10.478 * MgO[l] + 21.963
    sulfate_trend.append(sulfate)
    th_trend.append(th)

figs7, axs7 = plt.subplots(1, 2, sharex=True, figsize=(8, 6))
axs7[0].plot(df_plot["MgO"][24:39], df_plot["Th"][24:39], "o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
axs7[0].plot(MgO, th_trend, linestyle ="--", color = "grey")
slope = -10.478
intercept = 21.963
equation_text = f'Th (ppm) = {slope:.2f}MgO(wt.%) + {intercept:.2f}'
axs7[0].legend(labels = ["$\mathregular{MI_{V16}}$", equation_text], fontsize = 12, loc =(0.250,0.9))
               # labelspacing = 0.2, handletextpad=0.8, handlelength = 0.01, prop={'size': 10}, frameon=True)
axs7[0].annotate("(a) Th vs. MgO", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12)
axs7[0].set_ylabel('Th (ppm)')
axs7[0].set_xlabel('MgO (wt.%)')
axs7[0].set_ylim([8,20])
axs7[0].set_xlim([0.5, 1.5])


axs7[1].plot(df_plot["MgO"][0:20], df_plot["S6+"][0:20], "o", markerfacecolor="red", markeredgecolor="black", markersize=10)
axs7[1].plot(MgO, sulfate_trend, linestyle ="--", color = "black")
slope_s = -0.9707
intercept_s = 1.5908
equation_text_s = r"$\mathregular{S^{6+}}/\sum S$ = -0.97MgO(wt.%) + 1.59"
axs7[1].legend(labels = ["$\mathregular{MI_{thisstudy}}$", equation_text_s], fontsize = 12, loc =(0.28,0.86))
               # labelspacing = 0.2, handletextpad=0.8, handlelength = 0.01, prop={'size': 10}, frameon=True)
axs7[1].annotate(r"(b) $\mathregular{S^{6+}}/\sum S$ vs. MgO", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12)
axs7[1].set_ylabel(r"$\mathregular{S^{6+}}/\sum S$")
axs7[1].set_xlabel('MgO (wt.%)')
axs7[1].set_ylim([0.4, 1.01])
axs7[1].set_xlim([0.5, 1.5])
# #

# figure S1, major elements comparison to MELTS results
figs1, axs1 = plt.subplots(4, 2, sharex=True, figsize=(8, 14))
axs1 = axs1.flatten()
figs1.tight_layout()
axs1[0].plot(df_plot["MgO"][24:66], df_plot["SiO2"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
axs1[0].errorbar(df_plot["MgO"][66], df_plot["SiO2"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["SiO2"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[0].plot(df_plot["MgO"][0:20], df_plot["SiO2"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 10)
axs1[0].errorbar(df_plot["MgO"][23:24], df_plot["SiO2"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["SiO2"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[0].errorbar(df_plot["MgO"][72:73], df_plot["SiO2"][72:73], xerr=df_plot["MgO"][73:74],
             yerr=df_plot["SiO2"][73:74], fmt="*", markerfacecolor="orange", markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[0].plot(df_melts["MgO"][1:], df_melts["SiO2"][1:], linestyle="-", color="black")
axs1[0].set_ylabel('$\mathregular{SiO_2}$ (wt.%)')
axs1[0].set_ylim([60, 74])
axs1[0].set_xlim([0.5, 1.5])
axs1[0].annotate("(a) $\mathregular{SiO_2}$ (wt.%)", xy=(0.01, 0.05), xycoords="axes fraction", fontsize=12)

axs1[1].plot(df_plot["MgO"][24:66], df_plot["FeO"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
axs1[1].errorbar(df_plot["MgO"][66], df_plot["FeO"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["FeO"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[1].plot(df_plot["MgO"][0:20], df_plot["FeO"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 10)
axs1[1].errorbar(df_plot["MgO"][23:24], df_plot["FeO"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["FeO"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[1].errorbar(df_plot["MgO"][72], df_plot["FeO"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["FeO"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[1].plot(df_melts["MgO"][1:], df_melts["FEOT"][1:], linestyle="-", color="black")
axs1[1].set_ylabel(r'$FeO_T$ (wt.%)')
axs1[1].set_ylim([0, 5])
axs1[1].set_xlim([0.5,1.5])
axs1[1].annotate("(b) $\mathregular{FeO_T}$", xy=(0.01, 0.90), xycoords="axes fraction", fontsize=12)

axs1[2].plot(df_plot["MgO"][24:66], df_plot["Al2O3"][24:66],"o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
axs1[2].plot(df_plot["MgO"][0:20], df_plot["Al2O3"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize=10)
axs1[2].plot(df_melts["MgO"][1:], df_melts["Al2O3"][1:], linestyle="-", color="black")
axs1[2].errorbar(df_plot["MgO"][66], df_plot["Al2O3"][66], xerr=df_plot["MgO"][67],
                 yerr=df_plot["Al2O3"][67], fmt="d", markerfacecolor="grey", markeredgecolor="grey", markersize=6,
                 ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[2].errorbar(df_plot["MgO"][23:24], df_plot["Al2O3"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["Al2O3"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[2].errorbar(df_plot["MgO"][72], df_plot["Al2O3"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["Al2O3"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)

axs1[2].set_xlim([0.5, 1.5])
axs1[2].legend(labels = ["$\mathregular{MI_{V16}}$",  "$\mathregular{MI_{thisstudy}}$ ", "MELTS_v1.2","$\mathregular{mg_{V16}}$ ", "$\mathregular{mg_{thisstudy}}$ ", "whole rock"],
               loc =(0.6500,0.05),labelspacing = 0.2, handletextpad=0.8, handlelength = 0.01, prop={'size': 10}, frameon=True)
# ax1[2].set_xticklabels([0.50, 0.75, 1.00, 1.25, 1.50])
axs1[2].set_ylim([12, 18])
axs1[2].annotate("(c) $\mathregular{Al_2O_3}$", xy=(0.01, 0.90), xycoords="axes fraction", fontsize=12)
axs1[2].set_ylabel('$\mathregular{Al_2O_3}$(wt.%)')
# axs1[2].set_xlabel('MgO (wt.%)')
axs1[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
axs1[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)

axs1[3].plot(df_plot["MgO"][24:66], df_plot["CaO"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6)
axs1[3].errorbar(df_plot["MgO"][66], df_plot["CaO"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["CaO"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[3].plot(df_plot["MgO"][0:20], df_plot["CaO"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 10)
axs1[3].errorbar(df_plot["MgO"][23:24], df_plot["CaO"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["CaO"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="black", elinewidth=1, capthick=1, capsize=3)
axs1[3].errorbar(df_plot["MgO"][72], df_plot["CaO"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["CaO"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[3].plot(df_melts["MgO"][1:], df_melts["CaO"][1:], linestyle="-", color="black")
axs1[3].set_ylabel("CaO (wt.%)")
axs1[3].set_ylim([1.5, 3.5])
axs1[3].set_xlim([0.5, 1.5])
axs1[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
axs1[3].tick_params(axis="y", direction='in', length=5, pad = 6.5)
axs1[3].annotate("(d) CaO", xy=(0.01, 0.90), xycoords="axes fraction", fontsize=12)

axs1[4].plot(df_plot["MgO"][24:66], df_plot["TiO2"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
axs1[4].errorbar(df_plot["MgO"][66], df_plot["TiO2"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["FeO"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[4].plot(df_plot["MgO"][0:20], df_plot["TiO2"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 10)
axs1[4].errorbar(df_plot["MgO"][23:24], df_plot["TiO2"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["TiO2"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[4].errorbar(df_plot["MgO"][72], df_plot["TiO2"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["TiO2"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[4].plot(df_melts["MgO"][1:], df_melts["TiO2"][1:], linestyle="-", color="black")
axs1[4].set_ylim([0, 2])
axs1[4].set_xlim([0.5, 1.5])
axs1[4].set_ylabel('$\mathregular{TiO_2}$ (wt.%)')
axs1[4].set_xlabel ("MgO (wt.%)")
axs1[4].annotate("(e) $\mathregular{TiO_2}$ (wt.%)", xy=(0.01, 0.90), xycoords="axes fraction", fontsize=12)

axs1[5].plot(df_plot["MgO"][24:66], df_plot["H2O"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6)
axs1[5].plot(df_melts["MgO"][1:], df_melts["H2O"][1:], linestyle="-", color="black")
axs1[5].set_ylim([3, 6])
axs1[5].set_xlim([0.5, 1.5])
axs1[5].set_ylabel('$\mathregular{H_2O}$ (wt.%)')
axs1[5].set_xlabel ("MgO (wt.%)")
axs1[5].annotate("(f) $\mathregular{H_2O}$ (wt.%)", xy=(0.01, 0.90), xycoords="axes fraction", fontsize=12)

axs1[6].plot(df_plot["MgO"][24:66], df_plot["Na2O"][24:66], "o", markerfacecolor="white", markeredgecolor="grey",markersize = 6)
axs1[6].errorbar(df_plot["MgO"][66], df_plot["Na2O"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["Na2O"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[6].plot(df_plot["MgO"][0:20], df_plot["Na2O"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 10)
axs1[6].errorbar(df_plot["MgO"][23:24], df_plot["Na2O"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["Na2O"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="black", elinewidth=1, capthick=1, capsize=3)
axs1[6].errorbar(df_plot["MgO"][72], df_plot["Na2O"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["Na2O"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[6].plot(df_melts["MgO"][1:], df_melts["Na2O"][1:], linestyle="-",color="black")
axs1[6].set_ylabel('$\mathregular{Na_2O}$ (wt.%)')
# axs1[5].set_ylabel('MgO (wt.%)')
axs1[6].set_ylim([2, 6])
axs1[6].set_xlim([0.5, 1.5])
axs1[6]. set_xlabel ("MgO (wt.%)")
axs1[6].annotate("(g) $\mathregular{Na_2O}$ (wt.%)", xy=(0.01, 0.10), xycoords="axes fraction", fontsize=12)

axs1[7].plot(df_plot["MgO"][24:66], df_plot["K2O"][24:66], "o", markerfacecolor="white", markeredgecolor="grey",markersize = 6)
axs1[7].errorbar(df_plot["MgO"][66], df_plot["K2O"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["Na2O"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[7].plot(df_plot["MgO"][0:20], df_plot["K2O"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 10)
axs1[7].errorbar(df_plot["MgO"][23:24], df_plot["K2O"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["Na2O"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="black", elinewidth=1, capthick=1, capsize=3)
axs1[7].errorbar(df_plot["MgO"][72], df_plot["K2O"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["K2O"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=12,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
axs1[7].plot(df_melts["MgO"][1:], df_melts["K2O"][1:], linestyle="-",color="black")
axs1[7].set_ylabel('$\mathregular{K_2O}$ (wt.%)')
axs1[7].set_ylim([3, 7])
axs1[7].set_xlim([0.5, 1.5])
axs1[7].set_xlabel ("MgO (wt.%)")
axs1[7].annotate("(h) $\mathregular{K_2O}$ (wt.%)", xy=(0.01, 0.90), xycoords="axes fraction", fontsize=12)
figs1.savefig("FigS1_majorelements.jpg", dpi=300, format='jpg', bbox_inches='tight')
figs1.savefig("FigS1_majorelements.eps", format='eps', bbox_inches='tight')



#figure 2 major elements vs. MgO
fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(4.7, 6))
ax1 = ax1.flatten()
plt.subplots_adjust(left=0.15, top=0.95, bottom=0.1)
ax1[0].plot(df_plot["MgO"][24:66], df_plot["SiO2"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
ax1[0].errorbar(df_plot["MgO"][66], df_plot["SiO2"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["SiO2"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[0].plot(df_plot["MgO"][0:20], df_plot["SiO2"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 8)
ax1[0].errorbar(df_plot["MgO"][23:24], df_plot["SiO2"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["SiO2"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=8,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[0].errorbar(df_plot["MgO"][72:73], df_plot["SiO2"][72:73], xerr=df_plot["MgO"][73:74],
             yerr=df_plot["SiO2"][73:74], fmt="*", markerfacecolor="orange", markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[0].set_ylabel('$\mathregular{SiO_2}$ (wt.%)')
ax1[0].set_ylim([63, 73])
ax1[0].set_xlim([0.5, 1.5])
ax1[0].set_yticks(np.arange(63,73.5,2.5))
ax1[0].set_xticks(np.arange(0.5,1.6,0.2))
ax1[0].legend(labels=["","","","",""], loc=(0.75, 0.6), labelspacing=0.3, handletextpad=0.25, handlelength=0.005, prop={'size': 10}, ncol = 3, frameon = False)
ax1[0].annotate("MIs", xy=(0.75, 0.88), xycoords='axes fraction', fontsize=8)
ax1[0].annotate("MG", xy=(0.83, 0.88), xycoords='axes fraction', fontsize=8)
ax1[0].annotate("WR", xy=(0.91, 0.88), xycoords='axes fraction', fontsize=8)
ax1[0].annotate("ref. 17", xy=(0.58, 0.75), xycoords='axes fraction', fontsize=8)
ax1[0].annotate("this study", xy=(0.58, 0.65), xycoords='axes fraction', fontsize=8)
ax1[0].annotate("a ", xy=(-0.17, 1.02), xycoords='axes fraction', fontsize=12, weight = "bold")
rect = patches.Rectangle((1.07, 69), 0.425, 3.52, linewidth=0.8, edgecolor='black', facecolor='none')

# Add the patch to the Axes
ax1[0].add_patch(rect)

# rect = patches.Rectangle((0.5, 0.8), 3, 4, linewidth=4, edgecolor='black', facecolor="red")
# ax1[0].add_patch(rect)

ax1[1].plot(df_plot["MgO"][24:66], df_plot["FeO"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
ax1[1].errorbar(df_plot["MgO"][66], df_plot["FeO"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["FeO"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[1].plot(df_plot["MgO"][0:20], df_plot["FeO"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize = 8)
ax1[1].errorbar(df_plot["MgO"][23:24], df_plot["FeO"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["FeO"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=8,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[1].errorbar(df_plot["MgO"][72], df_plot["FeO"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["FeO"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[1].set_ylabel("$\mathregular{FeO_T}$ (wt.%)")
ax1[1].set_ylim([1.5, 5])
ax1[1].set_xlim([0.5,1.5])
ax1[1].annotate("b ", xy=(-0.17, 1.02), xycoords='axes fraction', fontsize=12, weight = "bold")

ax1[2].plot(df_plot["MgO"][24:66], df_plot["Al2O3"][24:66],"o", markerfacecolor="white", markeredgecolor="grey", markersize = 6)
ax1[2].errorbar(df_plot["MgO"][66], df_plot["Al2O3"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["Al2O3"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[2].plot(df_plot["MgO"][0:20], df_plot["Al2O3"][0:20], "o", markerfacecolor="red", markeredgecolor= "black",markersize=8)
ax1[2].errorbar(df_plot["MgO"][23:24], df_plot["Al2O3"][23:24], xerr=df_plot["MgO"][71:72],
             yerr=df_plot["Al2O3"][71:72], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=8,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[2].errorbar(df_plot["MgO"][72], df_plot["Al2O3"][72], xerr=df_plot["MgO"][73],
             yerr=df_plot["Al2O3"][73], fmt="*", markerfacecolor="orange",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax1[2].set_xlim([0.5, 1.51])
# ax1[2].set_xticklabels([0.50, 0.75, 1.00, 1.25, 1.50])
ax1[2].set_ylim([15, 18])
ax1[2].annotate("c ", xy=(-0.17, 1.02), xycoords='axes fraction', fontsize=12, weight = "bold")
ax1[2].set_ylabel('$\mathregular{Al_2O_3}$ (wt.%)')
ax1[2].set_xlabel('MgO (wt.%)')
ax1[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax1[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
fig1.savefig("Fig2_majorelements.jpg", dpi=300, format='jpg', bbox_inches='tight')
fig1.savefig("Fig2_majorelements.eps", format='eps', bbox_inches='tight')

#

# #
#figure 3 volatiles vs. MgO
fig3, ax3 = plt.subplots(3, 1, sharex=True, figsize = (4.7, 6))
ax3 = ax3.flatten()
plt.subplots_adjust(left=0.15, top=0.95, bottom=0.1)

ax3[0].plot(df_plot["MgO"][24:66], df_plot["S"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6)
ax3[0].errorbar(df_plot["MgO"][66], df_plot["S"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["S"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax3[0].plot(df_plot["MgO"][0:20], df_plot["S"][0:20], "o", markerfacecolor="red", markeredgecolor="black", markersize=8)
ax3[0].errorbar(df_plot["MgO"][23:24], df_plot["S"][23:24], xerr=df_plot["MgO"][71:72], yerr=df_plot["S"][71:72], fmt="d",
             markerfacecolor="red", markeredgecolor="black", markersize=8, ecolor="black", elinewidth=1, capthick=1, capsize=3)
ax3[0].set_xlim([0.5, 1.4])
ax3[0].set_ylim([0,  600])
ax3[0].set_ylabel("S (ppm)")
ax3[0].legend(labels=["","","","",""], loc=(0.85, 0.05), labelspacing=0.3, handletextpad=0.25, handlelength=0.005, prop={'size': 10}, ncol = 2, frameon = False)
ax3[0].annotate("MIs", xy=(0.84, 0.33), xycoords='axes fraction', fontsize=8)
ax3[0].annotate("MG", xy=(0.93, 0.33), xycoords='axes fraction', fontsize=8)
ax3[0].annotate("ref. 17", xy=(0.68, 0.22), xycoords='axes fraction', fontsize=8)
ax3[0].annotate("this study", xy=(0.68, 0.12), xycoords='axes fraction', fontsize=8)
ax3[0].annotate("a ", xy=(-0.15, 1.02), xycoords='axes fraction', fontsize=12, weight = "bold")
rect = patches.Rectangle((1.10, 46), 0.295, 215, linewidth=0.8, edgecolor='black', facecolor='none')
# Add the patch to the Axes
ax3[0].add_patch(rect)

ax3[1].plot(df_plot["MgO"][24:66], df_plot["Cl"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6)
ax3[1].errorbar(df_plot["MgO"][66], df_plot["Cl"][66], xerr=df_plot["MgO"][67],
             yerr=df_plot["Cl"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey", markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax3[1].plot(df_plot["MgO"][0:20], df_plot["Cl"][0:20], "o", markerfacecolor="red", markeredgecolor="black",markersize=8)
ax3[1].errorbar(df_plot["MgO"][23:24], df_plot["Cl"][23:24],  yerr=df_plot["Cl"][71:72], xerr=df_plot["MgO"][71:72],
                fmt="d", markerfacecolor="red", markeredgecolor="black",markersize=8, ecolor="black", elinewidth=1,
                capthick=1, capsize=3)
ax3[1].annotate("b", xy=(-0.15, 1.02), xycoords='axes fraction', fontsize=12, weight = "bold")

ax3[1].set_ylabel("Cl (ppm)")
ax3[1].set_xlim([0.5, 1.4])
ax3[1].set_ylim([1000, 5000])

ax3[2].plot(df_plot["MgO"][24:66], df_plot["H2O"][24:66]/df_plot["K2O"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6)
inset_ax = inset_axes(ax3[2], height="65%", width="65%", loc=3, bbox_to_anchor=(.01, .55, .6, .5), bbox_transform=ax3[2].transAxes)
inset_ax.plot(df_plot["MgO"][24:66], df_plot["H2O"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=4)
inset_ax.set_yticks(np.arange(2,6.1,1))
inset_ax.set_xlim(0.6, 1.4)  # Set x-axis range
inset_ax.set_xticks(np.arange(0.6, 1.41, 0.2))  # Major ticks every 0.2
inset_ax.yaxis.tick_right()
inset_ax.yaxis.set_ticks_position('right')  # Show ticks on both sides
inset_ax.yaxis.set_label_position('right')  # Set the label position to the right
inset_ax.set_xlabel("MgO (wt.%)", loc="left", fontsize=8)
inset_ax.set_ylabel("$\mathregular{H_2O}$ (wt.%)", fontsize=8)

ax3[2].set_xlim([0.5, 1.4])
ax3[2].set_ylim([0.5, 2])
ax3[2].set_xlabel("MgO (wt.%)")
ax3[2].set_ylabel(r"$\mathrm{H_2O}/\mathrm{ K_2O}$")
ax3[2].annotate("c", xy=(-0.15, 1.02), xycoords='axes fraction', fontsize=12, weight = "bold")
fig3.savefig("Fig3_volatiles.jpg", dpi=300, format='jpg', bbox_inches='tight')
fig3.savefig("Fig3_volatiles.eps", format='eps', bbox_inches='tight')

#
# #
#figure4 S vs. S6+/ST, Fe3+/FeT and dfmq
fig4, ax4 = plt.subplots(2, 2, figsize = (9.7, 6))
ax4 = ax4.flatten()
# fig4.tight_layout()
plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1)

ax4[1].errorbar(y= df_plot["Fe3+"][0:21]/100, x=df_plot["MgO"][0:21], yerr=[df_plot["ner"][0:21]/100, df_plot["per"][0:21]/100],
             fmt="o", markerfacecolor="red",markeredgecolor="black",markersize=10, ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax4[1].errorbar(y=df_plot["Fe3+"][21:23]/100, x=df_plot["MgO"][21:23], yerr=[df_plot["ner"][21:23]/100, df_plot["per"][21:23]/100],
                fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax4[1].set_ylim([0.2, 0.55])
ax4[1].set_xlim([0.4, 1.2])
ax4[1].set_xlabel("MgO (wt.%)")
ax4[1].set_ylabel("$\mathregular{Fe^{3+}}$/$\sum$Fe")
ax4[1].annotate("b", xy=(-0.1, 1.05), xycoords="axes fraction", fontsize=12, weight="bold")
ax4_a = fig4.add_subplot(222,frame_on = False)
ax4_a.errorbar(y=dfmq_AM[0:21],x =df_plot["MgO"][0:21],yerr=[dfmq_AM_neg[0:21], dfmq_AM_pos[0:21]], fmt="o", markerfacecolor="white",markeredgecolor="#009E73",markersize=6,
             ecolor="#009E73", elinewidth=1, capthick=1, capsize=3)
ax4_a.errorbar(y = dfmq_AM[21:24], x=df_plot["MgO"][21:24], yerr=[dfmq_AM_neg[21:24], dfmq_AM_pos[21:24]], fmt="d", markerfacecolor="white",markeredgecolor="#009E73",markersize=6,
             ecolor="#009E73", elinewidth=1, capthick=1, capsize=3)
ax4_a.axhspan(ymin=0.76, ymax=0.84, xmin=0, xmax=1, color='#009E73', alpha=0.3, label='FeTiMM')

# dfmq_ti = np.linspace(0, 1, 100)
# ax4_a.fill_betweenx(y=np.linspace(-1, 4, 100), x1=0.76, x2=0.84, color='grey', alpha=0.3, label='FeTiMM')
#ax4_a.fill_between(dfmq_ti, 0.4, 1.2, where=(dfmq_ti >= 0.76) & (dfmq_ti <= 0.84), color='grey', alpha=0.3, label='FeTiMM')
ax4_a.set_xlim([0.4, 1.2])
ax4_a.yaxis.tick_right()
ax4_a.yaxis.set_label_position('right')
ax4_a.tick_params(axis='y', colors="#009E73")
ax4_a.set_ylabel(r"$\Delta$FMQ", rotation=270, labelpad=10)
ax4_a.yaxis.label.set_color('#009E73')
ax4_a.set_ylim([-1, 4])
ax4_a.set_xlabel("MgO (wt.%)")
lg1= ax4[1].legend(["$\mathregular{MI_{this study}}$", "$\mathregular{MG_{this study}}$"], loc ="upper left",labelspacing = 0.3, handletextpad = 0.5,handlelength = 0.5, prop={'size': 10}, frameon=True)
lg1.get_frame().set_edgecolor("black")
indexes_one = np.where(df_plot["S6+"][0:21] == 1)[0]
indexes_notone = np.where(df_plot["S6+"][0:21] != 1)[0]

ax4[0].errorbar(df_plot["S6+"][indexes_one], df_plot["S"][indexes_one], xerr=[0.1*df_plot["S6+"][indexes_one], np.zeros(len(df_plot["S6+"][indexes_one]))],
            fmt="o", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax4[0].errorbar(df_plot["S6+"][indexes_notone], df_plot["S"][indexes_notone], xerr=0.1*df_plot["S6+"][indexes_notone],
            fmt="o", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax4[0].errorbar(df_plot["S6+"][21:24], df_plot["S"][21:24], xerr=[0.1*df_plot["S6+"][21:24], np.zeros(len(df_plot["S6+"][21:24]))],
             fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
# ax4[0].errorbar(df_plot["S6+"][12:20], df_plot["S"][12:20], xerr=[0.1*df_plot["S6+"][12:20], np.zeros(len(df_plot["S6+"][12:20]))],
#              yerr=0.1*df_plot["S"][12:20], fmt="o", markerfacecolor="red",markeredgecolor="black",markersize=10,
#              ecolor="grey", elinewidth=1, capthick=1, capsize=3)

ax4[0].set_xlim([0.4, 1.05])
ax4[0].set_ylim([0, 600])

ax4[0].set_ylabel("S (ppm)")
ax4[0].set_xlabel("$\mathregular{S^{6+}}$/$\sum$S")
ax4[0].annotate("a", xy=(-0.1, 1.05), xycoords="axes fraction", fontsize=12, weight="bold")

# ax4[2].errorbar(df_plot["dfmq_corr"][0:20], df_plot["S"][0:20],
#              yerr=0.1*df_plot["S"][0:20], fmt="o", markerfacecolor="grey",markeredgecolor="white",markersize=6,
#              ecolor="black", elinewidth=1, capthick=1, capsize=3)
ax4[2].errorbar(df_plot["d34S"][0:21], df_plot["S"][0:21],
             xerr=df_plot["2sigma"][0:21], fmt="o", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax4[2].errorbar(df_plot["d34S"][21:24], df_plot["S"][21:24],
             xerr=df_plot["2sigma"][21:24], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3)
ax4[2].annotate("c", xy=(-0.1, 1.05), xycoords="axes fraction", weight="bold", fontsize=12)
ax4[2].set_ylim([0, 600])
ax4[2].set_xlabel(r"$\delta^{34}$S ‰")
ax4[2].set_ylabel("S (ppm)")
ax4[2].set_xlim([-2, 12])
#ax4[2].legend(["$\mathregular{melt inclusions_{corr}}$", "$\mathregular{melt inclusions_{mea}}$","matrix glass"], loc ="upper left",labelspacing = 0.2, handletextpad = 0.8,handlelength = 0.01, prop={'size': 10}, frameon=False)

ax4[3].plot(df_plot["Annum"][24:66], df_plot["S"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6)
ax4[3].plot(df_plot["Annum"][66], df_plot["S"][66], "d", markerfacecolor="grey",markeredgecolor="grey",markersize=6)
ax4[3].plot(df_plot["Annum"][0:21], df_plot["S"][0:21], "o", markerfacecolor="red", markeredgecolor="black", markersize=10)
ax4[3].plot(df_plot["Annum"][21:24], df_plot["S"][21:24], "d",markerfacecolor="red", markeredgecolor="black", markersize=10)
ax4[3].set_xlabel("An%")
lg = ax4[3].legend(["$\mathregular{MIs_{ref. 17}}$", "$\mathregular{MG_{ref. 17}}$"], loc ="upper left",labelspacing = 0.2,
              handletextpad = 0.5,handlelength = 0.5, prop={'size': 10}, frameon=True)
lg.get_frame().set_edgecolor("black")
ax4[3].set_ylim([0, 600])
ax4[3].set_xlim([40, 90])
ax4[3].set_ylabel("S (ppm)")
ax4[3].annotate("d", xy=(-0.1, 1.05), xycoords="axes fraction", weight="bold", fontsize=12)

# fig4.tight_layout(rect=[0, 0, 1, 0.97])
fig4.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.93, hspace=0.25, wspace=0.25)
fig4.savefig("Fig4_results.jpg", dpi=300, format='jpg', bbox_inches='tight')
fig4.savefig("Fig4_results.eps", format='eps', bbox_inches='tight')
#

#figure6 kd, S, S degassing efficiency and d34S vs. MgO
lightblue = plt.cm.cividis(0.6)
lightpurple = plt.cm.viridis(0.7)
#
fig6, ax6 = plt.subplots(2, 2, figsize=(9.7, 6), gridspec_kw={"hspace": 0.35, "wspace": 0.3})
ax6 = ax6.flatten()
fig6.tight_layout()
for k in range(1, n):
    ax6[0].plot(MgO, df_mc_kdcombined.iloc[:, k], color="#FFA07A", alpha = 0.7, zorder=1)
ax6[0].plot(MgO[1:], df_mc_kdcombined.iloc[1:, 1:].mean(axis=1), color="red",zorder=4)
ax6[0].text(0.9, 40, "Kd_combined", fontsize = 14, color = "red")
ax6[0].errorbar(
    x=df_plot["MgO"][:len(kd_mi)],
    y=kd_mi,
    yerr=kd_mi_err,
    fmt='o',                # marker style
    ecolor='black',          # error bar color
    capsize=3, # error bar cap size
    zorder=5,
)
sc=ax6[0].scatter(x = df_plot["MgO"][0:len(kd_mi)], y=kd_mi, c=df_plot["S6+"][0:len(kd_mi)],cmap='viridis', s=60, edgecolors='black', linewidth=0.6, alpha=0.8, zorder=6)

cbar_ax = inset_axes(ax6[0], width="70%", height="5%", loc='upper left', bbox_to_anchor=(0.25, 0, 1, 1), bbox_transform=ax6[0].transAxes)
cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
# cbar.set_label('$\mathregular{S^{6+}}$/$\sum$S', labelpad=4, loc='left')
cbar.ax.text(-0.05, 0.5, '$\mathregular{S^{6+}}$/$\sum$S',
             va='center', ha='right', fontsize=11, transform=cbar.ax.transAxes)
ax6[0].annotate("a", xy=(-0.1, 1.05), xycoords="axes fraction", fontsize=14, weight="bold")
# ax6[0].text(1.2, 20, "Kd_ox", fontsize = 14, ha= "center", va="center", color="grey")
ax6[0].set_ylim([0, 300])
ax6[0].set_xlim([0.4, 1.4])
ax6[0].set_ylabel("kd", fontsize=12)
ax6[0].set_xlabel("MgO (wt.%)", fontsize=12)
# Extract mean and CI bounds

ax6[1].fill_between(MgO, low_s_melt, high_s_melt, color="#FFA07A", alpha=0.3, label='75% CI')
ax6[1].plot(MgO, mean_s_melt, color= "red", lw= 2)


ax6[1].fill_between(MgO, low_s_ox, high_s_ox, color="lightblue", alpha=0.3, label='75% CI')
ax6[1].plot(MgO, mean_s_ox, color= "lightblue", lw= 2)

ax6[1].plot(MgO, low_s_red, color= lightpurple, lw= 2)
ax6[1].plot(MgO, high_s_red, color= lightpurple, lw= 2)
ax6[1].plot(MgO, mean_s_red, color= lightpurple, lw= 2)

# ax6[1].plot(MgO, S_melt_red, color= "black", lw = 1.5)
# ax6[1].plot(MgO[1:], df_mc_ox.iloc[1:, 1:].mean(axis=1), color= "gray", lw= 1.5)
p1,=ax6[1].plot(df_plot["MgO"][24:66], df_plot["S"][24:66], "o", markerfacecolor="white", markeredgecolor="grey", markersize=6, label="$\mathregular{MI_{V16}}$")
p2 =ax6[1].errorbar(df_plot["MgO"][66], df_plot["S"][66], xerr=df_plot["MgO"][67],yerr=df_plot["S"][67], fmt="d", markerfacecolor="grey",markeredgecolor="grey",markersize=6,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3, label="$\mathregular{MI_{thisstudy}}$ ")
p3, =ax6[1].plot(df_plot["MgO"][0:20], df_plot["S"][0:20], "o", markerfacecolor="red", markeredgecolor="black", markersize=10, label="$\mathregular{MG_{V16}}$")
p4 =ax6[1].errorbar(df_plot["MgO"][23:24], df_plot["S"][23:24], xerr=df_plot["MgO"][71:72], yerr=df_plot["S"][71:72], fmt="d",
             markerfacecolor="red", markeredgecolor="black", markersize=10, ecolor="black", elinewidth=1, capthick=1, capsize=3, label = "$\mathregular{MG_{thisstudy}}$ ")
ax6[1].legend(handles=[p1, p2, p3, p4], labels=["", "", "", ""], loc=(0.21, 0.3),
              frameon=False, ncol=2, prop={'size': 10}, handlelength=0.005, handletextpad=0.5)

ax6[1].set_xlim([0.4, 1.4])
ax6[1].set_ylim([0,  1500])
ax6[1].set_xlabel("MgO (wt.%)", fontsize=12)
ax6[1].set_ylabel("S (ppm)", fontsize=12)
proxy_line = Line2D([0], [0], linestyle='none', c='none', marker='o', label='Proxy')
handles = [p1, p2, p3, p4]
ax6[1].legend(handles=handles, labels=["","","",""], loc= (0.21, 0.3), labelspacing=0.5, handletextpad=0.5, handlelength=0.005, prop={'size': 10}, ncol = 2, frameon = False)
ax6[1].annotate("MIs", xy=(0.2, 0.52), xycoords='axes fraction', fontsize=8)
ax6[1].annotate("MG", xy=(0.32, 0.52), xycoords='axes fraction', fontsize=8)
ax6[1].annotate("ref. 17", xy=(0.03, 0.45), xycoords='axes fraction', fontsize=8)
ax6[1].annotate("this study", xy=(0.03, 0.35), xycoords='axes fraction', fontsize=8)
transform=ax6[1].transAxes
rect6 = patches.Rectangle((0.01,0.28), 0.37, 0.32, linewidth=0.8, edgecolor='black', facecolor='none', transform=ax6[1].transAxes,  # important! keeps it in fraction units
    zorder=1)
# Add the patch to the Axes
ax6[1].add_patch(rect6)
# ax6[1].legend(handles, labels, loc ="center left",labelspacing = 0.2, handletextpad = 0.8,handlelength = 0.01, prop={'size': 10}, frameon=True)
ax6[1].annotate("b", xy=(-0.1, 1.05), xycoords="axes fraction", fontsize=14, weight="bold")

mean_d34s_open = summary_d34s_open["mean"]
low_d34s_open = summary_d34s_open["low_12.5%"]
high_d34s_open = summary_d34s_open["high_87.5%"]

mean_d34s_closed = summary_d34s_closed["mean"]
low_d34s_closed = summary_d34s_closed["low_12.5%"]
high_d34s_closed = summary_d34s_closed["high_87.5%"]


ax6[2].fill_between(MgO, low_d34s_open, high_d34s_open, color="#FFA07A", alpha=0.3, label='75% CI')
ax6[2].plot(MgO, mean_d34s_open, color= "red",linestyle= "--", lw= 2)

# ax6[2].plot(MgO, low_d34s_closed, color= "red", lw= 2)
# ax6[2].plot(MgO, high_d34s_closed, color= "red", lw= 2)
ax6[2].fill_between(MgO, low_d34s_closed, high_d34s_closed, color="#FFA07A", alpha=0.3, label='75% CI')
ax6[2].plot(MgO, mean_d34s_closed, color= "red",linestyle= "-", lw= 2)


ax6[2].plot(MgO, df_mc_d34s_open.iloc[:, 1:].mean(axis=1), color = "red", linestyle = "--", lw =2)
ax6[2].plot(MgO, df_mc_d34s_closed.iloc[:, 1:].mean(axis=1), color = "red", lw=2)
ax6[2].errorbar(df_plot["MgO"][0:20], df_plot["d34S"][0:20],
             yerr=df_plot["2sigma"][0:20], fmt="o", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3, linewidth=1)
ax6[2].errorbar(df_plot["MgO"][23:24], df_plot["d34S"][23:24],
             yerr=df_plot["2sigma"][23:24], fmt="d", markerfacecolor="red",markeredgecolor="black",markersize=10,
             ecolor="grey", elinewidth=1, capthick=1, capsize=3, linewidth=1)


ax6[2].annotate("c", xy=(-0.1, 1.05), xycoords="axes fraction", weight="bold", fontsize=14)
ax6[2].set_xlim([0.4, 1.4])
ax6[2].set_ylabel("$\u03B4^{34}$S ‰", fontsize=12)
ax6[2].set_xlabel("MgO (wt.%)", fontsize=12)
ax6[2].set_ylim([-2, 16])



ax6[3].fill_between(MgO, low_eff_ox, high_eff_ox, color="lightblue", alpha=0.3, label='75% CI')
ax6[3].plot(MgO, mean_eff_ox, color= "lightblue", lw= 2)

ax6[3].fill_between(MgO, low_eff_red, high_eff_red, color=lightpurple, alpha=0.3, label='75% CI')
ax6[3].plot(MgO, mean_eff_red, color= lightpurple, lw= 2)

ax6[3].fill_between(MgO, low_eff, high_eff, color="#FFA07A", alpha=0.3, label='75% CI')
ax6[3].plot(MgO, mean_eff, color= "red", lw= 2)

ax6[3].set_xlim([0.4, 1.4])
ax6[3].set_ylabel("degassing efficiency (%)", fontsize=12)
ax6[3].set_xlabel("MgO (wt.%)", fontsize=12)
ax6[3].set_ylim([0, 100])
ax6[3].annotate("d", xy=(-0.1, 1.05), xycoords="axes fraction", weight="bold", fontsize=14)

# fig6.savefig("Fig7_modelresults_2.jpg", dpi=300, format='jpg', bbox_inches='tight')
# fig6.savefig("Fig7_modelresults_2.eps", format='eps', bbox_inches='tight')



print(f"Final degassing efficiency for Samalas is {final_eff[len(MgO)-1]}%, between {low_eff[len(MgO)-1]}% and {high_eff[len(MgO)-1]}%"
      f"and for oxidized condition if {final_eff_ox[len(MgO)-1]}%,between {low_eff_ox[len(MgO)-1]}% and {high_eff_ox[len(MgO)-1]}%;"
      f"for reduced condition is {final_eff_red[len(MgO)-1]}%, between{low_eff_red[len(MgO)-1]}% and {high_eff_red[len(MgO)-1]}%.")

plt.show()

#############calculate the vapor composition in equilibrium with the MIs#########################################
# n = df.count(axis=1, level=None, numeric_only=True)
#
# dfmq = []
# fo2_AM = []
# fO2_ON = []
# fh2o = []
# vapor_ratio = []
# phiSO2 = []
# phiH2S = []
# phiH2O = []
# for i in range(0, 23):
#     composition = {"SiO2": df["SiO2"][i],
#                    "Al2O3": df["Al2O3"][i],
#                    "TiO2": df["TiO2"][i],
#                    "FeOT": df["FeO"][i],
#                    "MgO": df["MgO"][i],
#                    "CaO": df["CaO"][i],
#                    "Na2O": df["Na2O"][i],
#                    "K2O": df["K2O"][i],
#                    "P2O5": df["P2O5"][i],
#                    "MnO": df["MnO"][i],
#                    }
#     P = df["pressure"][i]
#     Tk = df["temperature"][i] + 273.15
#     ferric_ratio = df["ferric_corr"][i] / 100
#     fo2_0 = OxygenFugacity(P, Tk, composition)
#     dfmq.append(fo2_0.fo2(ferric_ratio) - fo2_0.fmq())
#     oxygen_fugacity = fo2_0.fo2(ferric_ratio)
#     fo2.append(oxygen_fugacity)
#     phi = Fugacity(P, df["temperature"][i])
#     water_fugacity = df["pressure"][i] * phi.phiH2O * 0.95*10
#
#     fh2o.append(water_fugacity)
#     phiH2S.append(phi.phiH2S)
#     phiSO2.append(phi.phiSO2)
#     phiH2O.append(phi.phiH2O)
#     SO2_ST = fo2_0.gas_quilibrium(fo2=oxygen_fugacity, phiso2=phi.phiSO2, phih2s=phi.phiH2S, fh2o=water_fugacity)
#     vapor_ratio.append(SO2_ST)
#
# df["dfmq"] = dfmq
# df["fo2"] = fo2
# df["fh2o"] = fh2o
# df["phiH2S"] = phiH2S
# df["phiSO2"] = phiSO2
# df["phiH2O"] = phiH2O
# df["SO2/ST"] = vapor_ratio
# df.to_csv("Samalas_results_corr.csv")
#############calculate the vapor composition in equilibrium with the MIs end #########################################

#############calculate the vapor composition in equilibrium with the experiments #########################################
# exp_composition = {"SiO2": df_exp["SiO2"],
#                    "Al2O3": df_exp["Al2O3"],
#                    "TiO2": df_exp["TiO2"],
#                    "FeOT": df_exp["FeO"],
#                    "MgO": df_exp["MgO"],
#                    "CaO": df_exp["CaO"],
#                    "Na2O": df_exp["Na2O"],
#                    "K2O": df_exp["K2O"],
#                    "P2O5": df_exp["P2O5"],
#                    "MnO": df_exp["MnO"],
#                    }
#
# exp_phih2o = []
# exp_phih2s = []
# exp_phiso2 = []
#
# for i in range(0, 101):
#     exp_phi = Fugacity(pressure=df_exp["pressure"][i], temperature=df_exp["temperature"][i])
#     exp_phih2o.append(exp_phi.phiH2O)
#     exp_phiso2.append(exp_phi.phiSO2)
#     exp_phih2s.append(exp_phi.phiH2S)
# df_exp["phiH2O"] = exp_phih2o
# df_exp["phiH2S"] = exp_phih2s
# df_exp["phiSO2"] = exp_phiso2
# exp_fO2 = OxygenFugacity(df_exp["pressure"], df_exp["temperature"]+273.15, exp_composition)
# exp_fH2O = df_exp["XH2O_f"]*df_exp["pressure"] * 10 * df_exp["phiH2O"]
# df_exp["waterfugacity"] = exp_fH2O
# exp_SO2_ST = exp_fO2.gas_quilibrium(fo2=df_exp["fO2"], phiso2=df_exp["phiSO2"], phih2s=df_exp["phiH2S"], fh2o=exp_fH2O)
# df_exp["SO2/ST"] = exp_SO2_ST
# df_exp.to_csv("Exp_results.csv")
# print(dfmq)
