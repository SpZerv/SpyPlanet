import tkinter as tk
from tkinter import ttk
from tkinter import *


root = Tk()
root.geometry("1580x980")
root.title('SpyPlanet')
#root.iconbitmap(r'C:\Users\User\Downloads\Πτυχιακή\exe\Planet.ico')
root.configure(bg='black')


my_canvas = Canvas(root, width=1580, height=4, bg="black", bd=1, highlightthickness=0, relief=FLAT)
my_canvas.grid(row=4, column=0, columnspan=72, sticky=EW)
my_canvas.create_line(0, 4, 1580, 4, fill="red")

my_canvas = Canvas(root, width=1000, height=4, bg="black", bd=1, highlightthickness=0, relief=FLAT)
my_canvas.grid(row=30, column=0, columnspan=36, sticky=EW)
my_canvas.create_line(0, 4, 1000, 4, fill="red")


my_canvas = Canvas(root, width=4, height=900, bg="black", bd=1, highlightthickness=0, relief=FLAT)
my_canvas.grid(row=3, column=35, rowspan=67)
my_canvas.create_line(4, 5, 4, 900, fill="red")




import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


#Data downloading and tpf

myLabel1 = Label(root,text="Enter the name of the star and\n the mission:",font='Helvetica 12 bold',fg="red",bg="black")
myLabel1.grid(row=0, column=20, sticky=N)
myLabel1 = Label(root,text="Enter the author(pipeline):",font='Helvetica 12 bold',fg="red",bg="black")
myLabel1.grid(row=2, column=20, sticky=N)


star = Entry(root,width=20,bg="white")
star.grid(row=0, column=24)
star.get()
mission = Entry(root,width=20,bg="white")
mission.grid(row=0, column=30)
mission.get()
author = Entry(root,width=20,bg="white")
author.grid(row=2, column=24)
author.get()

from pandastable import Table
from astropy.table import Table

def Data_Mast():

    global search_result,target,miss
    target = star.get()
    miss = mission.get()

    search_result = lk.search_lightcurve(target, mission=miss, cadence="long")
    search_result1 = search_result.table.to_pandas()


    pop = Toplevel(root)
    pop.title("Data")
    pop.config(bg="black")


    frame = tk.Frame(pop)
    frame.grid(row=0, column=0)


    pt = Table(frame,dataframe=search_result1,showtoolbar=True,showstatusbar=True)
    pt.show()
    return search_result


r = IntVar()

def Rb():
    global r,v

    if r.get() == 1:
        v = "sap_flux"
    else:
        v = "pdcsap_flux"

    return v


Radiobutton(root,text="SAP",variable=r,value=1,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=1, column=24)
Radiobutton(root,text="PDCSAP",variable=r,value=2,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=1, column=30)

myButton3a = Button(root, text="Define the LC", padx=22, pady=5, command=Rb, fg="red", bg="white")
myButton3a.grid(row=1, column=40)


def Author_Pipe():
    global search_result,auth,lc_collection

    auth = author.get()

    search_result = lk.search_lightcurve(target, mission=miss, author=auth, cadence="long")
    lc_collection = search_result.download_all(flux_column=v)
    lc_collection.plot();
    plt.show()

    myLabel = Label(root,text=star.get() + ": Data \n Downloaded",font='Helvetica 10 bold',fg="red",bg="black")
    myLabel.grid(row=2, column=30, sticky=N)


myLabel9= Label(root,text="Photometry",font='Helvetica 25 bold',fg="red",bg="black")
myLabel9.grid(row=5, column=0)

quarter1 = Entry(root,width=10,bg="white")
quarter1.grid(row=11, column=20)
quarter1.get()

myLabel45 = Label(root,text="(Optional)",fg="white",bg="black")
myLabel45.grid(row=11, column=20, sticky=W)

def tpf_plot():
    quart = quarter1.get()

    if miss=="Kepler":
        tpf = lk.search_targetpixelfile(target, mission="Kepler", quarter=quart, cadence="long").download()
    else:
        tpf = lk.search_targetpixelfile(target, mission="TESS", sector=quart, cadence="long").download()
    tpf.plot()
    plt.show()



myLabel2 = Label(root,text="See the TPF of the star",fg="white",bg="black")
myLabel2.grid(row=11, column=0)

myButton1 = Button(root, text="Find Star", padx=35, pady=5, command=Data_Mast, fg="red", bg="white")
myButton1.grid(row=0, column=40, sticky=N)

myButton1 = Button(root, text="Find Author's LC", padx=14, pady=5, command=Author_Pipe, fg="red", bg="white")
myButton1.grid(row=2, column=40, sticky=N)

myButton2 = Button(root, text="TPF Plot", padx=31, pady=5, command=tpf_plot, fg="blue", bg="white")
myButton2.grid(row=11, column=10)


#Light Curve and analysis


q = IntVar()

def Rb2():
    global w,q

    if q.get() == 1:
        w = 1
    else:
        w = 2

    return w



Radiobutton(root,text="One Quarter",variable=q,value=1,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=10, column=20)
Radiobutton(root,text="All Quarters",variable=q,value=2,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=10, column=24)


myButton3g = Button(root, text="Define Quarter", padx=15, pady=5, command=Rb2, fg="blue", bg="white")
myButton3g.grid(row=10, column=10)

myLabel2 = Label(root,text="Choose the quarters you \n  want to download",fg="white",bg="black")
myLabel2.grid(row=10, column=0)




quarter2 = Entry(root,width=10,bg="white")
quarter2.grid(row=12, column=20)
quarter2.get()



def Light_Curve():

    global lc,QUAR,w,q,v
    QUAR = quarter2.get()
    target = star.get()


    if w == 1:
        if miss=="Kepler":
            lc = lk.search_lightcurve(target, mission="Kepler", author=auth, quarter=QUAR, cadence="long").download(flux_column=v)
        else:
            lc = lk.search_lightcurve(target, mission="TESS", author=auth, sector=QUAR, cadence="long").download(flux_column=v)
    else:
        lc = lc_collection.stitch()

    lc = lc.remove_nans()
    lc.plot()
    plt.show()

myLabel3 = Label(root,text="See the Light Curve of the star",fg="white",bg="black")
myLabel3.grid(row=12, column=0)

myButton3 = Button(root, text="Light Curve", padx=21, pady=5, command=Light_Curve, fg="blue", bg="white")
myButton3.grid(row=12, column=10)









from lightkurve.correctors import SFFCorrector

myLabel45 = Label(root,text="(Optional)",fg="white",bg="black")
myLabel45.grid(row=15, column=20, sticky=W)

def SFF():
    global lc

    lc = SFFCorrector(lc).correct(windows=20)
    lc.plot()
    plt.show()

myLabel3b = Label(root,text="Self Flat Fielding Corrector",fg="white",bg="black")
myLabel3b.grid(row=15, column=0)

myButton3b = Button(root, text="SFF", padx=42, pady=5, command=SFF, fg="blue", bg="white")
myButton3b.grid(row=15, column=10)



wind = Entry(root,width=10,bg="white")
wind.grid(row=16, column=20, columnspan=2)
wind.get()



def Flat_lc():

    wind2 = wind.get()

    global lc
    lc = lc.flatten(window_length=int(wind2))
    lc.plot()
    plt.show()

myLabel4 = Label(root,text="Flat the Light Curve ",fg="white",bg="black")
myLabel4.grid(row=16, column=0)

myButton4 = Button(root, text="Flatten", padx=33, pady=5, command=Flat_lc, fg="blue", bg="white")
myButton4.grid(row=16, column=10)





orio1 = Entry(root,width=10,bg="white")
orio1.grid(row=17, column=20, columnspan=1)
orio1.get()

orio2 = Entry(root,width=10,bg="white")
orio2.grid(row=17, column=20, columnspan=1, sticky=E)
orio2.get()





def Masked_lc():
    global lc

    or1 = orio1.get()
    or2 = orio2.get()

    mask = (lc.time.value < float(or1)) | (lc.time.value > float(or2))
    lc = lc[mask]
    lc.plot()
    plt.show()

    return lc

myLabel43 = Label(root,text="Cut part of the Light Curve ",fg="white",bg="black")
myLabel43.grid(row=17, column=0)

myButton4 = Button(root, text="Cut", padx=41, pady=5, command=Masked_lc, fg="blue", bg="white")
myButton4.grid(row=17, column=10)



import numpy as np

oriop = Entry(root,width=10,bg="white")
oriop.grid(row=19, column=20, columnspan=1)
oriop.get()



best_fit_period = 0.0
def Periodogram_bls():

    global best_fit_period
    global periodogram
    orp = oriop.get()

    periodogram = lc.to_periodogram(method="bls", period=np.arange(0.5, float(orp), 0.001))
    periodogram.plot();
    best_fit_period = periodogram.period_at_max_power
    best_fit_period2 = "{:.3f}".format(best_fit_period)
    plt.show()

    myLabel5b = Label(root,text="Best fit period: " + str(best_fit_period2),fg="white",bg="black")
    myLabel5b.grid(row=19, column=24)

    return best_fit_period

myLabel5 = Label(root,text="Periodogram with the BLS method",fg="white",bg="black")
myLabel5.grid(row=19, column=0)

myButton5 = Button(root, text="Periodogram", padx=18, pady=5, command=Periodogram_bls, fg="blue", bg="white")
myButton5.grid(row=19, column=10)


import astropy.units as u


def Fold_lc():
    global lc,best_fit_period,lc2

    if not Period_fold.get():
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
    else:
        Period_Dimensionless = Period_fold.get()
        best_fit_period = Period_Dimensionless*u.s*60.0 * 60.0 * 24.0
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();

    lc2.plot()
    plt.xlabel("Time [JD]")
    plt.show()





Period_fold = Entry(root,width=10,bg="white")
Period_fold.grid(row=20, column=20, columnspan=2)
Period_fold.get()

myLabel6 = Label(root,text="Fold the Light Curve",fg="white",bg="black")
myLabel6.grid(row=20, column=0)

myButton6 = Button(root, text="Fold", padx=41, pady=5, command=Fold_lc, fg="blue", bg="white")
myButton6.grid(row=20, column=10)



def Zoom_lc():
    global lc, zoom1, zoom2, lc2, best_fit_period

    zoom1 = zoomxl.get()
    zoom2 = zoomxr.get()

    lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
    lc2.plot()
    lc2.set_xlim(float(zoom1), float(zoom2))
    plt.xlabel("Time [JD]")
    plt.show()
    return zoom1,zoom2






zoomxl = Entry(root,width=10,bg="white")
zoomxl.grid(row=22, column=20)
zoomxl.get()


zoomxr = Entry(root,width=10,bg="white")
zoomxr.grid(row=22, column=20, sticky=E)
zoomxr.get()

myLabel7 = Label(root,text="Zoom in the Light Curve",fg="white",bg="black")
myLabel7.grid(row=22, column=0)

myButton7 = Button(root, text="Zoom In", padx=30, pady=5, command=Zoom_lc, fg="blue", bg="white")
myButton7.grid(row=22, column=10)





##############################################################################################################
##############################################################################################################


#Exoplanet MCMC


myLabel9= Label(root,text="MCMC",font='Helvetica 25 bold',fg="red",bg="black")
myLabel9.grid(row=5, column=40)


import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares


quarter = Entry(root,width=10,bg="white")
quarter.grid(row=10, column=60)
quarter.get()

def Prepare():

    global x,y,bls_t0,bls_period,bls_depth,texp,miss,quart,target

    quart = quarter.get()
    target = star.get()

    if miss == "Kepler":
        lc_file = lk.search_lightcurve(target, author=auth, cadence="long", quarter=quart
                      ).download(quality_bitmask="hardest", flux_column="pdcsap_flux")

    else:
        lc_file = lk.search_lightcurve(target, author=auth, cadence="long", sector=quart
                      ).download(quality_bitmask="hardest", flux_column="pdcsap_flux")

    lc_file.plot()
    plt.show()

    lc = lc_file.remove_nans().normalize()
    time = lc.time.value
    flux = lc.flux
    m = lc.quality == 0
    with fits.open(lc_file.filename) as hdu:
        hdr = hdu[1].header

    texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
    texp /= 60.0 * 60.0 * 24.0

    ref_time = 0.5 * (np.min(time) + np.max(time))
    x = np.ascontiguousarray(time[m] - ref_time, dtype=np.float64)
    y = np.ascontiguousarray(1e3 * (flux[m] - 1.0), dtype=np.float64) + 1



    period_grid = np.exp(np.linspace(np.log(1), np.log(15), 50000))

    bls = BoxLeastSquares(x, y)
    bls_power = bls.power(period_grid, 0.1, oversample=20)

    # Save the highest peak as the planet candidate
    index = np.argmax(bls_power.power)
    bls_period = bls_power.period[index]
    bls_t0 = bls_power.transit_time[index]
    bls_depth = bls_power.depth[index]
    transit_mask = bls.transit_mask(x, bls_period, 0.2, bls_t0)

    # Plot the folded transit

    x_fold = (x - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period
    m = np.abs(x_fold) < 0.4




myLabel9b= Label(root,text="Choose quarter and \n prepare the Data for MCMC ",fg="white",bg="black")
myLabel9b.grid(row=10, column=40)


myButton9b = Button(root, text="Prepare Data", padx=34, pady=5, command=Prepare, fg="blue", bg="white")
myButton9b.grid(row=10, column=50)




import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt

import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess

#βαλε μεταβλητες για stellar parameters



global map_soln,map_soln0,model,model0,mask,y,x,build_model


nn = IntVar()
cb1 = Checkbutton(root, text="None", variable=nn,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=15, column=50)

ee = IntVar()
cb2 = Checkbutton(root, text="Eccentricity", variable=ee,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=15, column=60)

bb = IntVar()
cb3 = Checkbutton(root, text="Impact \n Parameter", variable=bb,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=15, column=70)

########

bbb = IntVar()
cb4 = Checkbutton(root, text="Impact Parameter", variable=bbb,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=16, column=50)

ppp = IntVar()
cb5 = Checkbutton(root, text="Planet Period(Days)",bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black", variable=ppp).grid(row=17, column=50)

rrr = IntVar()
cb6 = Checkbutton(root, text="Planet Radius(Rsun)",bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black", variable=rrr).grid(row=18, column=50)

eee = IntVar()
cb5 = Checkbutton(root, text="Eccentricity",bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black", variable=eee).grid(row=19, column=50)

ooo = IntVar()
cb6 = Checkbutton(root, text="Omega",bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black", variable=ooo).grid(row=20, column=50)


def Optim1():
    global map_soln,map_soln0,model,model0,mask,y,x,build_model

    rstar6 = rstar5.get()
    mstar6 = mstar5.get()
    #####
    bmc6 = bmc.get()
    permc6 = permc.get()
    rplmc6 = rplmc.get()
    eccnmc6 = eccnmc.get()
    ommc6 = ommc.get()


    if nn.get() == 1:

        global map_soln,map_soln0,model,model0,mask,y,x,build_model

        def build_model(mask=None, start=None):
            if mask is None:
                mask = np.ones(len(x), dtype=bool)
            with pm.Model() as model:

                if bbb.get() == 1:
                    b = float(bmc6)
                # Parameters for the stellar properties
                mean = pm.Normal("mean", mu=0.0, sd=10.0)
                u_star = xo.QuadLimbDark("u_star")

                # Stellar parameters
                M_star_huang = float(mstar6), 0.06
                R_star_huang = float(rstar6), 0.06
                BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
                m_star = BoundedNormal("m_star", mu=M_star_huang[0], sd=M_star_huang[1])
                r_star = BoundedNormal("r_star", mu=R_star_huang[0], sd=R_star_huang[1])

                # Orbital parameters for the planets
                t0 = pm.Normal("t0", mu=bls_t0, sd=1)
                log_period = pm.Normal("log_period", mu=np.log(bls_period), sd=1)
                log_r_pl = pm.Normal(
                    "log_r_pl",
                    sd=1.0,
                    mu=0.5 * np.log(1e-3 * np.array(bls_depth))
                    + np.log(R_star_huang[0]),
                )
                period = pm.Deterministic("period", tt.exp(log_period))
                r_pl = pm.Deterministic("r_pl", tt.exp(log_r_pl))
                ror = pm.Deterministic("ror", r_pl / r_star)
                b = xo.distributions.ImpactParameter("b", ror=ror)



                ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
                ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
                omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
                xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)

                # Transit jitter & GP parameters
                log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10)
                log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
                log_sigma_gp = pm.Normal("log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10)

                ##########

                if ppp.get() == 1:
                    period = float(permc6)

                if rrr.get() == 1:
                    r_pl = float(rplmc6)

                if eee.get() == 1:
                    ecc = float(eccnmc6)

                if ooo.get() == 1:
                    omega = float(ommc6)

                # Orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=r_star,
                    m_star=m_star,
                    period=period,
                    t0=t0,
                    b=b,
                    ecc=ecc,
                    omega=omega,
                )

                # Compute the model light curve
                light_curves = pm.Deterministic(
                    "light_curves",
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit, r=r_pl, t=x[mask], texp=texp
                    )
                    * 1e3,
                )
                light_curve = tt.sum(light_curves, axis=-1) + mean
                resid = y[mask] - light_curve

                # GP model for the light curve
                kernel = terms.SHOTerm(
                    sigma=tt.exp(log_sigma_gp),
                    rho=tt.exp(log_rho_gp),
                    Q=1 / np.sqrt(2),
                )
                gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
                gp.marginal("gp", observed=resid)
                pm.Deterministic("gp_pred", gp.predict(resid))

                # Fit for the maximum a posteriori parameters, I've found that I can get
                # a better solution by trying different combinations of parameters in turn
                if start is None:
                    start = model.test_point
                map_soln = pmx.optimize(
                    start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])
                map_soln = pmx.optimize(start=map_soln, vars=[b])
                map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
                map_soln = pmx.optimize(start=map_soln, vars=[u_star])
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])
                map_soln = pmx.optimize(start=map_soln, vars=[b])
                map_soln = pmx.optimize(start=map_soln, vars=[ecs])
                map_soln = pmx.optimize(start=map_soln, vars=[mean])
                map_soln = pmx.optimize(
                    start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln)

            return model, map_soln


        model0, map_soln0 = build_model()



    elif ee.get() == 1 and bb.get() == 1:



        def build_model(mask=None, start=None):
            if mask is None:
                mask = np.ones(len(x), dtype=bool)
            with pm.Model() as model:

                # Parameters for the stellar properties
                mean = pm.Normal("mean", mu=0.0, sd=10.0)
                u_star = xo.QuadLimbDark("u_star")

                # Stellar parameters
                M_star_huang = float(mstar6), 0.06
                R_star_huang = float(rstar6), 0.06
                BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
                m_star = BoundedNormal("m_star", mu=M_star_huang[0], sd=M_star_huang[1])
                r_star = BoundedNormal("r_star", mu=R_star_huang[0], sd=R_star_huang[1])

                # Orbital parameters for the planets
                t0 = pm.Normal("t0", mu=bls_t0, sd=1)
                log_period = pm.Normal("log_period", mu=np.log(bls_period), sd=1)
                log_r_pl = pm.Normal(
                    "log_r_pl",
                    sd=1.0,
                    mu=0.5 * np.log(1e-3 * np.array(bls_depth))
                    + np.log(R_star_huang[0]),
                )
                period = pm.Deterministic("period", tt.exp(log_period))
                r_pl = pm.Deterministic("r_pl", tt.exp(log_r_pl))
                ror = pm.Deterministic("ror", r_pl / r_star)




                # Transit jitter & GP parameters
                log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10)
                log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
                log_sigma_gp = pm.Normal("log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10)

                ##########

                if ppp.get() == 1:
                    period = float(permc6)

                if rrr.get() == 1:
                    r_pl = float(rplmc6)


                # Orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=r_star,
                    m_star=m_star,
                    period=period,
                    t0=t0,
                )

                # Compute the model light curve
                light_curves = pm.Deterministic(
                    "light_curves",
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit, r=r_pl, t=x[mask], texp=texp
                    )
                    * 1e3,
                )
                light_curve = tt.sum(light_curves, axis=-1) + mean
                resid = y[mask] - light_curve

                # GP model for the light curve
                kernel = terms.SHOTerm(
                    sigma=tt.exp(log_sigma_gp),
                    rho=tt.exp(log_rho_gp),
                    Q=1 / np.sqrt(2),
                )
                gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
                gp.marginal("gp", observed=resid)
                pm.Deterministic("gp_pred", gp.predict(resid))

                # Fit for the maximum a posteriori parameters, I've found that I can get
                # a better solution by trying different combinations of parameters in turn
                if start is None:
                    start = model.test_point
                map_soln = pmx.optimize(
                    start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])

                map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
                map_soln = pmx.optimize(start=map_soln, vars=[u_star])
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])


                map_soln = pmx.optimize(start=map_soln, vars=[mean])
                map_soln = pmx.optimize(
                    start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln)

            return model, map_soln


        model0, map_soln0 = build_model()



    elif ee.get() == 1 and bb.get() == 0:



        def build_model(mask=None, start=None):
            if mask is None:
                mask = np.ones(len(x), dtype=bool)
            with pm.Model() as model:

                if bbb.get() == 1:
                    b = float(bmc6)
                # Parameters for the stellar properties
                mean = pm.Normal("mean", mu=0.0, sd=10.0)
                u_star = xo.QuadLimbDark("u_star")

                # Stellar parameters
                M_star_huang = float(mstar6), 0.06
                R_star_huang = float(rstar6), 0.06
                BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
                m_star = BoundedNormal("m_star", mu=M_star_huang[0], sd=M_star_huang[1])
                r_star = BoundedNormal("r_star", mu=R_star_huang[0], sd=R_star_huang[1])

                # Orbital parameters for the planets
                t0 = pm.Normal("t0", mu=bls_t0, sd=1)
                log_period = pm.Normal("log_period", mu=np.log(bls_period), sd=1)
                log_r_pl = pm.Normal(
                    "log_r_pl",
                    sd=1.0,
                    mu=0.5 * np.log(1e-3 * np.array(bls_depth))
                    + np.log(R_star_huang[0]),
                )
                period = pm.Deterministic("period", tt.exp(log_period))
                r_pl = pm.Deterministic("r_pl", tt.exp(log_r_pl))
                ror = pm.Deterministic("ror", r_pl / r_star)
                b = xo.distributions.ImpactParameter("b", ror=ror)



                # Transit jitter & GP parameters
                log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10)
                log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
                log_sigma_gp = pm.Normal("log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10)

                ##########

                if ppp.get() == 1:
                    period = float(permc6)

                if rrr.get() == 1:
                    r_pl = float(rplmc6)

                # Orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=r_star,
                    m_star=m_star,
                    period=period,
                    t0=t0,
                    b=b,

                )

                # Compute the model light curve
                light_curves = pm.Deterministic(
                    "light_curves",
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit, r=r_pl, t=x[mask], texp=texp
                    )
                    * 1e3,
                )
                light_curve = tt.sum(light_curves, axis=-1) + mean
                resid = y[mask] - light_curve

                # GP model for the light curve
                kernel = terms.SHOTerm(
                    sigma=tt.exp(log_sigma_gp),
                    rho=tt.exp(log_rho_gp),
                    Q=1 / np.sqrt(2),
                )
                gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
                gp.marginal("gp", observed=resid)
                pm.Deterministic("gp_pred", gp.predict(resid))

                # Fit for the maximum a posteriori parameters, I've found that I can get
                # a better solution by trying different combinations of parameters in turn
                if start is None:
                    start = model.test_point
                map_soln = pmx.optimize(
                    start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])
                map_soln = pmx.optimize(start=map_soln, vars=[b])
                map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
                map_soln = pmx.optimize(start=map_soln, vars=[u_star])
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])
                map_soln = pmx.optimize(start=map_soln, vars=[b])

                map_soln = pmx.optimize(start=map_soln, vars=[mean])
                map_soln = pmx.optimize(
                    start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln)

            return model, map_soln


        model0, map_soln0 = build_model()



    elif bb.get()  == 1 and ee.get() == 0:


        def build_model(mask=None, start=None):
            if mask is None:
                mask = np.ones(len(x), dtype=bool)
            with pm.Model() as model:

                # Parameters for the stellar properties
                mean = pm.Normal("mean", mu=0.0, sd=10.0)
                u_star = xo.QuadLimbDark("u_star")

                # Stellar parameters
                M_star_huang = float(mstar6), 0.06
                R_star_huang = float(rstar6), 0.06
                BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
                m_star = BoundedNormal("m_star", mu=M_star_huang[0], sd=M_star_huang[1])
                r_star = BoundedNormal("r_star", mu=R_star_huang[0], sd=R_star_huang[1])

                # Orbital parameters for the planets
                t0 = pm.Normal("t0", mu=bls_t0, sd=1)
                log_period = pm.Normal("log_period", mu=np.log(bls_period), sd=1)
                log_r_pl = pm.Normal(
                    "log_r_pl",
                    sd=1.0,
                    mu=0.5 * np.log(1e-3 * np.array(bls_depth))
                    + np.log(R_star_huang[0]),
                )
                period = pm.Deterministic("period", tt.exp(log_period))
                r_pl = pm.Deterministic("r_pl", tt.exp(log_r_pl))
                ror = pm.Deterministic("ror", r_pl / r_star)


                ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
                ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
                omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
                xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)

                # Transit jitter & GP parameters
                log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10)
                log_rho_gp = pm.Normal("log_rho_gp", mu=0, sd=10)
                log_sigma_gp = pm.Normal("log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10)

                ##########
                if ppp.get() == 1:
                    period = float(permc6)

                if rrr.get() == 1:
                    r_pl = float(rplmc6)

                if eee.get() == 1:
                    ecc = float(eccnmc6)

                if ooo.get() == 1:
                    omega = float(ommc6)

                # Orbit model
                orbit = xo.orbits.KeplerianOrbit(
                    r_star=r_star,
                    m_star=m_star,
                    period=period,
                    t0=t0,
                    ecc=ecc,
                    omega=omega,
                )

                # Compute the model light curve
                light_curves = pm.Deterministic(
                    "light_curves",
                    xo.LimbDarkLightCurve(u_star).get_light_curve(
                        orbit=orbit, r=r_pl, t=x[mask], texp=texp
                    )
                    * 1e3,
                )
                light_curve = tt.sum(light_curves, axis=-1) + mean
                resid = y[mask] - light_curve

                # GP model for the light curve
                kernel = terms.SHOTerm(
                    sigma=tt.exp(log_sigma_gp),
                    rho=tt.exp(log_rho_gp),
                    Q=1 / np.sqrt(2),
                )
                gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
                gp.marginal("gp", observed=resid)
                pm.Deterministic("gp_pred", gp.predict(resid))

                # Fit for the maximum a posteriori parameters, I've found that I can get
                # a better solution by trying different combinations of parameters in turn
                if start is None:
                    start = model.test_point
                map_soln = pmx.optimize(
                    start=start, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])

                map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
                map_soln = pmx.optimize(start=map_soln, vars=[u_star])
                map_soln = pmx.optimize(start=map_soln, vars=[log_r_pl])

                map_soln = pmx.optimize(start=map_soln, vars=[ecs])
                map_soln = pmx.optimize(start=map_soln, vars=[mean])
                map_soln = pmx.optimize(
                    start=map_soln, vars=[log_sigma_lc, log_sigma_gp, log_rho_gp]
                )
                map_soln = pmx.optimize(start=map_soln)

            return model, map_soln


        model0, map_soln0 = build_model()







rstar5 = Entry(root,width=20,bg="white")
rstar5.grid(row=11, column=50)
rstar5.get()

mstar5 = Entry(root,width=20,bg="white")
mstar5.grid(row=12, column=50)
mstar5.get()

myLabel10a = Label(root,text="Radius of Star(Rsun)",fg="white",bg="black")
myLabel10a.grid(row=11, column=40)

myLabel0c = Label(root,text="Mass of Star(Msun)",fg="white",bg="black")
myLabel0c.grid(row=12, column=40)


myLabel10b= Label(root,text="Define some parameters as zero",fg="white",bg="black")
myLabel10b.grid(row=15, column=40)

####### Σταθερες γιa MCMC

bmc = Entry(root,width=20,bg="white")
bmc.grid(row=16, column=60)
bmc.get()

permc = Entry(root,width=20,bg="white")
permc.grid(row=17, column=60)
permc.get()

rplmc = Entry(root,width=20,bg="white")
rplmc.grid(row=18, column=60)
rplmc.get()

eccnmc = Entry(root,width=20,bg="white")
eccnmc.grid(row=19, column=60)
eccnmc.get()

ommc = Entry(root,width=20,bg="white")
ommc.grid(row=20, column=60)
ommc.get()


myLabel22 = Label(root,text="Define constants for the MCMC",fg="white",bg="black")
myLabel22.grid(row=16, column=40)



######

myLabel10= Label(root,text="Optimize your parameters \n through Gaussian Process",fg="white",bg="black")
myLabel10.grid(row=22, column=40)


myButton10 = Button(root, text="Optimization 1", padx=27, pady=5, command=Optim1, fg="blue", bg="white")
myButton10.grid(row=22, column=50)




def Opt_Plot():

    global map_soln,map_soln0,model,model0,mask,y,x,plot_light_curve,gp_mod

    def plot_light_curve(soln, mask=None):
        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        ax = axes[0]
        ax.plot(x[mask], y[mask], "k", label="data")
        gp_mod = soln["gp_pred"] + soln["mean"]
        ax.plot(x[mask], gp_mod, color="C2", label="gp model")
        ax.legend(fontsize=10)
        ax.set_ylabel("relative flux [ppt]")

        ax = axes[1]
        ax.plot(x[mask], y[mask] - gp_mod, "k", label="de-trended data")
        for i, l in enumerate("b"):
            mod = soln["light_curves"][:, i]
            ax.plot(x[mask], mod, label="planet {0}".format(l))
        ax.legend(fontsize=10, loc=3)
        ax.set_ylabel("de-trended flux [ppt]")

        ax = axes[2]
        mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
        ax.plot(x[mask], y[mask] - mod, "k")
        ax.axhline(0, color="#aaaaaa", lw=1)
        ax.set_ylabel("residuals [ppt]")
        ax.set_xlim(x[mask].min(), x[mask].max())
        ax.set_xlabel("time [days]")

        return fig


    _ = plot_light_curve(map_soln0)
    plt.show()


myLabel11= Label(root,text="Detrend the curve and plot it\n with the optimized parameters",fg="white",bg="black")
myLabel11.grid(row=30, column=40)


myButton11 = Button(root, text="Optimized Lightcurve ", padx=7, pady=5, command=Opt_Plot, fg="blue", bg="white")
myButton11.grid(row=30, column=50)



def Outliers():

    global map_soln,map_soln0,model,model0,mask,y,x

    mod = (
        map_soln0["gp_pred"]
        + map_soln0["mean"]
        + np.sum(map_soln0["light_curves"], axis=-1)
    )
    resid = y - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask = np.abs(resid) < 5 * rms

    plt.figure(figsize=(10, 5))
    plt.plot(x, resid, "k", label="data")
    plt.plot(x[~mask], resid[~mask], "xr", label="outliers")
    plt.axhline(0, color="#aaaaaa", lw=1)
    plt.ylabel("residuals [ppt]")
    plt.xlabel("time [days]")
    plt.legend(fontsize=12, loc=3)
    _ = plt.xlim(x.min(), x.max())
    plt.show()


myLabel12= Label(root,text="Remove the outliers from your Data",fg="white",bg="black")
myLabel12.grid(row=35, column=40)


myButton12 = Button(root, text="Remove Outliers", padx=22, pady=5, command=Outliers, fg="blue", bg="white")
myButton12.grid(row=35, column=50)



def Optim2():

    global map_soln,map_soln0,model,model0,mask,y,x,build_model,plot_light_curve

    model, map_soln = build_model(mask, map_soln0)
    _ = plot_light_curve(map_soln, mask)
    plt.show()

myLabel13= Label(root,text="Re-Optimize your parameters \nwithout the outliers",fg="white",bg="black")
myLabel13.grid(row=40, column=40)


myButton13 = Button(root, text="Optimization 2", padx=27, pady=5, command=Optim2, fg="blue", bg="white")
myButton13.grid(row=40, column=50)


def Sampling_MCMC():

    global map_soln,map_soln0,model,model0,mask,y,x,trace,omega,ecc,r_pl,b,period,r_star,u_star,m_star,t0,mean

    np.random.seed(4271579)
    with model:
        trace = pmx.sample(
            tune=2500,
            draws=2000,
            start=map_soln,
            cores=2,
            chains=2,
            initial_accept=0.8,
            target_accept=0.96,
            return_inferencedata=True,
            progressbar=True,
        )




myLabel14= Label(root,text="Run the Sampling for the MCMC",fg="white",bg="black")
myLabel14.grid(row=41, column=40)

myButton14 = Button(root, text="Sampling", padx=41, pady=5, command=Sampling_MCMC, fg="blue", bg="white")
myButton14.grid(row=41, column=50)


import arviz as az
from pandastable import Table

def Summary_MCMC():
    global pop,arsum

    if nn.get() == 1:
        arsum = az.summary(trace,var_names=["omega","ecc","r_pl","b","t0","period","r_star","m_star","u_star","mean"])

    elif ee.get() == 1 and bb.get() == 1:
        arsum = az.summary(trace,var_names=["r_pl","t0","period","r_star","m_star","u_star","mean"])

    elif ee.get() == 1 and bb.get() == 0:
        arsum = az.summary(trace,var_names=["r_pl","b","t0","period","r_star","m_star","u_star","mean"])

    elif ee.get() == 0 and bb.get() == 1:
        arsum = az.summary(trace,var_names=["omega","ecc","r_pl","t0","period","r_star","m_star","u_star","mean"])

    pop = Toplevel(root)
    pop.title("MCMC Summary")
    pop.config(bg="black")


    frame = tk.Frame(pop)
    frame.grid(row=0, column=0)

    pt = Table(frame,dataframe=arsum,showtoolbar=True,showstatusbar=True)
    pt.show()

    arsum2 = str(arsum)

    myLabel15b= Label(pop,text=arsum2 + " ",fg="white",bg="black")
    myLabel15b.grid(row=0, column=10)
    plt.show()
    return arsum

myLabel15= Label(root,text="See your parameters after the MCMC",fg="white",bg="black")
myLabel15.grid(row=42, column=40)


myButton15 = Button(root, text="Summary", padx=40, pady=5, command=Summary_MCMC, fg="blue", bg="white")
myButton15.grid(row=42, column=50)



def Final_Plot():

    global map_soln,map_soln0,model,model0,mask,y,x,trace,x_fold,gp_mod,p

    flat_samps = trace.posterior.stack(sample=("chain", "draw"))

    # Compute the GP prediction
    gp_mod = np.median(flat_samps["gp_pred"].values + flat_samps["mean"].values[None, :], axis=-1)

    # Get the posterior median orbital parameters
    p = np.median(flat_samps["period"])
    t0 = np.median(flat_samps["t0"])


    plt.figure(figsize=(8, 8))

    # Plot the folded data
    x_fold = (x[mask] - t0 + 0.5 * p) % p - 0.5 * p
    plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)

    # Overplot the phase binned light curve
    bins = np.linspace(-0.41, 0.41, 50)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=y[mask])
    denom[num == 0] = 1.0
    plt.plot(0.5 * (bins[1:] + bins[:-1]), 1+(num / denom-1)/1e3, "o", color="C2", label="binned")

    # Plot the folded model
    inds = np.argsort(x_fold)
    inds = inds[np.abs(x_fold)[inds] < 0.3]
    pred = np.percentile(flat_samps["light_curves"][inds, 0], [16, 50, 84], axis=-1)
    plt.plot(x_fold[inds], 1+(pred[1])/1e3, color="C1", label="model")


    # Annotate the plot with the planet's period
    txt = "period = {0:.5f} +/- {1:.5f} d".format(np.mean(flat_samps["period"].values), np.std(flat_samps["period"].values))

    plt.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    plt.legend(fontsize=10, loc=4)
    plt.xlim(-0.5 * p/6, 0.5 * p/6)
    plt.xlabel("time since transit [days]")
    plt.ylabel("de-trended flux")

    plt.show()
    return gp_mod,p

myLabel16 = Label(root,text="Plot your final model",fg="white",bg="black")
myLabel16.grid(row=43, column=40)


myButton16 = Button(root, text="Model Plot", padx=37, pady=5, command=Final_Plot, fg="blue", bg="white")
myButton16.grid(row=43, column=50)



import corner
import astropy.units as u
def Corner_Plot():

    global map_soln,map_soln0,model,model0,mask,y,x,trace

    trace.posterior["r_earth"] = (trace.posterior["r_pl"].coords,(trace.posterior["r_pl"].values * u.R_sun).to(u.R_earth).value)


    if nn.get() == 1:

            _ = corner.corner(
                trace,
                var_names=["period", "r_earth", "b", "ecc","omega"],
                labels=[
                    "period [days]",
                    "radius [Earth radii]",
                    "impact param",
                    "eccentricity",
                    "omega"
                ],
            )



    elif ee.get() == 1 and bb.get() == 1:

            _ = corner.corner(
                trace,
                var_names=["period", "r_earth", "r_star"],
                labels=[
                    "period [days]",
                    "radius [Earth radii]",
                    "star radius"

                ],
            )



    elif ee.get() == 1 and bb.get() == 0:

            _ = corner.corner(
                trace,
                var_names=["period", "r_earth", "b","r_star"],
                labels=[
                    "period [days]",
                    "radius [Earth radii]",
                    "impact param",
                    "star radius"
                ],
            )




    elif ee.get() == 0 and bb.get() == 1:

            _ = corner.corner(
                trace,
                var_names=["period", "r_earth","r_star", "ecc","omega"],
                labels=[
                    "period [days]",
                    "radius [Earth radii]",
                    "star radius",
                    "eccentricity",
                    "omega"
                ],
            )

    plt.show()




myLabel17 = Label(root,text="Plot the corner plot of your parameters",fg="white",bg="black")
myLabel17.grid(row=44, column=40)


myButton17 = Button(root, text="Corner Plot", padx=36, pady=5, command=Corner_Plot, fg="blue", bg="white")
myButton17.grid(row=44, column=50)



myLabel18g = Label(root,text="           \n                ",fg="blue",bg="black")
myLabel18g.grid(row=46, column=40)

myLabel19g = Label(root,text="           \n                ",fg="blue",bg="black")
myLabel19g.grid(row=47, column=40)

myLabel20g = Label(root,text="           \n                ",fg="blue",bg="black")
myLabel20g.grid(row=48, column=40)




##############################################################################################################
##############################################################################################################


import numpy as np
import exoplanet as xo


myLabel9= Label(root,text="Manual Simulation",font='Helvetica 20 bold',fg="red",bg="black")
myLabel9.grid(row=35, column=0)

s = IntVar(0)

def Rb():
    global s,v

    if s.get() == 1:
        v = 1
    elif s.get() == 2:
        v = 2
    else:
        v = 3

    return v


Radiobutton(root,text="Simulated model",variable=s,value=1,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=40, column=20)

Radiobutton(root,text="Simulated model \nwith observed data",variable=s,value=2,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=40, column=24)

Radiobutton(root,text="Simulated model \nwith data after MCMC",variable=s,value=3,bg="black",fg="white",
                  activebackground='white', activeforeground='white',selectcolor="black").grid(row=40, column=30)

myButton3a = Button(root, text="Define the Model", padx=22, pady=5, command=Rb, fg="blue", bg="white")
myButton3a.grid(row=41, column=24)







def Zografia():
    global v,rstar2,b2,per2,sma2,rpl2,eccn2,om2,u12,u22,zoom1,zoom2,t02



    rstar2 = rstar.get()
    b2 = b1.get()
    per2 = per.get()
    sma2 = sma.get()
    rpl2 = rpl.get()
    eccn2 = eccn.get()
    om2 = om.get()
    u12 = u1.get()
    u22 = u2.get()
    t02 = t0.get()

    if v==1:

        orbit = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=float(rstar2), b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))

        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)
        u = [float(u12), float(u22)]
        light_curve = (xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=float(rpl2), t=t).eval())

        plt.plot(t, light_curve + 1, color="C0", lw=2)
        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)
        plt.show()

    elif v==2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")

        orbit = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=float(rstar2), b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))

        t = np.linspace(float(zoom1), float(zoom2), 1000)
        u = [float(u12), float(u22)]
        light_curve = (xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=float(rpl2), t=t).eval())

        plt.plot(t, light_curve + 1, color="C0", lw=2)
        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))
        plt.show()

    elif v==3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p, 0.5 * p)
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)


        orbit = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=float(rstar2), b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))

        t = np.linspace(-0.5 * p, 0.5 * p, 1000)
        u = [float(u12), float(u22)]
        light_curve = (xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=float(rpl2), t=t).eval())

        plt.plot(t, light_curve + 1, color="C0", lw=2)
        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)
        plt.show()
    return v,rstar2,b2,per2,sma2,rpl2,eccn2,om2,u12,u22,zoom1,zoom2




rstar = Entry(root,width=10,bg="white")
rstar.grid(row=40, column=10, sticky=W)
rstar.get()

b1 = Entry(root,width=10,bg="white")
b1.grid(row=41, column=10, sticky=W)
b1.get()

per = Entry(root,width=10,bg="white")
per.grid(row=42, column=10, sticky=W)
per.get()

sma = Entry(root,width=10,bg="white")
sma.grid(row=43, column=10, sticky=W)
sma.get()

rpl = Entry(root,width=10,bg="white")
rpl.grid(row=44, column=10, sticky=W)
rpl.get()

eccn = Entry(root,width=10,bg="white")
eccn.grid(row=45, column=10, sticky=W)
eccn.get()

om = Entry(root,width=10,bg="white")
om.grid(row=46, column=10, sticky=W)
om.get()

u1 = Entry(root,width=10,bg="white")
u1.grid(row=47, column=10, sticky=W)
u1.get()

u2 = Entry(root,width=10,bg="white")
u2.grid(row=48, column=10, sticky=W)
u2.get()

t0 = Entry(root,width=10,bg="white")
t0.grid(row=42, column=24)
t0.get()

myLabel8a = Label(root,text="Radius of Star(Rsun)",fg="white",bg="black")
myLabel8a.grid(row=40, column=0)

myLabel8b = Label(root,text="Impact Parameter",fg="white",bg="black")
myLabel8b.grid(row=41, column=0)

myLabel8c = Label(root,text="Planet Period(Days)",fg="white",bg="black")
myLabel8c.grid(row=42, column=0)

myLabel8d = Label(root,text="Semi-major Axis(Rsun)",fg="white",bg="black")
myLabel8d.grid(row=43, column=0)

myLabel8e = Label(root,text="Radius of Planet(Rsun)",fg="white",bg="black")
myLabel8e.grid(row=44, column=0)

myLabel8f = Label(root,text="Eccentricity",fg="white",bg="black")
myLabel8f.grid(row=45, column=0)

myLabel8g = Label(root,text="Omega(radians)",fg="white",bg="black")
myLabel8g.grid(row=46, column=0)

myLabel8h = Label(root,text="Limb Darkening Coeff 1",fg="white",bg="black")
myLabel8h.grid(row=47, column=0)

myLabel8i = Label(root,text="Limb Darkening Coeff 2",fg="white",bg="black")
myLabel8i.grid(row=48, column=0)

myLabel8a = Label(root,text="Time of Reference(Days)",fg="white",bg="black")
myLabel8a.grid(row=42, column=20)


myButton8 = Button(root, text="Exoplanet Model", padx=30, pady=5, command=Zografia, fg="blue", bg="white")
myButton8.grid(row=43, column=24)




z = 1;
def show(event):
    global z

    if clicked.get() == "Planet Radius":
        z = 1
    elif clicked.get() == "Star Radius" :
        z = 2
    elif clicked.get() == "Impact Parameter" :
        z = 3
    elif clicked.get() == "Semi-Major Axis":
        z = 4
    elif clicked.get() == "Eccentricity" :
        z = 5
    elif clicked.get() == "Omega":
        z = 6

    return z

options = ["Planet Radius", "Star Radius", "Impact Parameter", "Semi-Major Axis", "Eccentricity", "Omega"]

clicked = StringVar()
clicked.set(options[0])

drop = OptionMenu(root, clicked, *options, command=show)
drop.grid(row=44, column=24)


myLabel90 = Label(root,text="Choose a parameter to see \n how it affects the shape of the LC",fg="white",bg="black")






myLabel4 = Label(root,text="First value,last value \n and step of the parameter",fg="white",bg="black")
myLabel4.grid(row=45, column=20)

param1 = Entry(root,width=10,bg="white")
param1.grid(row=45, column=24, columnspan=1, sticky=W)
param1.get()

param2 = Entry(root,width=10,bg="white")
param2.grid(row=45, column=24, columnspan=1)
param2.get()

param3 = Entry(root,width=10,bg="white")
param3.grid(row=45, column=24, columnspan=1,sticky=E)
param3.get()


import exoplanet as xo
import numpy as np
import matplotlib.pyplot as plt

def Zog():
    global v,rstar2,b2,per2,sma2,rpl2,eccn2,om2,u12,u22,zoom1,zoom2

    par1 = float(param1.get())*100;
    par2 = float(param2.get())*100;
    par3 = float(param3.get())*100;

    if z == 1 and v == 1:
        u = [float(u12), float(u22)]

        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)

        for r_pl in np.arange(par1, par2+1, par3):
            r_pl=r_pl/100;

            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),a=float(sma2),r_star=float(rstar2),b=float(b2),
                                     ecc=float(eccn2),omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1,r=r_pl,t=t).eval()
            plt.plot(t, light_curve+1, label=f"$r_pl={r_pl}$")



        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 1 and v == 2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")


        u = [float(u12), float(u22)]
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for r_pl in np.arange(par1, par2+1, par3):
            r_pl=r_pl/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=float(rstar2), b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=r_pl, t=t).eval()
            plt.plot(t, light_curve+1, label=f"$r_pl={r_pl}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 1 and v == 3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p/6, 0.5 * p/6)


        u = [float(u12), float(u22)]
        t = np.linspace(-0.5 * p, 0.5 * p, 1000)


        for r_pl in np.arange(par1, par2+1, par3):
            r_pl=r_pl/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=float(rstar2), b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=r_pl, t=t).eval()
            plt.plot(t, light_curve+1, label=f"$r_pl={r_pl}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

################################################################################################r_star
    elif z == 2 and v == 1:
        u = [float(u12), float(u22)]
        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for r_star in np.arange(par1, par2+1, par3):
            r_star=r_star/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=r_star,  b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$r_star={r_star}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 2 and v == 2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")


        u = [float(u12), float(u22)]
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for r_star in np.arange(par1, par2+1, par3):
            r_star=r_star/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=r_star,  b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$r_star={r_star}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 2 and v == 3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p, 0.5 * p)
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)




        u = [float(u12), float(u22)]
        t = np.linspace(-0.5 * p, 0.5 * p, 1000)


        for r_star in np.arange(par1, par2+1, par3):
            r_star=r_star/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), r_star=r_star,  b=float(b2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$r_star={r_star}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()


################################################################################################impact_parameter
    elif z == 3 and v == 1:
        u = [float(u12), float(u22)]
        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for b in np.arange(par1, par2+1, par3):
            b=b/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), b=b,  r_star=float(rstar2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$b={b}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 3 and v == 2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")


        u = [float(u12), float(u22)]
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for b in np.arange(par1, par2+1, par3):
            b=b/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), b=b, r_star=float(rstar2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$b={b}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 3 and v == 3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p, 0.5 * p)
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)




        u = [float(u12), float(u22)]
        t = np.linspace(-0.5 * p, 0.5 * p, 1000)


        for b in np.arange(par1, par2+1, par3):
            b=b/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2), a=float(sma2), b=b, r_star=float(rstar2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$b={b}$")



        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()


################################################################################################Semi-Major Axis
    elif z == 4 and v == 1:
        u = [float(u12), float(u22)]
        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for a in np.arange(par1, par2+1, par3):
            a=a/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2), a=a, r_star=float(rstar2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$a={a}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 4 and v == 2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")


        u = [float(u12), float(u22)]
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for a in np.arange(par1, par2+1, par3):
            a=a/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2), a=a, r_star=float(rstar2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$a={a}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 4 and v == 3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p, 0.5 * p)
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)




        u = [float(u12), float(u22)]
        t = np.linspace(-0.5 * p, 0.5 * p, 1000)


        for a in np.arange(par1, par2+1, par3):
            a=a/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2), a=a, r_star=float(rstar2),
                                     ecc=float(eccn2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$a={a}$")



        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/65)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()


################################################################################################Eccentricity
    elif z == 5 and v == 1:
        u = [float(u12), float(u22)]
        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for ecc in np.arange(par1, par2+1, par3):
            ecc=ecc/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2), ecc=ecc, r_star=float(rstar2),
                                     a=float(sma2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$ecc={ecc}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 5 and v == 2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")


        u = [float(u12), float(u22)]
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for ecc in np.arange(par1, par2+1, par3):
            ecc=ecc/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2), ecc=ecc, r_star=float(rstar2),
                                     a=float(sma2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$ecc={ecc}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 5 and v == 3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p, 0.5 * p)
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)


        u = [float(u12), float(u22)]
        t = np.linspace(-0.5 * p, 0.5 * p, 1000)


        for ecc in np.arange(par1, par2+1, par3):
            ecc=ecc/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2), ecc=ecc, r_star=float(rstar2),
                                     a=float(sma2), omega=float(om2), t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$ecc={ecc}$")



        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()


################################################################################################Omega
    elif z == 6 and v == 1:
        u = [float(u12), float(u22)]
        zoom1 = -0.5*float(per2)
        zoom2 = 0.5*float(per2)
        t = np.linspace(float(zoom1), float(zoom2), 1000)

        for omega in np.arange(par1, par2+1, par3):
            omega=omega/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2),  r_star=float(rstar2),
                                     a=float(sma2), ecc=float(eccn2), omega=omega, t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$omega={omega}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(t.min()/6, t.max()/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 6 and v == 2:
        lc2 = lc.fold(period=best_fit_period, epoch_time=periodogram.transit_time_at_max_power).errorbar();
        lc2.plot()
        lc2.set_xlim(float(zoom1), float(zoom2))
        plt.xlabel("Time [JD]")


        u = [float(u12), float(u22)]
        t = np.linspace(float(zoom1), float(zoom2), 1000)


        for omega in np.arange(par1, par2+1, par3):
            omega=omega/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2),  r_star=float(rstar2),
                                     a=float(sma2), ecc=float(eccn2), omega=omega, t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$omega={omega}$")


        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(float(zoom1), float(zoom2))

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()

    elif z == 6 and v == 3:
        plt.plot(x_fold, 1+(y[mask] - gp_mod)/1e3, ".k", label="data", zorder=-1000)
        plt.xlim(-0.5 * p, 0.5 * p)
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)




        u = [float(u12), float(u22)]
        t = np.linspace(-0.5 * p, 0.5 * p, 1000)


        for omega in np.arange(par1, par2+1, par3):
            omega=omega/100;
            orbit1 = xo.orbits.KeplerianOrbit(period=float(per2),  b=float(b2),  r_star=float(rstar2),
                                     a=float(sma2), ecc=float(eccn2), omega=omega, t0=float(t02))
            light_curve = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit1, r=float(rpl2), t=t).eval()
            plt.plot(t, light_curve+1, label=f"$omega={omega}$")



        plt.ylabel("Flux")
        plt.xlabel("Time [JD]")
        _ = plt.xlim(-0.5 * p/6, 0.5 * p/6)

        plt.legend(bbox_to_anchor=(1.32,0.95))
        plt.show()






myLabel4 = Label(root,text="Multiple LC Model",fg="white",bg="black")
myLabel4.grid(row=46, column=20)

myButton4 = Button(root, text="Multiple LC Model", padx=33, pady=5, command=Zog, fg="blue", bg="white")
myButton4.grid(row=46, column=24)





root.mainloop()



