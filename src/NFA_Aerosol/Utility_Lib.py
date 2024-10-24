# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:54:00 2023

@author: B279683
"""

import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os as os

###############################################################################
###############################################################################
###############################################################################

def Diffusion_loss(Data_in, bin_mids, D_tube, L, Q, T = 293, P = 101.3e3):
    """
    Determine the diffusion loss of particles in tubing. The results are reported
    as the fraction of particles going through the tubing relative to the 
    concentration at the inlet for each particle size in the bins parameter. 
    
    In order to correct for the diffusion loss, the particle data should be 
    divided by this vector to represent the actual particle concentration of the
    sampled aerosol.
    
    All equations are from and labeled according to the book: 
    Willeke K, Baron PA. Aerosol measurement: principles, techniques, and 
    applications. 1st edition New York, NY, USA: Van Nostrand Reinhold; 1993.

    Parameters
    ----------
    Data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_mids : numpy.array
        Array of midpoints for all particle size bins. Sizes should be in nm.
    D_tube : float
        Diameter of the tubing used for sampling. Diameter should be in meters
    L : float
        Length of the tubing used for sampling. Length should be in meters.
    Q : float
        Flow through the tubing in questions. The flow should be in L/min
    T : Temperature, optional
        Temperature of the aerosol in kelvin. The default is 293 K.
    P : Pressure, optional
        Pressure of the sampled aerosol in Pa. The default is 101.3e3 Pa.

    Returns
    -------
    Data_return : numpy.array
        An array of diffusion loss corrected data equivalent to the data_in shape 
        with columns of datetime, total conc, and size bin concentrations.
    eff : numpy.array
        Ratio of particle concentration before and after diffusion losses for
        each particle size in the bins parameter. To apply the results, the
        reported particle concentrations should be divided by this vector.
    
    """
    # Boltzmann's constant
    k = 1.380649e-23
    
    # Convert particle sizes to m
    Dp = np.array(bin_mids) * 1e-9 # m
    
    # Convert flow from L/min to m3/s
    Q = Q/(1000*60) # m3/s
    
    # Calculate flow velocity in the tubing
    V = Q/(1/4*np.pi*D_tube**2) # m/s
    
    # Mean free path at 20 C and atm pressure with correction for pressure and 
    # temperature variations. B&W (3-6)
    mfp_standard = 66.5*1e-9 # m
    mfp = mfp_standard*(101e3/P)*(T/293.15)*((1+110/293.15)/(1+110/T)) # m
    
    # Gas viscosity at room temperature and pressure with correction for pressure 
    # and temperature variations. B&W (3-10)
    eta_standard = 1.708e-5 # kg/(m*s)
    eta = eta_standard*((T/273.15)**1.5)*((393.396)/(T+120.246)) # kg/(m*s)
    
    # Gas density at room temperature and pressure with correction for pressure
    # and temperature variations
    rho = 1.293*(273.15/T)*(P/101.3e3) # kg/m^3
    
    # calculate Knudsen number. B&W (3-7)
    Kn = 2*mfp/Dp
    
    # Cunningham slip correction factor by Allen and Raabe (1985). B&W (3-8)
    Cc = 1+Kn*(1.142+0.558*np.exp(-0.999/Kn))
    
    # Reynolds number. B&W (6-3)
    Re = rho*V*D_tube/eta
    
    # Diffusion coefficient for particles in gas. B&W (3-13)
    Dc = k*T*Cc/(3*np.pi*eta*Dp)
    
    # Schmidt number. B&W (3-17)
    Sc = eta/(rho*Dc);
    
    # Convenient parameter when calculating the Sherwood number. B&W (6-44)
    xi = np.pi*Dc*L/Q
    
    # The flow is laminar if Re < ~2000
    if Re < 2000:
        # Sherwood number for laminar flow, Holman (1972). B&W (6-43)
        Sh = 3.66+0.2672/(xi+0.10079*xi**(1/3))
    # If Re > ~4000 the flow is turbulent. In the transition region it is difficult 
    # to determine, so here we assume the flow is turbulent if it is not laminar
    else:
        # Sherwood number for turbulent flow, Friedlander (1977). B&W (6-45)
        Sh = 0.0118*Re**(7/8)*Sc**(1/3)
    
    # Transport efficiency with diffusive particle loss. B&W (6-42)
    eff = np.exp(-Sh*xi)
    
    # Apply the diffusion loss correction
    Data_return = Data_in.copy()
    Data_return[:,2:] = Data_return[:,2:]/eff
    Data_return[:,1] = Data_return[:,2:].sum(axis=1)
    
    return Data_return, eff

###############################################################################
###############################################################################
###############################################################################

def File_list(path, search = 0):
    """
    Function to generate a list of all files in a folder. If the search 
    parameter is set to a string, only files containing the string keyword are 
    in the returned list.

    Parameters
    ----------
    path : str
        The path to the desired folder location.
    search : str, optional
        A string keyword so that only files with the keyword in their name will
        be included in the returned list. The default is 0.

    Returns
    -------
    files : list
        List of all the files in the specified folder, which also contain the
        specified keyword if defined.

    """
    # If a keyword is specified include it in the list criteria
    if search:
        files = [x for x in os.listdir(path) if "{0}".format(search) in x]
    # If no keyword is specified, just list all files in the path
    else:
        files = [x for x in os.listdir(path)]
    files = [path + "\\" + i for i in files]
    return files

###############################################################################
###############################################################################
###############################################################################

def Fit_lognormal(bin_pop,bin_mids,mu_guess=0,sigma_guess=2,factor_guess=150):
    """
    A function to fit a lognormal distribution.

    Parameters
    ----------
    bin_pop : numpy.array
        An array of the particle size number distribution used as the y values
        of the lognormal fit.
    bin_mids : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
    mu_guess : float, optinonal
        If specified, acts as the initial guess of the particle mode, meaning
        the size where the particle size distribution peaks. If not specified
        the function will determine the size with the highest concentration and
        use it as the guess. This means that the function will fail if other
        peaks or artefacts with high number concentrations are present. 
        The default is 0.
    sigma_guess : float, optional
        Initial guess for the geometric standard deviation factor. A good guess
        is the size at peak height divided by the size at 2/3 peak height in
        the decending direction. E.g. the PSD peaks at 200 nm and is at 2/3 
        height at 140 nm, so the sigma_guess parameter should be 200/140 = 1.4.
        The default is 2.  
    factor_guess : float, optional
        Initial guess for the parameter used to scale the lognormal distribution.
        Getting a good estimate can be difficult, but a guess in the same order
        of magnitude as the peak height, is a good start e.g. 10e5. 
        The default is 150.

    Returns
    -------
    popt : list
        A list containing the fitted parameters mu, sigma, and factor. To get
        the geometric
    perr : list
        Error estimates for the fitted parameters.

    """
    # Specify x and y data to fit
    xdata = bin_mids
    ydata = bin_pop
    
    # Set an initial guess for mu, which should be the size at peak max
    if mu_guess:
        mu = mu_guess
    else:
        mu = xdata[ydata.argmax()]
    
    # Gather all the initial guesses for parameters to fit in a list
    init_guess = [mu,sigma_guess,factor_guess]
    
    # Do the fit
    popt, pcov   = curve_fit(Lognormal, xdata ,ydata,p0 = init_guess)
    
    # Get error estimates of the fits
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

###############################################################################
###############################################################################
###############################################################################

def Lognormal(bin_mid, mu,sigma,factor):
    """
    The mathmatical expression of a lognormal distribution. The function can be
    used to genereate a theoretical lognormal distribution and is also used by
    the Fit_lognormal function.

    Parameters
    ----------
    bin_mid : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
    mu : float
        The mode of the lognormal distribution.
    sigma : float
        The geometric standard devaition factor of the lognormal distribution.
    factor : float
        A scaling parameter needed to control and fit the particle numbers.

    Returns
    -------
    lognormal_function : function
        The lognormal function is returned.

    """
    # The lognormal function with log transformed inputs
    lognormal_function = ((1/(np.sqrt(2*np.pi) * np.log10(sigma))) *  np.exp(-((np.log10(bin_mid) - np.log10(mu))**2) / (2*np.log10(sigma)**2)))*factor
    return lognormal_function

###############################################################################
###############################################################################
###############################################################################

def Lung_Dep_ICRP(data_in,bin_mids,respvol,exposure_time=0,plot=0):
    """
    Function to calculate the lung deposited mass or surface area, based on the 
    average ICRP lung deposition model, the measured particle distributions, the 
    specified exposure time, and the estimated respirable volume. The unit of the
    input data determines the output, so if mass data is used, the unit should
    be ug/m3 in order to get ug deposited throughout the exposure. If surface
    area data is used, the unit should be um2/m3 to get LDSA values of um2
    deposited throughout the exposure time. 

    Parameters
    ----------
    data_in : numpy.array
        Particle size disbtribution data with columns of datetime, total conc, 
        and size bin data
    bin_mids : numpy.array
        Array of midpoints for all particle size bins. Sizes should be in nm..
    respvol : float
        Respirable volume rate in L/min. A setting of 25 L/min corresponds to
        light exercise, while a setting of 9 L/min correspond to a person at 
        rest. Heavy exercise is 50 L/min.
    exposure_time : int, optional
        Time of exposure in seconds. If no value is specified, the deposited 
        fraction is calculated using the length of the dataset, meaning exposure 
        for the entire duration of the measurements. The default is 0.
    plot : int, optional
        A boolen flag (1 or 0) to set whether or not to plot the ICRP model
        deposition curves. The default is 0.

    Returns
    -------
    Dep_HA : float
        Total deposition of particles deposited in the head airway determined 
        from the mean concentration of the dataset and the specified exposure 
        time and respirable volume.
    Dep_TB : float
        Total deposition of particles deposited in the tracheo bronchial region 
        determined from the mean concentration of the dataset and the specified 
        exposure time and respirable volume.
    Dep_AL : float
        Total deposition of particles deposited in the alveolar region determined 
        from the mean concentration of the dataset and the specified exposure 
        time and respirable volume.
    Dep_TT : float
        Total deposition of particles deposited in respiratory system determined 
        from the mean concentration of the dataset and the specified exposure time 
        and respirable volume.

    """
    # Convert particle sizes from nm to um
    Dp = bin_mids*1e-3 #um
    
    # Inhalable deposition fraction 
    IF = 1 - 0.5*(1 - 1 / (1 + 0.00076 * Dp**2.8 ))
     
    # Head airway deposition fraction
    DF_HA = IF * (1 / (1 + np.exp(6.84 + 1.183 * np.log(Dp))) + 1 / (1 + np.exp(0.924 - 1.885 * np.log(Dp))))
    
    # Tracheo bronchial deposition fraction
    DF_TB = (0.00352 / Dp) * (np.exp(-0.234 * (np.log(Dp) + 3.40)**2) + 63.9 * np.exp(-0.819 * (np.log(Dp) - 1.61)**2))
    
    # Alveolar deposition fraction
    DF_AL = (0.0155 / Dp) * (np.exp(-0.416 * (np.log(Dp)+2.84)**2) + 19.11 * np.exp(-0.482 * (np.log(Dp) - 1.362)**2)) 
    
    # Total depostion fraction
    DF_TT = DF_HA + DF_TB + DF_AL
    
    # Plot the depositon graf if specified
    if plot:
        plt.figure()
        plt.plot(bin_mids,IF,label="Inhalable")
        plt.plot(bin_mids,DF_TT,label="Total Depositiion")
        plt.plot(bin_mids,DF_HA,label="Head Airways")
        plt.plot(bin_mids,DF_TB,label="Tracho Bronchial")
        plt.plot(bin_mids,DF_AL,label="Alveolar")
        plt.legend()
        plt.xscale("log")
        plt.grid(axis="both",which="both")
        plt.ylabel("Deposition Fraction")
        plt.xlabel("Particle size, nm")
    
    # Calculate the volume of inhaled air for the specified time
    respvol = respvol * 1e-3 / 60 # [L/min] -> [m3/s]
    if exposure_time == 0:
        duration = data_in[-1,0] - data_in[0,0]
        exposure_time = duration.seconds
    
    # Calculate the deposited particles in the different airway regions
    Dep_conc_HA = data_in[:,2:] * DF_HA # deposited in head airways at each time
    Dep_conc_TB = data_in[:,2:] * DF_TB # deposited in Tracheo bronchial region at each time
    Dep_conc_AL = data_in[:,2:] * DF_AL # deposited in Alveolar region at each time
    Dep_conc_TT = data_in[:,2:] * DF_TT # deposited in total at each time
    
    # Sum depostion of all sizebins and take the average to get the average
    # deposition during the measurement.
    Dep_HA = Dep_conc_HA.sum(axis=1).mean() 
    Dep_TB = Dep_conc_TB.sum(axis=1).mean() 
    Dep_AL = Dep_conc_AL.sum(axis=1).mean() 
    Dep_TT = Dep_conc_TT.sum(axis=1).mean() 
    
    # Calculate the deposited fraction, based on the volume of inhaled air
    Dep_HA = Dep_HA * respvol*exposure_time 
    Dep_TB = Dep_TB * respvol*exposure_time
    Dep_AL = Dep_AL * respvol*exposure_time
    Dep_TT = Dep_TT * respvol*exposure_time
    
    return Dep_HA,Dep_TB,Dep_AL,Dep_TT

###############################################################################
###############################################################################
###############################################################################

def Normalize_dndlogdp(data_in,bin_edges):
    """
    Function to normalize an array of particle xxx concentrations (xxx can be
    mass, volume, surface area, and number), going from dxxx values to 
    dxxx/dlogDp.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one 
        more value than the length of the "data_in" parameter

    Returns
    -------
    Data_return : numpy.array
        An array of normalized data equivalent to the data_in shape with columns
        of datetime, total conc, and normalized size bin concentrations.

    """
    # Calculate the normalization vector
    dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
    
    # Copy the input data to not mess with the parent
    Data_return = data_in.copy()
    
    # Select relevant data
    data = Data_return[:,2:]
    
    # Apply normalization
    Normed = data/dlogDp
    # Normed_total = np.sum(Normed,axis=1)
    
    # Store the normalized data instead of the non-normalized and return it
    # Data_return[:,1] = Normed_total
    Data_return[:,1] = data_in[:,1]  # this is what we usually mean by Total
    Data_return[:,2:] = Normed
    
    return Data_return

###############################################################################
###############################################################################
###############################################################################

def num2mass(data_in,bin_mids,density=1.0):
    """
    Function to convert from non-normalized number concentration to mass
    concentration, assuming spherical particles with a diameter equal to the
    size bin mid points, and a density as specified.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data. The unit should be #/cm3
    bin_mids : numpy.array
        Array containing the mid-points of all sizebins. The array should have 
        the same number of bins as the "data_in" parameter
    density : float, optinonal
        A density can be specified. The unit should be g/cm3. The default is a
        density of 1.0 g/cm3.

    Returns
    -------
    Data_return : numpy.array
        An array of mass concentration data equivalent to the data_in shape with 
        columns of datetime, total mass conc in gram, and size bin mass concentrations
        in ug/m3.

    """
    # Copy the original data
    Data_return = data_in.copy()
    
    # Select the relevant data and bins
    data = Data_return[:,2:].astype("float64")
    bins = np.array(bin_mids, dtype="float64")
    
    # Determine conversion vector from number to volume, cm3
    Num2Vol = (4./3.)*np.pi*((bins*1e-7)/2.)**3
    
    # Convert from volume to mass via the specified density, g
    Num2mass = Num2Vol * density
    
    # Apply the conversion vector to the particle number concentrations, g
    dm = data * Num2mass 

    # Convert from g/cm3 to ug/m3 
    mass_return = dm * 1e12
    
    # Determine the total mass at each timestep, ug/m3
    total_mass = mass_return.sum(axis=1)

    Data_return[:,2:] = mass_return
    Data_return[:,1] = total_mass
    
    return Data_return

###############################################################################
###############################################################################
###############################################################################

def num2surface(data_in,bins_in):
    """
    Function to convert from non-normalized number concentration to surface area
    concentration, assuming spherical particles with a diameter equal to the
    size bin mid points.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bins : numpy.array
        Array containing the mid-points of all sizebins. The array should have 
        the same number of bins as the "data_in" parameter
    
    Returns
    -------
    Data_return : numpy.array
        An array of surface area concentration data equivalent to the data_in 
        shape with columns of datetime, total surface area concentration in 
        nm2/cm3, and size bin surface area concentrations in nm2/cm3 of air.

    """
    Data_return = data_in.copy()
    data = Data_return[:,2:].astype("float64")
    bins = np.array(bins_in, dtype="float64")
    
    # nm2 surface per particle in each size bin
    Num2surface = 4*np.pi*(bins/2.)**2
    
    # aply conversion (nm**2 / cm**3)
    Surface = data * Num2surface 

    # Store the size bin data
    Data_return[:,2:] = Surface
    
    # Calculate and store the total surface area
    Data_return[:,1] = Surface.sum(axis=1)

    return Data_return

###############################################################################
###############################################################################
###############################################################################

def num2vol(data_in,bins_in):
    """
    Function to convert from non-normalized number concentration to volume
    concentration, assuming spherical particles with a diameter equal to the
    size bin mid points.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bins : numpy.array
        Array containing the mid-points of all sizebins. The array should have 
        the same number of bins as the "data_in" parameter
    
    Returns
    -------
    Data_return : numpy.array
        An array of volume concentration data equivalent to the data_in shape with 
        columns of datetime, total volume concentration in ul/m3, and size bin 
        volume concentrations in nm3/cm3.

    """
    Data_return = data_in.copy()
    data = Data_return[:,2:].astype("float64")
    bins = np.array(bins_in, dtype="float64")
    
    # Calculate volume per particle for each bin
    Num2Vol = (4./3.)*np.pi*(bins/2.)**3

    # apply conversion to get nm3/cm3 
    Vol = data * Num2Vol 
    
    # sum to get the total volume
    total_vol = Vol.sum(axis=1)

    # store the converted size bin data
    Data_return[:,2:] = Vol
    
    # store the converted total volume
    Data_return[:,1] = total_vol
    
    return Data_return
    
###############################################################################
###############################################################################
###############################################################################

def PM_calc(data_in,bin_edges,*PM):
    """
    Function to calculate PM from size bins from an array similar to those
    returned from Load_xxx functions. 
    
    Note that the data should already be converted to mass e.g. ug/m3 as the
    output of this function will maintain the same units as the input.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data. The size bin data should
        be converted to mass based units e.g. ug/m3.
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one
        additional value when compared to the sizebin data in the "data_in" 
        parameter
    *PM : integer or float
        PM limit or limits to use, given in um. Typical values are 0.1, 1, or
        10 to yield PM0.1, PM1, and PM10 respectively. Note that multiple PM
        limits can be specified at the same time e.g. 
        PM_calc(data,bins,0.1,1,10)
        In which case the returned data array will have a column for each PM mass
    Returns
    -------
    Data_return : numpy.array
        An array of mass concentration data. The first column is datetime. 
        The other column or columns are PM values corresponding to each time for
        the specified PM limit.

    """
    # Remove the datetime and total conc columns as these are not needed for PM
    # calculations
    data = data_in[:,2:]
    
    # Make a new array with the necessary shape
    Data_return = np.zeros((data_in.shape[0],len(PM)+1),dtype="object")
    
    # Set the first column of the new array equal to the datetime values of the input
    Data_return[:,0] = data_in[:,0]
    
    header = ["Datetime"]
    counter = 1
    delete_index = []
    
    # Run through the or all of the PM limits specified
    for i in PM:
        # convert the current PM limit frim um to nm
        PM_lim = i * 1000
        
        # Determine the number of bins smaller than the specified PM limit
        bin_within_range = np.array(bin_edges<PM_lim).sum()
        
        
        if bin_within_range == 0:
            # If none of the size bins are above the PM limit, continue to the
            # next specified PM limit 
            print("PM{0} is smaller than all size bins, so it will not be included in the output array".format(i))
            delete_index += [counter]
            counter += 1
            continue
        
        elif bin_within_range == 1:
            # If only the first bin is partially within the PM limit, calculate
            # the fraction within the limit and multiply with the mass concentration
            bin_frac = (PM_lim - bin_edges[0]) / (bin_edges[1]-bin_edges[0])
            final_PM = data[:,0]*bin_frac
            
            # Add a header
            header += ["PM{0}".format(i)]
            
        elif bin_within_range < bin_edges.shape[0]:
            # If several bins are within the PM limit, sum the relevant ones
            Initial_PM = data[:,:bin_within_range-1].sum(axis=1)
            
            # Determine the fraction of the highest relevant bin within the specified
            # PM limit
            bin_frac = (PM_lim - bin_edges[bin_within_range-1]) / (bin_edges[bin_within_range] - bin_edges[bin_within_range-1])
            
            # Apply the fraction to the mass concentration of the bin. Here it
            # is assumed that the mass concentration is evenly distributed within
            # the size bin
            mass_frac = data[:,bin_within_range-1]*bin_frac
    
            # Add the two valuyes for a final mass concentration
            final_PM = Initial_PM + mass_frac
            
            # Add a header
            header += ["PM{0}".format(i)]
            
        else:
            # If all bins are smaller than the specified PM limit, simply sum them
            final_PM = data.sum(axis=1)
            
            # Add a header
            header += ["PM{0}".format(i)]
       
        # Store all the PM values in a new array
        Data_return[:,counter] = final_PM
        counter += 1    
    
    # Delete columns if they are empty
    Data_return = np.delete(Data_return,delete_index,axis=1)
    
    return Data_return, header

###############################################################################
###############################################################################
###############################################################################

def Save_2_Excel(data_in,header,filename,path):
    """
    Function to store data to an excel file.

    Parameters
    ----------
    data_in : np.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    header : list
        A list of column names as returned by the Load_xxx functions
    filename : str
        Desired file name without the file extension. E.g. "ELPI_test" will be
        stored as "ELPI_test.xlsx" at the specified path
    path : str
        Path to the desired folder in which to store the file. E.g. 
        'L:\\PG-Nanoteknologi\\PERSONER\\xxx'

    Returns
    -------
    Nothing is returned by this function

    """
    # Combine the data with the header in a pandas.DataFrame for easy saving 
    DataFile = pd.DataFrame(data_in,columns=header)
    
    # Write the DataFrame to an excel file with the specified path and filename
    DataFile.to_excel("{0}\\{1}.xlsx".format(path,filename),index=False)
    
    # Print in the consol that the function finished
    print("File is now saved")

###############################################################################
###############################################################################
###############################################################################

def Save_plot(figure,filename,path,extension=".png"):
    """
    Function to save a figure at the specified path and with the given filename 
    and extension.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Handle for the figure to save. Such handles are always returned by the
        plotting functions of the NFA Python Library.
    filename : str
        Desired file name without the file extension. E.g. "ELPI_test" will be
        stored as "ELPI_test.png" at the specified path
    path : str
        Path to the desired folder in which to store the plot. E.g. 
        'L:\\PG-Nanoteknologi\\PERSONER\\xxx'
    extension : str, optional
        Decired file extension e.g. ".tif" or ".bmp". The default is ".png".
    Returns
    -------
    Nothing is returned by this function

    """
    # Save the figure with a dot per inch of 500 to have a good resoulution.
    figure.savefig("{0}\\{1}{2}".format(path,filename,extension),dpi=500)
    print("Plot was saved")

###############################################################################
###############################################################################
###############################################################################

def segment_dataset(data_in,segments):
    """
    Function to segment a dataset into different activities e.g. background and
    different workplace processes. Each activity can have multiple time 
    segments as long as a start time always has a corresponding end time. The
    times for each activity should be kept in serperate lists and joined in an
    overall list with all the activity start and end times.
    example:
        # The background has 2 segments from 12 to 14.30 and from 15.30 to 16.20
        background = [datetime.datetime(2018,11,28,12,0,0), datetime.datetime(2018,11,28,14,30,0),
                      datetime.datetime(2018,11,28,15,30,0), datetime.datetime(2018,11,28,16,20,0)
                      ]
        # pouring occured from 14.33 to 15.24
        pouring = [datetime.datetime(2018,11,28,14,33,0), datetime.datetime(2018,11,28,15,24,0)]
        
        # sanding occured from 16.23 to 17.00
        sanding = [datetime.datetime(2018,11,28,16,23,0), datetime.datetime(2018,11,28,17,0,0)]
        
        # Store all time lists in an overall list
        Activities = [background, pouring, sanding]
        
        # Feed to function for indexing
        segment_dataset(data_in,Activities)
        
    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    segments : list
        list of lists containing start and end times for activities. See example
        in the function description.

    Returns
    -------
    index : numpy.array
        Array of indexes, indicating which datapoints belongs to each activity.
        Non-categorized datapoints will have an index of 0, while others will
        have 1, 2, 3 and so on, in the order given in the segments variable.

    """
    
    # Generate an array of zeros for indexing
    index = np.zeros(data_in.shape[0])
    
    # Loop through the different time segments
    for j,i in enumerate(segments):
        
        # If only one time segment was given
        if len(i) == 2:
            start = i[0]
            end = i[1]
            
            # Find the indexes, where the experiment time is between the specified
            # start and end time and give these indexes a value of j+1
            index[(data_in[:,0]>start) & (data_in[:,0]<end)] = j+1
        
        # If there is more than one time segment for the given activity
        else:
            # store all starting points in one and end points in another variable
            starts = i[::2]
            ends = i[1::2]
            
            # Loop through all time segments and find the indexes, where the 
            # experiment time is between the specified start and end times and 
            # set these indexes to j+1
            for k in range(len(starts)):
                index[(data_in[:,0]>starts[k]) & (data_in[:,0]<ends[k])] = j+1
    
    return index

###############################################################################
###############################################################################
###############################################################################

def time_crop(data_in,start=0,end=0):
    """
    Function to crop a dataset, by setting a start and/or an end time. This is
    needed e.g. when making direct comparison between datasets in order to
    ensure the same number of datapoints.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    start : datetime, optional
        Specify starting time to crop by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to stop cropping by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.

    Returns
    -------
    Data_return : numpy.array
        New array following the same structure as the input array but with new
        times.

    """
    # Store the time
    times = data_in[:,0]
    
    # Find the index of times fulfilling the start and/or end time conditions
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = times>start
        elif (start == 0) & (end != 0):
            index = times<end
        else:
            index = (times>start) & (times<end)
        
        # Select the data rows fulfilling the time criterias
        Data_return = data_in[index,:]
        
    return Data_return    

###############################################################################
###############################################################################
###############################################################################

def time_rebin(data_in, resize_factor):
    """
    Rebin a dataset to lower the time resolution and reduce noise. Concentrations 
    will be averaged over the specified resize facotr e.g. going from every 
    second to every 5th second if the resize_factor is 5. 
    
    NOTE! The function will drop datapoints at the end of the dataset untill
    the number of datapoints is divisible by the resize factor.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    resize_factor : int
        Factor of which to downsize the dataset with.

    Returns
    -------
    Data_return : numpy.array
        Rebinned array following the same structure as the input array but with 
        less rows as these have been averaged.

    """
    # Check if the remainder upon division is zero. If not, raise an error and
    # ask for a new factor or a cropped dataset to match the factor
    while data_in.shape[0]%resize_factor != 0:
        data_in = data_in[:-1,:]
    
    # Copy the dataset to avoid issues with potential parent relationship
    Reshape = data_in.copy()
    
    start = Reshape[0,0]
    Reshape[:,0] = Reshape[:,0]-start
    
    # Determine the new length after averaging
    new_length = Reshape.shape[0]//resize_factor
    
    # Determine the most efficient shape to use for rebinning
    new_shape = new_length,Reshape.shape[0]//new_length,Reshape.shape[1],1
    
    # Reshape the initial array and mean over the specified number of rows
    Data_return = Reshape.reshape(new_shape).mean(-1).mean(1)
    Data_return[:,0] = Data_return[:,0]+start
        
    return Data_return    

###############################################################################
###############################################################################
###############################################################################

def time_shift(data_in,delta):
    """
    Function to adjust the times of a dataset. All times can be shifted in either
    positive or negative direction by the specified number of seconds in the
    delta parameter. 

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    delta : int
        Number of seconds to shift the dataset. The shift can be both positive
        and negative

    Returns
    -------
    Data_return : numpy.array
        New array following the same structure as the input array but with new
        times.

    """
    # Store the datetime values
    time = data_in[:,0]
    
    # Store the needed time shift
    delta = datetime.timedelta(0,delta)
    
    # Shift the times
    new_time = time + delta
    
    # Copy the original dataset and insert the shifted times
    Data_return = data_in.copy()
    Data_return[:,0] = new_time
    
    return Data_return    

###############################################################################
###############################################################################
###############################################################################

def Unnormalize_dndlogdp(data_in,bin_edges):
    """
    Function to unnormalize an array of particle xxx concentrations (xxx can be
    mass, volume, surface area, and number), going from dxxx/dlogDp to dx values.

    Parameters
    ----------
    data_in : numpy.array
        An array of normalized data as returned by the Load_xxx functions with 
        columns of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one 
        more value than the length of the "data_in" parameter

    Returns
    -------
    Data_return : numpy.array
        An array of normalized data equivalent to the data_in shape with columns
        of datetime, total conc, and normalized size bin concentrations.

    """
    # Calculate the normalization vector
    dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
    
    # Copy the input data to not mess with the parent
    Data_return = data_in.copy()
    
    # Select relevant data
    data = Data_return[:,2:]
    
    # Apply normalization
    UnNormed = data*dlogDp
    UnNormed_total = np.sum(UnNormed,axis = 1)
    
    # Store the normalized data instead of the non-normalized and return it
    Data_return[:,2:] = UnNormed
    Data_return[:,1] = UnNormed_total
    
    return Data_return

def calculate_average(data, start_time, end_time):
    """
    Function to calculate the average of the second column of a data array 
    within a specified time range. The data array is expected to have datetime 
    objects in the first column and numerical values in the second column.

    Parameters
    ----------
    data : numpy.array
        A 2D array where the first column contains datetime objects and the 
        second column contains numerical values for which the average is to be calculated.
    start_time : datetime
        The start time for the period over which to calculate the average.
    end_time : datetime
        The end time for the period over which to calculate the average.

    Returns
    -------
    float
        The average of the second column of the data array within the specified 
        time range. Returns NaN if no data is available in the time span.
        
        Created by: PLF
    """
    filtered_data = data[(data[:, 0] >= start_time) & (data[:, 0] <= end_time)]
    if len(filtered_data) > 0:
        return np.mean(filtered_data[:, 1])
    else:
        return np.nan  # Return NaN if no data is available in the time span

def filter_data(data, start_time, end_time):
    """
    Function to filter data for a specific time span

    Parameters
    ----------
    data : numpy.array
        A 2D array where the first column contains datetime objects and the 
        second column contains numerical values for which the average is to be calculated.
    start_time : datetime
        The start time for the period over which to calculate the average.
    end_time : datetime
        The end time for the period over which to calculate the average.

    Returns
    -------
    data : numpy.array
        
        Created by: PLF
    """
    filtered_data = data[(data[:, 0] >= start_time) & (data[:, 0] <= end_time)]
    return filtered_data