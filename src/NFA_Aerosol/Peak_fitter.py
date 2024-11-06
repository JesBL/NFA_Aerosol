# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:54:00 2023

@author: B351796
"""

import numpy as np
from scipy.optimize import curve_fit
from matplotlib.dates import date2num, num2date

###############################################################################
"""The following is a list of simple functions fed into the peak_fit function"""
###############################################################################
def exp_func(t, k, P0):
    """Calculates the concentration P(t) for a population experincing
    a first order decay: dP/dt = - k*P
    
    Parameters
    ----------
    t: float
        Time at which the concentration is investigated (s)
    k: float
        Rate constant for the second order order decay (cm3/s)
    P0: float
        Starting concentration of particles (1/cm3)
        
    Returns
    -------
    P(t): float (1/cm3)
    The resulting concentration P at time t"""
    
    k=abs(k)
    P0=abs(P0)
    return P0 * np.exp(-k * t)
###############################################################################
def lin_func(t, c, P0):
    """Calculates the concentration P(t) for a population experincing
        a second order decay: dP/dt = - k*P^2
    
    Parameters
    ----------
    t: float
        Time at which the concentration is investigated (s)
    c: float
        Rate constant for the second order order decay (cm3/s)
    P0: float
        Starting concentration of particles (1/cm3)
        
    Returns
    -------
    P(t): float (1/cm3)
    The resulting concentration P at time t"""
    
    c=abs(c)
    inv_P0=1/abs(P0)
    return 1/(c*t + inv_P0)
##############################################################################
def third_func(t,K,C,P0):
    """Calculates the concentration P(t) for a population experincing
        a third order decay of the form: dP/dt = - (K*P + C*P^2)
    
    Parameters
    ----------
    t: float
        Time at which the concentration is investigated (s)
    K: float
        First order rate constant (1/s)
    C: float
        Second order rate constant (cm3/s)
    P0: float
        Starting concentration of particles (1/cm3)
        
    Returns
    -------
    P(t): float (1/cm3)
    The resulting concentration P at time t"""
    
    K=abs(K)
    C=abs(C)
    return 1/((P0*C+K)/(P0*K)*np.exp(K*t)-C/K)

###############################################################################
def Conc1_func(t,k,E,P0):
    """Calculates the concentration from an emission experincing a first order
    decay: dP/dt = E- K*P
    
    Parameters
    ----------
    t: float
        duration of the emission (s)
    k: float
        First order rate constant (1/s)
    E: float
        Emission strength (1/cm3s)
    P0: float
        Starting concentration of particles (1/cm3)
        
    Returns
    -------
    P(t): float (1/cm3)
    The resulting concentration P as a function of the emission duration"""
    
    k=abs(k)
    E=abs(E)
    P0=abs(P0)

    return np.exp(-k*t)*(E*(np.exp(k*t)-1)+k*P0)/k
###############################################################################
def Conc2_func(t,c,E,P0):
    """Calculates the concentration from an emission experincing a second order
    decay dP/dt = E- K*P^2
    
    Parameters
    ----------
    t: float
        duration of the emission (s)
    c: float
        Second order rate constant (cm3/s)
    E: float
        Emission strength (1/cm3s)
    P0: float
        Starting concentration of particles (1/cm3)
        
    Returns
    -------
    P(t): float (1/cm3)
    The resulting concentration P as a function of the emission duration"""
    
    c=abs(c)**0.5
    e=abs(E)**0.5
    P0=abs(P0)

    return e*np.tanh(np.arctanh(c*P0/e)+c*e*t)/c
###############################################################################
def Conc3_func(t,K,C,E,P0):
    """Calculates the concentration from an emission experincing a third order
    decay dP/dt = E - (K*P + C*P^2)
    
    Parameters
    ----------
    t: float
        duration of the emission (s)
    K: float
        First order rate constant (1/s)
    C: float
        Second order rate constant (cm3/s)
    E: float
        Emission strength (1/cm3s)
    P0: float
        Starting concentration of particles (1/cm3)
        
    Returns
    -------
    P(t): float (1/cm3)
    The resulting concentration P as a function of the emission duration"""
    
    K=abs(K)
    C=abs(C)
    E=abs(E)
    P0=abs(P0)
    sq=(K**2+4*C*E)**0.5
    return (sq*np.tanh(np.arctanh((2*C*P0/sq))+1/2*sq*t)-K)/(2*C)
###############################################################################
#Calculates the R^2 value for a function
def R2(data,fit):
    #data: 
    # residual sum of squares
    ss_res = np.sum((data - fit) ** 2)
    # total sum of squares
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    # r-squared
    return round((1 - (ss_res / ss_tot)),3)
###############################################################################
###############################################################################
# def linear_func(x,A,B=0):
#     """Calculates a first order equation.
    
#     Parameters
#     ----------
#     x: float
#         Variable 
#     A: float
#         Slope of equaiton
#     B: float
#         intercept, is initially set to 0
        
#     Returns
#     -------
#     y: A*x+B"""
    
#     return A*x + B

# ###############################################################################
###############################################################################    
def Emission_determintation(Fit,t,BG,Max,K,C=0,P=(0,6)):
    """Calculates the emission in number of particles per second per cm3
    based on the maximum concentration reached, time it took to reach it,
    and the fitted decay rate.
    
    Parameters
    ----------
    Fit: str
        Type of fit, either "Exp", "Lin" or "Three", refering to the three decay fits.
    t: float
        duration of the emission (s)
    BG: float
        Background concentration (1/cm3)
    Max: float
        The maximum concentration reached at the end of the emission. (1/cm3)
    K: float
        Princicple rate constant from the fit.
        Unit: Exp = 1/s, Lin = cm3/s, Three = 1/s
    C: float
        Secondary rate constant, default= 0. Only used for fit = "Three". (cm3/s)
    P: tuple (int, int)
        Range of indexes over which the powers of 10 should fit the emission. 10^P
            
    Returns
    -------
    E: float 
        The resulting average emission over the period (1/cm3s)"""
    
    current_fit=0
    new_fit=0
    E=0
    for p in range(P[0],P[1]):
        find=False
        for i in range(1,10):
            try:
                if Fit=="Exp":
                    estimate=(Conc1_func(t,K,i*10**p,BG)-Max)**2
                elif Fit=="Lin":
                    estimate=(Conc2_func(t,K,i*10**p,BG)-Max)**2
                elif Fit=="Three":
                    estimate=(Conc3_func(t,K,C,i*10**p,BG)-Max)**2
                    
                if current_fit==0 and estimate>0:
                    current_fit=estimate
    
                elif estimate<current_fit:
                    current_fit=estimate
                    
                elif estimate>current_fit:
                    #As the function has one minimum, we stop the search when the difference increases again
                    find=True
                    if i==1:
                        i=10
                        p=p-1
                    break
            except: 
                pass  
        
        if find==True:
            break  
    for v in range(-5000,5000):
        try:
            if Fit=="Exp":
                estimate=(Conc1_func(t,K,(i-1+v/10000)*10**p,BG)-Max)**2
            elif Fit=="Lin":
                estimate=(Conc2_func(t,K,(i-1+v/10000)*10**p,BG)-Max)**2
            elif Fit=="Three":
                estimate=(Conc3_func(t,K,C,(i-1+v/10000)*10**p,BG)-Max)**2
            try: round(estimate)
            except: continue
            if new_fit==0:
                new_fit=estimate
                
            elif estimate<new_fit:
                new_fit=estimate
                
            elif estimate>new_fit:
                E=(i-1+(v-1)/10000)*10**p
                break
        except: 
            pass
    return E

###############################################################################
def Peak_fitter(Data,Start=0,End=0,bin_mids=[],Peak_filter=0,Decay_length=60,Search_range=30,P=(-4,5)):
    """Find the location of the peak by identifying the maximum value,
    and fits one of three functions to the peak, sa well as provide emission calculations
    
    Parameters
    ----------
    Data : numpy.array
        An array of data as returned by the Load_xxx functions with 
        columns of datetime, total conc
    Start : datetime
        Precise or estimateted time of the begining of the peak
    End : datetime
        Precise or estimateted time of the top of the peak
    bin_mids: list
        If the instrument is sized, apply here the list of bin_mids
    Peak_filter : float
        Setpoint of a minimum the peak has to be larger than. 
    Decay_length : float
        Length in minutes of the decaying peak that should be fitted.
        A longer stretch of time to fit normally yields better results,
        but as the values return towards the background level the decay should 
        be limited. 
    Search_range : float
        Range in minutes around the estimated End time where the peak is expected to be found.
    P : tuple
        Range of power to be investigated for emission assesment
        
    Returns
    -------
    Output : dictonary
        A dictonary of values:
            PT : Found time of the peak (datetime)
            BG : Background concentration before the begining of the peak (float)
            Fit : Fit type and  associated values and errors
            P0 : Fitted conctration at the peak in 1/cm3 (float)
            Emission_rate : 
    """   
    output={}
    ###################################################################
    """Find the location of the peak by identifying the maximum value"""
    ###################################################################                 
    #Calculates the date as number
    ex_date = int(date2num(Data[0,0]))
    
    #Initial data cleaning to remove nan values
    Data_clean=np.isnan(Data[:,1].astype(float))
    Data_clean= [not elem for elem in Data_clean]
            
    Data=Data[Data_clean]
    
    #Establish the search range in seconds
    if Start==0:
        Peak_Start=Data[0,0]
    else:
        Peak_Start=Start    
    T0 = Peak_Start.hour*60*60 + Peak_Start.minute*60 + Peak_Start.second
    
    if End==0 and Start==0:
        Search_range=(Data[-1,0]-Data[0,0]).total_seconds()
        Peak_End=Data[-1,0]
    elif End==0:
        Peak_End=Data[-1,0]
    else:
        Peak_End=End   
    TE = Peak_End.hour*60*60 + Peak_End.minute*60 + Peak_End.second
    
    Time = (date2num(Data[:,0])-ex_date)*24*60*60

    #Determines if the peaks should be found for differnet size bins if applicable.
    if bin_mids == []:
        Sized = False
        Concentration = Data[:, 1]  
    else:
        Sized = True
        Concentration = Data[:,1:]        
        
    if max(Data[:,1])<Peak_filter:
        Peak = False
        search_mask = (Time > (T0 - Search_range)) & (Time < TE + Search_range)
    elif max(Data[:,1] )>=Peak_filter:
        Peak = True
        search_mask = (Time <= (TE+Search_range)) & (Time>TE-Search_range)
    
    #Round to the nearest second
    Time = Time[search_mask]
    for i in range(0,len(Time)):
        Time[i]=round(Time[i])

    concentration = Concentration[search_mask]            
    ###################################################################
    """Set up the for loop for the size bins"""
    ###################################################################
    if Sized==True:
        
        ###################################################################
        """Find the location of the peak or valley by identifying the maximum value or minimum value"""
        ###################################################################
        BG_tot=0
        P0_tot=0
        E_tot=0
        #for s in range(0,len(Concentration[0,1:])):
        for s in range(0,len(bin_mids)):
            peak_index = np.argmax(concentration[:,s])
            peak_time = Time[peak_index]
            conc=Concentration[:,s+1]
            #Removes zero values from the data set
            mask = conc>0
            conc=conc[mask]
            Timed=(date2num(Data[:,0])-ex_date)*24*60*60
            Timed=Timed[mask]
            #Generate the name for the reporting folder with the bin size
            name= str(bin_mids[s])
            output[name]={}
            ###################################################################
            """Find the location of the pre-peak minimum and determine the background value"""
            ###################################################################
            if len(conc)<10:
                output[name]={'PT':Start,
                    'BG':"NaN",
                    'P0':"NaN",
                    'Emission_rate':0
                    }
                continue
            #print("Pre-peak min occured at ",int(valley_time/3600),":",int(valley_time%3600/60),":",int(valley_time%60))
            BG_min_index=np.argmin(conc)
                                           
            #Determines the background averages and spread before the peak
            BG_avg = round(np.average(conc[:BG_min_index+1]),1)

            if Peak == True: #If a peak is found, a more thorough search starts for determining the emission rate
                
                #Defines the decay period between the peak and the return to the background level
                Decay_mask = (Timed>=peak_time) & (Timed<=peak_time + Decay_length*60)
                
                #Defines the decay data from its mask
                decay_period = Timed[Decay_mask]-peak_time
                decay_conc = conc[Decay_mask]
                Exp_fit={}
                Lin_fit={}
                Three_fit={}
                ###############################################################
                """Fit the found peak to an exponential decay assuming first order equation"""
                ###############################################################
                try:
                    parameters, covariance =curve_fit(exp_func,decay_period,decay_conc,p0=[1/(60*60),decay_conc[0]],bounds=(0,np.inf))
                    fit_k, fit_P0 = parameters
        
                    SE = np.sqrt(np.diag(covariance))
                    SE_k , SE_P0 = SE
        
                    exp_fit=exp_func(decay_period,fit_k,fit_P0)
        
                    # residual sum of squares
                    r2_ex=R2(decay_conc,exp_fit)
                    
                    Exp_fit={'P0':fit_P0,
                             'k':fit_k,
                             'SE_P0': SE_P0,
                             'SE_k': SE_k,
                             'r2': round(r2_ex,3)}
                    if r2_ex>0.5:
                            Exp=True
                    else: Exp=False
                
                except: Exp=False
                ###############################################################
                """Fit the found peak to a decay with a second order equation"""
                ###############################################################
                try:
                    parameters, covariance =curve_fit(lin_func,decay_period,decay_conc, p0=[1,decay_conc[0]],bounds=(0,np.inf))
                    fit_c, fit_P0 = parameters
        
                    SE = np.sqrt(np.diag(covariance))
                    SE_c , SE_P0 = SE
        
                    lin_fit=lin_func(decay_period,fit_c,fit_P0)
                    # residual sum of squares
                    r2_lin=R2(decay_conc,lin_fit)
                    
                    Lin_fit={'c':fit_c,
                             'P0':fit_P0,
                             'SE_m': SE_c,
                             'SE_P0': abs(1/fit_P0**2*SE_P0),
                             'r2': round(r2_lin,3)}
                    if r2_lin>0.5:
                            Lin=True
                    else: Lin=False
                    
                except: Lin=False
                ###############################################################
                """Fit the found peak to an combined first and second order equation"""
                ###############################################################
                try:
                    parameters, covariance =curve_fit(third_func,decay_period,decay_conc, p0=[1E-4,1E-5,decay_conc[0]],bounds=(0,np.inf))
                    fit_K, fit_C, fit_P0 = parameters.copy()
            
                    SE = np.sqrt(np.diag(covariance))
                    SE_K , SE_C, SE_P0 = SE.copy()
            
                    threehalf_fit=third_func(decay_period,fit_K,fit_C,fit_P0)
                    # residual sum of squares
                    r2_three=R2(decay_conc,threehalf_fit)
                    Three_fit={'K':fit_K,
                               'C':fit_C,
                               'P0':fit_P0,
                               'SE_K': SE_K,
                               'SE_C': SE_C,
                               'SE_P0': SE_P0,
                               'r2': round(r2_three,3)}
                    
                    if r2_three>0.5:
                            Three=True
                    else: Three=False
                    
                except: Three=False
                ###############################################################   
                """Determine the releasd amount of particles"""
                ###############################################################
                P0=0
                if Three==True and Lin==True and Exp==True:
                    if ((1-Three_fit['r2'])*1.25 < 1-Lin_fit['r2']) and (
                        (1-Three_fit['r2'])*1.5 < 1-Exp_fit['r2']):                
                        P0=Three_fit['P0']
                        fit={'Three':Three_fit}
                        
                    elif ((1-Three_fit['r2'])*1.25 > 1-Lin_fit['r2']) and (
                        (1-Lin_fit['r2'])*1.25 < 1-Exp_fit['r2']): 
                        P0=Lin_fit['P0']
                        fit={'Lin':Lin_fit}
                        
                    else:
                        P0=Exp_fit['P0']
                        fit={'Exp':Exp_fit}
                        
                elif Three==True and Lin==True and Exp==False:
                    if (1-Three_fit['r2'])*1.25 < 1-Lin_fit['r2']:
                        P0=Three_fit['P0'] 
                        fit= {'Three':Three_fit}
                        
                    else:
                        P0=Lin_fit['P0']
                        fit={'Lin':Lin_fit}
                        
                elif Three==True and Lin==False and Exp==True:
                    if (1-Three_fit['r2'])*1.25 < 1-Exp_fit['r2']:
                        P0=Three_fit['P0'] 
                        fit= {'Three':Three_fit}
                        
                    else:
                        P0=Exp_fit['P0']
                        fit={'Exp':Exp_fit}
                        
                elif Lin==True and Exp==True:
                    if (1-Lin_fit['r2'])*1.25 < 1-Exp_fit['r2']:
                        P0=Lin_fit['P0']
                        fit={'Lin':Lin_fit}
                        
                    else:
                        P0=Exp_fit['P0']
                        fit={'Exp':Exp_fit}
                        
                elif Lin==True and Exp==False:
                    P0=Lin_fit['P0']
                    fit={'Lin':Lin_fit}
                    
                elif Exp==True:
                    P0=Exp_fit['P0']
                    fit={'Exp':Exp_fit}
                    
                else:
                    P0=max(conc)
                    fit={None}
                    
                ###############################################################   
                """Determine the emission rate"""
                ###############################################################    
                E=0
                if End==0:
                    t = peak_time - T0
                else: t=TE-T0
                
                try:
                    if 'Exp' in fit:
                        E=Emission_determintation('Exp', t, BG_avg,P0, Exp_fit['k'],0,P)
                    elif 'Lin' in fit:
                        E=Emission_determintation('Lin', t, BG_avg,P0, Lin_fit['c'],0,P)
                    elif 'Three' in fit:
                        E=Emission_determintation('Three', t, BG_avg,P0, Three_fit['K'],Three_fit['C'],P)
                    else: E=0
                except: E=0
                    
                output[name]={'PT':num2date(ex_date+peak_time/24/3600),  #Time of peak
                    'BG':BG_avg, #Background concentration in 1/cm3
                    'Fit':fit,          #Adds the best fitting fit function
                    'P0': P0,          #Peak concentration in 1/cm3
                    'Emission_rate':E       #Average emission across the period in 1/cm3s
                     }

            else:
                output[name]={'PT':Start,
                    'BG':min(conc),
                    'P0':max(concentration[:,s]),
                    'Emission_rate':0
                    }
            BG_tot=BG_tot+output[name]['BG']
            P0_tot=P0_tot+output[name]['P0']
            E_tot=E_tot+output[name]['Emission_rate']
        #Adds together the P0, BG and emission rates for all the bin-sizes to give a total value
        output['Total']={'PT':num2date(ex_date+peak_time/24/3600),
                'BG':BG_tot,
                'P0':P0_tot,
                'Emission_rate':E_tot
                }
    else:
        
        ###################################################################
        """Find the location of the peak or valley by identifying the maximum value or minimum value"""
        ###################################################################                  
        if Peak == True:
            peak_index = np.argmax(concentration)
            peak_time = Time[peak_index]
            
            output['Total']={}
            conc=Concentration
            
            mask = conc>0
            conc=conc[mask]
            Timed=(date2num(Data[:,0])-ex_date)*24*60*60
            Timed=Timed[mask]
            ###################################################################
            """Find the location of the pre-peak minimum and determine the background value"""
            ###################################################################
            if len(conc)<10:
                output[name]={'PT':"NaN",
                    'BG':"NaN",
                    'P0':"NaN",
                    'Emission_rate':0
                    }
                pass

            #print("Pre-peak min occured at ",int(valley_time/3600),":",int(valley_time%3600/60),":",int(valley_time%60))
            BG_min_index=np.argmin(conc)
                                           
            #Determines the background averages and spread before the peak
            BG_avg = round(np.average(conc[:BG_min_index]+1),1)

            #elif Peak == True: #If a peak is found, a more thorough search starts for determining the emission rate
                
            #Defines the decay period between the peak and the return to the background level
            Decay_mask = (Time>=peak_time) & (Time<=peak_time + Decay_length*60)

            #Defines the decay data from its mask
            decay_period = Time[Decay_mask]-peak_time
            decay_conc = concentration[Decay_mask]
            Exp_fit={}
            Lin_fit={}
            Three_fit={}
            ###############################################################
            """Fit the found peak to an exponential decay assuming first order equation"""
            ###############################################################
            try:
                parameters, covariance =curve_fit(exp_func,decay_period,decay_conc,p0=[1/(60*60),decay_conc[0]],bounds=(0,np.inf))
                fit_k, fit_P0 = parameters
    
                SE = np.sqrt(np.diag(covariance))
                SE_k , SE_P0 = SE
    
                exp_fit=exp_func(decay_period,fit_k,fit_P0)
    
                # residual sum of squares
                r2_ex=R2(decay_conc,exp_fit)
                
                Exp_fit={'P0':fit_P0,
                         'k':fit_k,
                         'SE_P0': SE_P0,
                         'SE_k': SE_k,
                         'r2': round(r2_ex,3)}
                if r2_ex>0.5:
                        Exp=True
                else: Exp=False
            
            except: Exp=False
            ###############################################################
            """Fit the found peak to a decay with a second order equation"""
            ###############################################################
            try:
                parameters, covariance =curve_fit(lin_func,decay_period,decay_conc, p0=[1,decay_conc[0]],bounds=(0,np.inf))
                fit_c, fit_P0 = parameters
    
                SE = np.sqrt(np.diag(covariance))
                SE_c , SE_P0 = SE
    
                lin_fit=lin_func(decay_period,fit_c,fit_P0)
                # residual sum of squares
                r2_lin=R2(decay_conc,lin_fit)
            
                Lin_fit={'c':fit_c,
                         'P0':fit_P0,
                         'SE_m': SE_c,
                         'SE_P0': abs(1/fit_P0**2*SE_P0),
                         'r2': round(r2_lin,3)}
                if r2_lin>0.5:
                        Lin=True
                else: Lin=False
                
            except: Lin=False
            ###############################################################
            """Fit the found peak to an combined first and second order equation"""
            ###############################################################
            try:
                parameters, covariance =curve_fit(third_func,decay_period,decay_conc, p0=[1E-4,1E-5,decay_conc[0]],bounds=(0,np.inf))
                fit_K, fit_C, fit_P0 = parameters.copy()
        
                SE = np.sqrt(np.diag(covariance))
                SE_K , SE_C, SE_P0 = SE.copy()
        
                #print(k_disc)
                threehalf_fit=third_func(decay_period,fit_K,fit_C,fit_P0)
                # residual sum of squares
                r2_three=R2(decay_conc,threehalf_fit)
                Three_fit={'K':fit_K,
                           'C':fit_C,
                           'P0':fit_P0,
                           'SE_K': SE_K,
                           'SE_C': SE_C,
                           'SE_P0': SE_P0,
                           'r2': round(r2_three,3)}
                
                if r2_three>0.5:
                        Three=True
                else: Three=False
                
            except: Three=False
            ###############################################################   
            """Determine the releasd amount of particles"""
            ###############################################################
            P0=0
            if Three==True and Lin==True and Exp==True:
                if ((1-Three_fit['r2'])*1.25 < 1-Lin_fit['r2']) and (
                    (1-Three_fit['r2'])*1.5 < 1-Exp_fit['r2']):                
                    P0=Three_fit['P0']
                    fit={'Three':Three_fit}
                    
                elif ((1-Three_fit['r2'])*1.25 > 1-Lin_fit['r2']) and (
                    (1-Lin_fit['r2'])*1.25 < 1-Exp_fit['r2']): 
                    P0=Lin_fit['P0']
                    fit={'Lin':Lin_fit}
                    
                else:
                    P0=Exp_fit['P0']
                    fit={'Exp':Exp_fit}
                    
            elif Three==True and Lin==True and Exp==False:
                if (1-Three_fit['r2'])*1.25 < 1-Lin_fit['r2']:
                    P0=Three_fit['P0'] 
                    fit= {'Three':Three_fit}
                    
                else:
                    P0=Lin_fit['P0']
                    fit={'Lin':Lin_fit}
                    
            elif Three==True and Lin==False and Exp==True:
                if (1-Three_fit['r2'])*1.25 < 1-Exp_fit['r2']:
                    P0=Three_fit['P0'] 
                    fit= {'Three':Three_fit}
                    
                else:
                    P0=Exp_fit['P0']
                    fit={'Exp':Exp_fit}
                    
            elif Lin==True and Exp==True:
                if (1-Lin_fit['r2'])*1.25 < 1-Exp_fit['r2']:
                    P0=Lin_fit['P0']
                    fit={'Lin':Lin_fit}
                    
                else:
                    P0=Exp_fit['P0']
                    fit={'Exp':Exp_fit}
                    
            elif Lin==True and Exp==False:
                P0=Lin_fit['P0']
                fit={'Lin':Lin_fit}
                
            elif Exp==True:
                P0=Exp_fit['P0']
                fit={'Exp':Exp_fit}
                
            else:
                P0=max(conc)
                fit={None}
            ###############################################################   
            """Determine the emission rate"""
            ###############################################################    
            E=0
            
            if End==0:
                t = peak_time - T0
            else: t=TE-T0

            
            try:
                if 'Exp' in fit:
                    E=Emission_determintation('Exp', t, BG_avg, P0, Exp_fit['k'],0,P)
                elif 'Lin' in fit:
                    E=Emission_determintation('Lin', t, BG_avg, P0, Lin_fit['c'],0,P)
                elif 'Three' in fit:
                    E=Emission_determintation('Three', t, BG_avg, P0, Three_fit['K'],Three_fit['C'],P)
                else: E=0
            except: E=0
            
            output['Total']={'PT':num2date(ex_date+peak_time/24/3600),  #Time of peak
                'BG':BG_avg, #Background concentration in 1/cm3
                'Fit':fit,          #Adds the best fitting fit function
                'P0': P0,          #Peak concentration in 1/cm3
                'Emission_rate':E       #Average emission across the period in 1/cm3s
                 }

        else:
            output['Total']={'PT':Start,
                'BG':min(concentration),
                'P0':max(concentration),
                'Emission_rate':0
                 }

    return output
