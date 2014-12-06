import os

from astropy.io.fits import open as fits_open
import numpy as np
from astropy.convolution import convolve as ap_convolve
from astropy.convolution import Box1DKernel
from scipy.signal import medfilt # Median Filter from scipy
from scipy.interpolate import interp1d


def open_kepler(filename):
    """ Opens a Kepler filename and retrieves corrected flux """
    if os.path.exists(filename)==False:
        print "{} not found".format(filename)
        return [-1],[-1],[-1]

    hdu = fits_open(filename)
    #print hdu[1].data.dtype.names
    time = hdu[1].data.field("TIME")

    ## Use the corrected flux, not raw
    flux = hdu[1].data.field("PDCSAP_FLUX")
    flux_err = hdu[1].data.field("PDCSAP_FLUX_ERR")
    hdu.close()
    return time,flux,flux_err

def calc_sigma(time,flux,flux_err):
    """
    Following McQuillan et al. (2013), determines the standard deviation
    of the flux by iteratively smoothing the flux and removing outliers,
    then calculating the standard deviation of the residuals between the
    smoothed flux and the original flux.

    """

    x,y,yerr = np.float64(time),np.float64(flux),np.float64(flux_err)
    #print len(x),len(y),len(yerr)
    bad = (np.isnan(y) | np.isnan(yerr) | np.isnan(x))
    good = np.where(bad==False)[0]
    x,y,yerr = x[good],y[good],yerr[good]
    #print len(x),len(y),len(yerr)
    
    x1,y1,yerr1 = np.copy(x),np.copy(y),np.copy(yerr)
    # Loop three times, applying a median filter then a boxcar filter before clipping 3-sigma outliers
    for i in range(1):
        y_med = medfilt(y1,11)
        y_boxed = ap_convolve(y_med, Box1DKernel(11),boundary="extend")
        residuals = y_med - y1
        sigma_residuals = np.std(residuals)
        to_clip = (abs(residuals)>(3*sigma_residuals))
        to_keep = np.where(to_clip==False)[0]
        x1,y1,yerr1 = x1[to_keep],y1[to_keep],yerr1[to_keep]
        #print len(x1),len(y1),len(yerr1)


    # After outliers have been removed, apply the smoothing one last time
    # then determine the residuals from this smoothed curve. 
    y_smoothed1 = medfilt(y1,11)
    y_smoothed = ap_convolve(y_smoothed1, Box1DKernel(11))
    # smoothing screws up the endpoints, which screws up the std(residuals)
    residuals = (y_smoothed - y1)[10:-10]
    sigma_res = np.std(residuals)
    
    return x,y,yerr,sigma_res



def fill_gaps(time,flux,flux_err):
    """
    patches gaps in the Kepler data by linearly interpolating between the 
    points on either side of the gap, and then adding noise drawn from a 
    Gaussian with width equal to the standard deviation of the residuals
    calculated by calc_sigma()
    
    """

    x,y,yerr,sigma_res = calc_sigma(time,flux,flux_err)
    cadence = np.median(np.diff(x))
    tolerance = 5e-4 # There's some very slight variation in cadence that I'm ignoring
    missing = np.where(abs(np.diff(x)-cadence)>tolerance)[0]
    # The actual missing points will be between missing and missing+1

    number_missing = np.asarray(np.round(np.diff(x)[missing]/cadence) - 1,int)
    single_missing = missing[number_missing==1]
    mult_missing = missing[number_missing>1]
    missing_times = (x[single_missing] + x[single_missing+1])/2.0
    
    for i in mult_missing:
        #print x[i]-x[i+1],cadence
        num_missing = np.int((x[i+1] - x[i])/cadence)
        #print num_missing,x[missing],x[missing+1]
        addl_times = x[i] + [cadence*(j+1) for j in range(num_missing)]
        #print addl_times
        missing_times = np.append(missing_times, addl_times)
    #print missing_times
    
    
    # linearly interpolate between existing points
    interp_function = interp1d(x,y,kind="linear")
    replacement_flux = interp_function(missing_times) 
    #print replacement_flux
    replacement_flux = replacement_flux + np.random.normal(loc=0,scale=sigma_res,size=np.shape(missing_times))
    replacement_flux_errs = np.ones(len(missing_times))*sigma_res
    #print np.median(y_smoothed),np.median(replacement_flux)
    
    # Reassemble the time series. If a duplicate point was accidentally added, remove it
    x2 = np.append(x,missing_times)
    y2 = np.append(y,replacement_flux)
    y2err = np.append(yerr,replacement_flux_errs)
    sort_loc = np.argsort(x2)
    #print len(x2),len(sort_loc),len(y2),len(y2err)
    x3,y3,yerr3 = x2[sort_loc],y2[sort_loc],y2err[sort_loc]
    
    extra = (abs(np.diff(x3))<tolerance)
    good = np.where(extra==False)[0]
    x,y,yerr = x3[good],y3[good],yerr3[good]
    
    return x,y,yerr
