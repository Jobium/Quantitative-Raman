"""
# ==================================================

This script contains all functions to be used by other scripts.

# ==================================================
"""

import os
import numpy as np
import pandas as pd
import lmfit as lmfit
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy import stats

from itertools import combinations
from itertools import combinations_with_replacement

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Colourblind-friendly palette developed by Paul Tol ( https://personal.sron.nl/~pault/#sec:qualitative )
Colours_dark = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
Colours_light = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD']
Colour_List = Colours_dark + Colours_light

"""
functions to add:
1) outlier detection
2) basic spectral plot generator
3) mapping tools
"""



"""
# ==================================================
# classes for consistently storing and handling measurement, spectrum data
# ==================================================
"""

class Spectrum:
    # class containing all relevant info for a particular set of spectra in a measurement
    def __init__(self, key, y, parent=None, label=None, log=[], indices=None, debug=False):
        # method called when creating instance of Spectrum class
        self.key = key                  # key for referencing this spectrum (string)
        self.ylabel = label             # label for y axis (string)
        self.log = log                  # records all processing steps applied so far (list of strings)
        if isinstance(parent, Measurement):
            self.parent = parent        # reference parent Measurement instance
        else:
            self.parent = None
        y = np.asarray(y)
        if y.ndim == 1:
            # received a single spectrum in 1D array
            self.y = y[:,np.newaxis]    # y values forced into 2D array
        elif np.shape(y)[1] == 1:
            # received a single spectrum in 2D array
            self.y = y                  # y values as 2D array
        else:
            # received multiple spectra in 2D array
            self.y = y                  # y values as 2D array
        if np.all(np.asarray(indices) != None):
            if len(indices) != np.shape(self.y)[1]:
                raise Exception("Spectrum instance must receive index list that matches length of y array!")
            else:
                self.indices = indices      # track indices of spectra in measurement, 0-indexed
        else:
            # no indices passed, assume 0-N
            self.indices = range(np.shape(self.y)[1])
        # end of __init__() method

    def __iter__(self):
        # returns spectrum
        yield self.y
                
    # class methods for handling spectra
    def start(self, value=None):
        # get first x value
        if value != None:
            return value
        else:
            return self.parent.x_start
    def end(self, value=None):
        # get last x value
        if value != None:
            return value
        else:
            return self.parent.x_end
    def y_min(self, start=None, end=None):
        # get lowest y value in range
        start = self.start(start)
        end = self.end(end)
        sliced = np.where((start <= self.parent.x) & (self.parent.x <= end))
        return np.amin(self.y[sliced])
    def ymax(self, start=None, end=None):
        # get highest y value in range
        start = self.start(start)
        end = self.end(end)
        sliced = np.where((start <= self.parent.x) & (self.parent.x <= end))
        return np.amax(self.y[sliced])
    def norm(self, amax=None, amin=None, start=None, end=None):
        # get normalised spectrum
        start = self.start(start)
        end = self.end(end)
        if np.all(amax == None):
            amax = np.amax(self.y)
        if np.all(amin == None):
            amin = np.amin(self.y)
        sliced = np.where((start <= self.parent.x) & (self.parent.x <= end))
        return (self.y[sliced] - amin) / (amax - amin)
    def mean(self, start=None, end=None):
        # get mean spectrum (or mean value)
        start = self.start(start)
        end = self.end(end)
        sliced = np.where((start <= self.parent.x) & (self.parent.x <= end))
        return np.mean(self.y[sliced], axis=1)
    def median(self, start=None, end=None):
        # get median spectrum (or median value)
        start = self.start(start)
        end = self.end(end)
        sliced = np.where((start <= self.parent.x) & (self.parent.x <= end))
        return np.median(self.y[sliced], axis=1)
    
    # end of Spectrum class

class Measurement:
    # class containing all recorded data for a measurement object
    def __init__(self, **kwargs):
        # method called when creating instance of measurement class
        # check that essential parameters included in kwargs
        check = ['ID', 'title', 'sample', 'technique', 'x', 'y', 'Fig_dir', 'Out_dir']
        check = [s for s in check if s not in kwargs]
        if len(check) > 0:
            raise Exception("Error! Cannot create a Measurement instance without the following arguments: %s" % ",".join(check))
        if 'technique' in kwargs.keys():
            if kwargs['technique'] == 'Raman' and 'laser_wavelength' not in kwargs:
                raise Exception("Error! Cannot create a Measurement instance for a Raman spectrum without specifying laser_wavelength!")
        del check
        # add parameters to instance
        for k in kwargs:
            # dynamically add properties based on what was passed to measurement()
            setattr(self, k, kwargs[k])
            
        # handle input x properties
        self.x = kwargs['x']
        if kwargs['technique'].lower() in ['f', 'fluor', 'fluorescence', 'p', 'pl', 'photolum', 'photoluminescence']:
            # fluorescence, wavelength domain
            self.technique = 'Photoluminescence'
            self.xlabel = 'Wavelength (nm)'
            self.frequency = wavelength2frequency(kwargs['x'])
            self.wavelength = self.x
        elif kwargs['technique'].lower() in ['r', 'raman']:
            # Raman measurements, raman shift domain
            self.technique = 'Raman'
            self.xlabel = 'Raman Shift (cm$^{-1}$)'
            self.raman_shift = self.x
            self.wavelength = shift2wavelength(kwargs['x'], kwargs['laser_wavelength'])
            self.laser_wavelength = int(kwargs['laser_wavelength'])
        else:
            # default to Infrared, frequency domain
            self.technique = 'Infrared'
            self.xlabel = 'Frequency (cm$^{-1}$)'
            self.frequency = self.x
            self.wavelength = frequency2wavelength(kwargs['x'])
        self.x_start = np.amin(self.x)
        self.x_end = np.amax(self.x)
            
        # add placeholder info if needed
        props = {
            'ykey': 'y',
            'ylabel': "Intensity",
            'log': [],
            'generate_average': False,  # generate average spectrum?
            'output_folder': '',        # 
            'debug': False              # print debug messages during Measurement creation?
        }
        for key, prop in props.items():
            if key not in kwargs:
                kwargs[key] = prop
            
        # convert input y into a Spectrum instance
        self.add_spectrum(kwargs['ykey'], kwargs['y'], label=kwargs['ylabel'], log=kwargs['log'], debug=kwargs['debug'])
        if kwargs['generate_average'] == True:
            # add Spectrum instance for average spectrum as well
            y_av = self[kwargs['ykey']].mean()
            self.add_spectrum('%s_av' % kwargs['ykey'], y_av, label='Average'+kwargs['ylabel'], log=kwargs['log'] + ['averaged %s spectra' % np.shape(kwargs['y'])[1]], debug=kwargs['debug'])
        
        if type(kwargs['output_folder']) == str and kwargs['output_folder'] != '':
            # use specified figure, output folder structure
            if kwargs['output_folder'][-1] != '/':
                kwargs['output_folder'] += '/'
            self.fig_dir = '%s%s/' % (kwargs['Fig_dir'], kwargs['output_folder'])
            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir)
            self.out_dir = '%s%s/' % (kwargs['Out_dir'], kwargs['output_folder'])
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
        else:
            # use default figure, output folder structure
            if self.technique == 'Raman':
                # for Raman: /sample/wavelength/measurement/
                self.fig_dir = '%s%s/%snm/%s/' % (kwargs['Fig_dir'], kwargs['sample'], kwargs['laser_wavelength'], kwargs['title'])
                if not os.path.exists(self.fig_dir):
                    os.makedirs(self.fig_dir)
                self.out_dir = '%s%s/%snm/%s/' % (kwargs['Out_dir'], kwargs['sample'], kwargs['laser_wavelength'], kwargs['title'])
                if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
            else:
                # for FTIR: /sample/measurement/
                self.fig_dir = '%s%s/%s/' % (kwargs['Fig_dir'], kwargs['sample'], kwargs['title'])
                if not os.path.exists(self.fig_dir):
                    os.makedirs(self.fig_dir)
                self.out_dir = '%s%s/%s/' % (kwargs['Out_dir'], kwargs['sample'], kwargs['title'])
                if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
            
        # remove unneeded parameters
        for prop in ['Fig_dir', 'Out_dir', 'ykey', 'ylabel', 'generate_average', 'make_folders', 'debug']:
            if prop in self.__dict__.keys():
                delattr(self, prop)
                
        if kwargs['debug'] == True:
            print('    end of Measurement creation method')
            
        # end of __init__() method
        
    def __getitem__(self, item):
        # method called when subscripting the instance, e.g. measurement1[property]
        return getattr(self, item)  # return specified property
        
    def __call__(self, i, j, normalise=False):
        # method called when referencing the instance like a function, e.g. measurement1(x_key, y_key)
        if normalise == True and hasattr(self, j) == True:
                # return specified x and normalised y properties
                return self[i], self[j].norm()
        else:
            # return specified x and y properties
            return np.copy(self[i]), np.copy(self[j].y)
        
    def add_spectrum(self, key, y, label=None, log=[], indices=None, debug=False):
        # function for creating a new spectrum instance inside measurement instance
        if hasattr(self, str(key)) == True and debug == True:
            print("CAUTION: overwriting %s" % str(key))
        # finish add_spectrum() method by adding Spectrum instance to parent Measurement instance
        parent = self
        setattr(self, str(key), Spectrum(key, np.asarray(y), parent=parent, label=label, log=log, indices=indices, debug=debug))
        if debug == True:
            print('    end of add_spectrum method')
    
    # end of Measurement class
    
def get_measurements(data_dict, IDs):
    # function for returning a list of measurement instances by ID
    output = []
    for ID in IDs:
        if ID not in data_dict.keys():
            raise Exception("ID %s not in data dict!" % ID)
        else:
            output.append(data_dict[ID])
    return output

def get_fitted_peak(measurement, key, prop=None, peak_index=None, position=None, spec_index=None, debug=False):
    # function for returning a particular fitted peak property from one or more measurement instances
    output = []
    if hasattr(measurement, key) == False:
        raise Exception("no %s data for measurement %s!" % (key, measurement.title))
    else:
        peak_data = getattr(measurement, key)
        spec_indices = np.unique(peak_data['spec_index'])
        for i in spec_indices:
            # for each fitted spectrum, get required info
            temp = peak_data[peak_data['spec_index'] == i]
            # find appropriate row index
            if peak_index != None:
                # use peak index, starting at lowest index for this spec
                index = temp.index.values[peak_index]
            elif position != None:
                # get peak nearest to specified position
                index = np.argmin(np.abs(temp['centers'] - float(position)))
            if prop != None:
                # return specified column, index
                temp = temp.loc[index, prop]
            else:
                # return specified index, all columns
                temp = temp.loc[index]
            output.append(np.asarray(temp))
    return np.asarray(output)



"""
# ==================================================
# functions for converting between raman shift, wavelength, frequency, and energy
# ==================================================
"""

def wavelength2shift(W, ex=785.):
    # convert wavelength (in nm) to raman shift (in cm-1)
    if type(ex) in [str, int, np.str_]:
        ex = float(ex)
    S = ((1./ex) - (1./W)) * (10**7)
    return S

def shift2wavelength(S, ex=785.):
    # convert raman shift (in cm-1) to wavelength (in nm)
    if type(ex) in [str, int, np.str_]:
        ex = float(ex)
    W = 1./((1./ex) - S/(10**7))
    return W

def wavelength2frequency(W):
    # convert wavelength (in nm) to absolute frequency (in cm-1)
    return (1./W) * (10**7)

def frequency2wavelength(F):
    # convert absolute frequency (in cm-1) to wavelength (in nm)
    return (1./F) / (10**-7)

def shift2frequency(S, ex=785.):
    # convert raman shift (in cm-1) to absolute frequency (in cm-1)
    if type(ex) in [str, int, np.str_]:
        ex = float(ex)
    W = 1./((1./ex) - S/(10**7))
    return wavelength2frequency(W)
    
def frequency2shift(F, excitation=785.):
    # convert absolute frequency (in cm-1) to raman shift (in cm-1)
    if type(ex) in [str, int, np.str_]:
        ex = float(ex)
    W = frequency2wavelength(F)
    S = ((1./ex) - (1./W)) * (10**7)
    return S

def wavelength2energy(W):
    # convert wavelength (in nm) to energy (in J)
    hc = 1.98644586*10**-25 # in J meters
    return hc / (np.asarray(W)/10**9)

def energy2wavelength(E):
    # convert photon energy (in J) to wavelength (in nm)
    hc = 1.98644586*10**-25 # in J meters
    return (hc / np.asarray(E)) * 10**9

def photon_count(W, E_tot):
    # convert a total laser energy (in J) at given wavelength (in nm) to photon count
    hc = 1.98644586*10**-25 # in J meters
    E_p = hc / (np.asarray(W)/10**9)
    return np.asarray(E_tot) / E_p

def intensity2snr(I, noise):
    # convert intensity to a signal:noise ratio
    return I / noise

def snr2intensity(SNR, noise):
    # convert a signal:noise ratio to intensity
    return SNR * noise

def transmittance2absorbance(T, percentage=True):
    # convert transmittance (in % or a unit interval) to absorbance (in arbitrary units)
    if np.any(T > 1.):
        A = 2 - np.log10(T)
    else:
        A = -np.log10(T)
    return A

def absorbance2transmittance(A, percentage=True):
    # convert absorbance (in arbitrary units) to transmittance (in % or a unit interval)
    if percentage == True:
        T = 10**(2-A)
    else:
        T = 10**(-A)
    return T

"""
# ==================================================
# functions for smoothing, trimming and normalising spectra
# ==================================================
"""

def smooth_spectrum(y, window_length, polyorder):
    # function for smoothing data based on Savitsky-Golay filtering
    if window_length % 2 != 1:
        window_length += 1
    y_smooth = savgol_filter(y, window_length, polyorder)
    return y_smooth

def slice_spectrum(x, y, std=None, start=400, end=4000, normalise=False, tabs=0, debug=False):
    # function for slicing a spectrum to a specific range and normalising if needed
    # argument summary:
    #   - x:            x values (1D or 2D array)
    #   - y:            y values (1D or 2D array)
    #   - std:          st. dev. of y (optional, 1D or 2D array)
    #   - start:        start of x axis range to process (int or float)
    #   - end:          end of x axis range to process (int or float)
    #   - normalise:    normalise data to maximum (boolean)
    #   - debug:        print debug messages (boolean)
    if debug == True:
        print("    "*(tabs) + "slicing spectrum...")
        print("    "*(tabs+1) + "input arrays:", np.shape(x), np.shape(y))
        print("    "*(tabs+1) + "y array ndim:", y.ndim)
    if start > np.amax(x) or end < np.amin(x):
        print("    "*(tabs+1) + "slice range:", start, end)
        print("    "*(tabs+1) + "spectrum x range:", np.amin(x), np.amax(x))
        raise Exception("attempting to slice spectrum beyond x range!")
    if hasattr(std, "__len__") and np.all(std != None):
        std_check = True
    else:
        std_check = False
    if debug == True and std_check == True:
        print("    "*(tabs+1) + "stdev array:", np.shape(std))
    y_max = []
    if x.ndim == y.ndim and y.ndim > 1:
        # multiple spectra, each with own x and y
        x_slice = []
        y_slice = []
        std_slice = []
        for i in range(np.shape(y)[1]):
            sliced = np.ravel(np.where((start <= x[:,i]) & (x[:,i] <= end)))
            x_slice.append(x[sliced,i])
            y_max.append(np.amax(y[sliced,i]))
            if normalise == True:
                y_slice.append(y[sliced,i] / y_max[-1])
                if std_check == True:
                    std_slice.append(std[sliced,i] / y_max[-1])
            else:
                y_slice.append(y[sliced,i])
                if std_check == True:
                    std_slice.append(std[sliced,i])
    elif y.ndim > 1:
        # multiple spectra with common x axis
        sliced = np.ravel(np.where((start <= x) & (x <= end)))
        x_slice = x[sliced]
        y_slice = []
        std_slice = []
        for i in range(np.shape(y)[1]):
            y_max.append(np.amax(y[sliced,i]))
            if normalise == True:
                y_slice.append(y[sliced,i] / y_max[-1])
                if std_check == True:
                    std_slice.append(std[sliced,i] / y_max[-1])
            else:
                y_slice.append(y[sliced,i])
                if std_check == True:
                    std_slice.append(std[sliced,i])
    else:
        # single spectrum
        sliced = np.ravel(np.where((start <= x) & (x <= end)))
        x_slice = x[sliced]
        y_max.append(np.amax(y[sliced]))
        if normalise == True:
            y_slice = y[sliced] / y_max[-1]
            if std_check:
                std_slice = std[sliced] / y_max[-1]
        else:
            y_slice = y[sliced]
            if std_check:
                std_slice = std[sliced]
    x_slice = np.asarray(x_slice)
    y_slice = np.asarray(y_slice)
    if std_check == True:
        std_slice = np.asarray(std_slice)
        if debug == True:
            print("    "*(tabs+1) + "sliced arrays:", np.shape(x_slice), np.shape(y_slice), np.shape(std_slice))
            print("    "*(tabs+1) + "x range:", np.amin(x_slice), np.amax(x_slice))
            print("    "*(tabs+1) + "y range:", np.amin(y_slice), np.amax(y_slice), np.amax(y_max))
        return x_slice, y_slice, std_slice
    else:
        if debug == True:
            print("    "*(tabs+1) + "sliced arrays:", np.shape(x_slice), np.shape(y_slice))
            print("    "*(tabs+1) + "x range:", np.amin(x_slice), np.amax(x_slice))
            print("    "*(tabs+1) + "y range:", np.amin(y_slice), np.amax(y_slice), np.amax(y_max))
        return x_slice, y_slice

def normalise_spectrum(x, y, std=None, y_max=None, max_start=400, max_end=4000, debug=False):
    # function for normalising a spectrum
    # argument summary:
    #   - x:        x values (1D or 2D array)
    #   - y:        y values (1D or 2D array)
    #   - std:      st. dev. of y (optional, 1D or 2D array)
    #   - start:    start of x axis range to process (int or float)
    #   - end:      end of x axis range to process (int or float)
    #   - debug:    print debug messages (boolean)
    # check if stdev included
    if np.all(std != None):
        std_check = True
    else:
        std_check = False
    # check y_max
    if y_max != None:
        # use input y_max
        pass
    else:
        # instead use max intensity from range max_start - max_end
        y_max = find_max(x, y, max_start, max_end)[1]
    if std_check == True:
        return y / y_max, std / y_max
    else:
        return y / y_max
    
def interpolate_spectra(measurements, x_key='raman_shift', y_key='y_av_sub', alt_key='y_av', normalise=False, x_list=[], start=None, end=None, tabs=0, debug=False):
    # function for interpolating multiple measurement instances so that their x values match
    # argument summary:
    #   - measurements:     list of Measurement instances to average
    #   - x_key:            key for x axis values to get from measurement (string)
    #   - y_key:            key for y axis values to get from measurement (string)
    #   - alt_key:          alternate key for y axis values to get if y_key is not available
    #   - normalise:        normalise spectra? (boolean)
    #   - x_list:           optional list of x positions for interpolating data (list of ints/floats)
    #   - start:            trim x range to start here
    #   - end:              trim x range to end here
    
    # ==================================================
    # get data from measurement objects
    x_temp = []
    y_temp = []
    for measurement in measurements:
        if hasattr(measurement, x_key) == False:
            raise Exception("    "*(tabs+1) + "%s not in %s measurement data!" % (x_key, measurement.title))
        elif hasattr(measurement, y_key) == False and hasattr(measurement, alt_key) == False:
            raise Exception("    "*(tabs+1) + "%s and %s not in %s measurement data!" % (y_key, alt_key, measurement.title))
        if hasattr(measurement, y_key) == True:
            key = y_key
        else:
            key = alt_key
        x, y = measurement(x_key, key)
        x_temp.append(x)
        if y.ndim > 1:
            # y is multi-spec measurement, reduce to average
            y = np.mean(y, axis=1)
        if normalise == True:
            # normalise spectrum first
            y_temp.append(y/np.amax(y))
        else:
            y_temp.append(y)
        
    # ==================================================
    # determine x list for interpolation and trim to start-end
    if len(x_list) > 0:
        # use specified x values for interpolation
        x_interp = x_list
    else:
        # all passed x values are the same, no interpolation needed
        x_interp = x_temp[0]
    if start == None:
        start = np.amin(x_interp)
    if end == None:
        end = np.amax(x_interp)
    sliced = np.where((start <= x_interp) & (x_interp <= end))
    x_interp = np.asarray(x_interp)[sliced]
    if debug == True:
        print("    "*(tabs+1), "interpolating spectra...")
        print("    "*(tabs+2), np.shape(x_interp), np.shape(x_temp), np.shape(y_temp))
    
    # ==================================================
    # interpolate y spectra based on x_interp
    y_interp = [np.interp(x_interp, x_temp[i], y_temp[i]) for i in range(len(measurements))]
    y_interp = np.vstack(y_interp)
    
    # return interpolated x, interpolated y values
    return x_interp, y_interp

def average_spectrum(x, y, indices=[], debug=False):
    # function for averaging together a subset of a measurement's Spectrum instance
    # argument summary:
    #   - x:        x values (1D array)
    #   - y:        y values (2D array)
    #   - indices:  list of indices to use, or [] for all
    #   - debug:    print debug messages (boolean)
    
    # trim y array to indices if provided
    if len(indices) > 0:
        y = y[:,indices]
    
    # average y array if possible
    if y.ndim > 1:
        y_av = np.mean(y, axis=1)
    else:
        y_av = y
    
    if debug == True:
        print(np.shape(x), np.shape(y), np.shape(y_av))

    # return x, average y values
    return x, y_av
    
def average_spectra(measurements, x_key='raman_shift', y_key='y_av_sub', alt_key='y_av', normalise=False, x_list=[], start=None, end=None, tabs=0, debug=False):
    # function for averaging together multiple separate measurement instances
    # argument summary:
    #   - measurements:     list of Measurement instances to average
    #   - x_key:            key for x axis values to get from measurement (string)
    #   - y_key:            key for y axis values to get from measurement (string)
    #   - normalise:        normalise spectra before averaging? (boolean)
    #   - x_list:           optional list of x positions for interpolating data (list of ints/floats)
    #   - start:            trim x range to start here
    #   - end:              trim x range to end here
    
    # ==================================================
    # get interpolated x,y data
    x, y = interpolate_spectra(measurements, x_key=x_key, y_key=y_key, alt_key=alt_key, normalise=normalise, x_list=x_list, start=start, end=end, tabs=tabs, debug=debug)
    
    # ==================================================
    # average interpolated spectra
    y_av = np.mean(y, axis=0)
    
    if debug == True:
        print(np.shape(x), np.shape(y), np.shape(y_av))
    
    # return interpolated x, average interpolated y values
    return x, y_av

"""
# ==================================================
# functions for fitting and subtracting a baseline
# ==================================================
"""

def average_list(x, y, point_list, window, debug=False):
    # function for taking a set of user-defined points and creating arrays of their average x and y values
    if debug == True:
        print("        ", point_list)
    x_averages = np.zeros_like(point_list, dtype=float)
    y_averages = np.zeros_like(point_list, dtype=float)
    point_num = 0
    for i in range(np.size(point_list)):
        point_num += 1
        x_averages[i], y_averages[i] = local_average(x, y, point_list[i], window)
        if debug == True:
            print("        point", str(point_num), ": ", x_averages[i], y_averages[i])
    return x_averages, y_averages

def local_average(x, y, center, window):
    # function for finding the average position for a set of points with a given center +/- window
    center_ind = np.argmin(np.absolute(x - center))
    start_ind = center_ind - window
    end_ind = center_ind + window
    if start_ind < 0:
        start_ind = 0
    if end_ind > len(y)-1:
        end_ind = len(y)-1
    x_temp = x[start_ind:end_ind]
    y_temp = y[start_ind:end_ind]
    x_average = (np.average(x_temp))
    y_average = (np.average(y_temp))
    return x_average, y_average

def f_polynomial(x, *params):
    # function for generating an exponential baseline
    y = params[0]
    for i in range(1, len(params)):
        y += params[i] * x**i
    return y

def polynomial_fit(x_averages, y_averages, sigma, order=15, debug=False):
    # function for fitting selected average data-points using a polynominal function
    if len(x_averages) > int(order):
        guess = np.zeros((int(order)))
    else:
        guess = np.zeros_like(x_averages)
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polynomial, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print("        fitted parameters:", fit_coeffs)
    return fit_coeffs, fit_covar

def subtract_baseline(measurement, x_key='raman_shift', y_key='y', output_key='y_sub', x_list=[], base='polynomial', order=15, find_minima=True, window=35, fixed_ends=True, baseline_by_section=False, splits=[], save_plot=False, show_plot=True, plot_name=None, log=None, tabs=0, debug=False):
    # script for subtracting a baseline by fitting it with a polynomial
    # argument summary:
    #   - measurement:      Measurement object containing data to be processed
    #   - x_key:            key for x axis values to get from measurement (string)
    #   - y_key:            key for y axis values to get from measurement (string)
    #   - output_key:       key for recording results in measurement (string)
    #   - x_list:           list of x positions to evaluate when fitting reference to target (list or 1D array)
    #   - base:             function for generating baseline curve (string)
    #   - order:            order for polynomial (int)
    #   - find_minima:      find minimum value within window around each point? (boolean)
    #   - window:           window for finding minima around each point, in x axis units (int or float)
    #   - fixed_ends:       force baseline to pass through ends of spectrum? (boolean)
    #   - baseline_by_section:  divide spectrum into user-defined sections for baselining separately? (boolean)
    #   - splits:           points on x axis for dividing spectrum into sections (list of ints or floats)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    #   - debug:            print debug messages? (boolean)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(tabs) + "running baseline subtraction on measurement %s" % measurement.title)
    if len(x_list) == 0:
        raise Exception("    "*(tabs+1) + "no x positions passed to subtract_baseline function!")
    if isinstance(measurement, Measurement) == False:
        raise Exception("    "*(tabs+1) + "first argument passed to subtract_baseline function must be a Measurement instance!")
    elif hasattr(measurement, x_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % x_key)
    elif hasattr(measurement, y_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % y_key)
    if baseline_by_section == True and len(splits) == 0:
        raise Exception("    "*(tabs+1) + "if baseline_by_section argument is true, splits argument must contain at least one value!")
    if type(log) != str:
        log = 'baselined using %s' % base
    
    # ==================================================
    # check if plotting is required
    plot = False
    if save_plot == True or show_plot == True:
        plot = True
    if plot == True and plot_name == None:
        # use default suffix for baseline
        plot_name = "baselined"
        
    # ==================================================
    # get data from measurement object
    x, y = measurement(x_key, y_key)
    old_log = measurement[y_key].log
    if debug == True:
        print("    "*(tabs+1) + "input arrays:", x_key, np.shape(x), y_key, np.shape(y))
        print("    "*(tabs+1) + "previous y processing:", "\n".join(old_log))
        
    # ==================================================
    # smooth data for fitting
    if y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        y_s = smooth_spectrum(y, 5, 3)[:,np.newaxis]
    elif np.shape(y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
        if plot == True:
            print("    "*(tabs+1) + "cannot plot results of peak subtraction for multi-spec measurement!")
            plot = False
    
    # ==================================================
    # begin figure if necessary
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)  # ax1: original target spec overlaid with rescaled reference spec
        ax2 = plt.subplot(212, sharex=ax1)  # ax2: target spec after subtraction
        ax1.set_title("%s\nBaseline Subtraction" % (measurement.title))
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
        ax1.set_xlim(measurement.x_start, measurement.x_end)
        # plot original spectrum in ax1
        ax1.plot(x, y, 'k', label='input spectrum')
    
    # ==================================================
    # trim x_list to points that fall within spectrum x range
    x_list = np.asarray(x_list)
    x_list = np.sort(x_list[np.where((np.amin(x)+10 <= x_list) & (x_list <= np.amax(x)-10))])
    # reduce polynomial order if it exceeds number of points-1
    if len(x_list)-1 < order:
        order = len(x_list)-1
    if debug == True:
        print("    "*(tabs+1) + "x range: %0.f - %0.f" % (np.amin(x), np.amax(x)))
        print("    "*(tabs+1) + "x_list for subtraction:", x_list)
        if base == 'polynomial':
            print("    "*(tabs+1) + "polynomial order:", order)
        
    # ==================================================
    # divide up data into sections (default: 1)
    if baseline_by_section == True and len(splits) > 0:
        fixed_ends = True
        # convert input splits (in x axis units) into a sorted list of indices for use by np.split()
        section_splits = np.asarray([np.argmin(np.abs(x - float(point))) for point in splits if point > np.amin(x) and point < np.amax(x)])
        if debug == True:
            print("    "*(tabs+1) + "spectrum split at x=", ", ".join([str(split) for split in splits]))
    else:
        section_splits = 1
        if debug == True:
            print("    "*(tabs+1) + "no spectrum splitting")
    x_splits = np.split(x, section_splits, axis=0)
    y_splits = np.split(y_s, section_splits, axis=0)
    
    # ==================================================
    # do baseline fitting, section by section
    fitted_baseline = [np.zeros_like(split) for split in y_splits]
    for i2, x_slice, y_slice in zip(range(0, len(x_splits)), x_splits, y_splits):
        # for each section
        if debug == True:
            print("    "*(tabs+1) + "section %s x, y arrays:" % i2, np.shape(x_slice), np.shape(y_slice))
            print("    "*(tabs+2) + "section %s x range: %0.1f - %0.1f cm-1" % (i2, np.amin(x_slice), np.amax(x_slice)))
        for i in range(np.shape(y_s)[1]):
            # for each spectrum in slice
            if debug == True:
                print("    "*(tabs+2) + "spectrum %s" % i)
            
            # ==================================================
            # trim x_list to section
            local_x_list = np.asarray(x_list)[np.where((np.amin(x_slice) <= np.asarray(x_list)) & (np.asarray(x_list) < np.amax(x_slice)))]
            if find_minima == True:
                # find minimum value within window around each point in local_x_list 
                points = []
                for point in local_x_list:
                    points.append(find_min(x_slice, y_slice[:,i], point-window, point+window)[0])
            else:
                points = local_x_list
            
            # ==================================================
            # create arrays of average values for each point +/-5 pixels
            x_averages, y_averages = average_list(x_slice, y_slice[:,i], points, 5, debug=debug)
            # add fixed first and last points if applicable, and create sigma array for point weighting
            sigma = np.ones_like(y_averages)
            if fixed_ends == True:
                for index in [5, -6]:
                    x_0, y_0 = local_average(x_slice, y_slice[:,i], x_slice[index], 5)
                    x_averages = np.append(x_averages, x_0)
                    y_averages = np.append(y_averages, y_0)
                    sigma = np.append(sigma, 0.1)
            # sort x_list into ascending order
            sort = np.argsort(x_averages)
            x_averages = x_averages[sort]
            y_averages = y_averages[sort]
            sigma = sigma[sort]

            # ==================================================
            # attempt to fit section data using specified base function
            while True:
                try:
                    # first attempt (original order)
                    local_order = order
                    if local_order > len(y_averages)-1:
                        local_order = len(y_averages)-1
                    fit_coeffs, fit_covar = polynomial_fit(x_averages, y_averages, sigma, order=local_order, debug=debug)
                    basefit = f_polynomial(x_slice, *fit_coeffs)
                    if debug == True:
                        print("    "*(tabs+1) + "basefit:", np.shape(basefit))
                        print("    "*(tabs+1) + "baseline section:", np.shape(fitted_baseline[i2][:,i]))
                    fitted_baseline[i2][:,i] = basefit
                    break
                except Exception as e:
                    if debug == True:
                        print("    "*(tabs+1) + "something went wrong! Exception:", e)
                        print("    "*(tabs+2) + "attempting another fit with reduced polynomial order...")
                    try:
                        # second attempt (reduced order)
                        if local_order - 1 > 1:
                            local_order -= 1
                        fit_coeffs, fit_covar = polynomial_fit(x_averages, y_averages, sigma, order=local_order, debug=debug)
                        basefit = f_polynomial(x_slice, *fit_coeffs)
                        if debug == True:
                            print("    "*(tabs+2) + "basefit:", np.shape(basefit))
                            print("    "*(tabs+2) + "baseline section:", np.shape(fitted_baseline[i2][:,i]))
                        fitted_baseline[i2][:,i] = basefit
                        if plot == True:
                            ax1.scatter(x_averages, y_averages, c=Colours_light[i % len(Colours_light)], zorder=3)
                            ax1.plot(x_slice, basefit, Colours_light[i % len(Colours_light)], zorder=2)
                        break
                    except Exception as e:
                        if debug == True:
                            print("    "*(tabs+2) + "something went wrong! Exception:", e)
                            print("    "*(tabs+3) + "attempting third fit with reduced polynomial order...")
                        try:
                            # second attempt (reduced order)
                            if local_order - 1 > 1:
                                local_order -= 1
                            fit_coeffs, fit_covar = polynomial_fit(x_averages, y_averages, sigma, order=local_order, debug=debug)
                            basefit = f_polynomial(x_slice, *fit_coeffs)
                            if debug == True:
                                print("    "*(tabs+3) + "basefit:", np.shape(basefit))
                                print("    "*(tabs+3) + "baseline section:", np.shape(fitted_baseline[i2][:,i]))
                            fitted_baseline[i2][:,i] = basefit
                            if plot == True:
                                ax1.scatter(x_averages, y_averages, c=Colours_light[i % len(Colours_light)], zorder=3)
                                ax1.plot(x_slice, basefit, Colours_light[i % len(Colours_light)], zorder=2)
                            break
                        except Exception as e:
                            if debug == True:
                                print("    "*(tabs+3) + "something went wrong again! Exception:", e)
                                print("    "*(tabs+4) + "3 attempts failed. Taking minimum for flat baseline instead.")
                            # both attempts failed, use flat baseline instead
                            fitted_baseline[i2][:,i] = np.full_like(y_slice, np.amin(y_slice))
                            break
            # add section baseline to plot
            if plot == True and single_spec == True:
                ax1.scatter(x_averages, y_averages, c=Colours_light[i2 % len(Colours_light)], zorder=3)
                ax1.plot(x_slice, fitted_baseline[i2][:,0], c=Colours_light[i2 % len(Colours_light)], zorder=2, label='section %s' % i2)
    
    if debug == True:
        print("    "*(tabs+1) + "split baseline arrays:", [np.shape(split) for split in fitted_baseline])
    
    # ==================================================
    # concatenate sections into a single baseline and subtract from data
    fitted_baseline = np.concatenate(fitted_baseline, axis=0)
    y_sub = y - fitted_baseline
    
    # ==================================================
    # save subtracted spectrum to measurement
    measurement.add_spectrum(output_key, y_sub, label="Baselined Intensity (counts)", log=old_log + [log])
    if debug == True:
        print("    "*(tabs+1) + "output %s array:" % output_key, np.shape(y_sub))
        print("    "*(tabs+1) + "output log:", old_log + [log])
    
    # ==================================================
    # complete figure if required
    if plot == True and single_spec == True:
        x_min = np.amin(x_averages)
        x_max = np.amax(x_averages)
        ax1.set_xlim(x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min))
        ax2.set_xlim(x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min))
        y_min = np.amin(y_averages)
        y_max = np.amax(y_averages)
        ax1.set_ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
        # plot subtracted spectrum in ax2
        ax2.plot(x, y_sub, 'k', label='baselined')
        # finish figure
        ax1.legend()
        ax2.legend()
        plt.minorticks_on()
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_%s.png" % (measurement.fig_dir, measurement.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(tabs+1) + "end of subtract_baseline function")

"""
# ==================================================
# functions for fitting and subtracting a reference spectrum
# ==================================================
"""

def reference_curve(params, ref_x, ref_y):
    # function for plotting resulting curve from reference_fit() function
    ref_y_rescaled = ref_y * params['scale_factor']
    # add polynomial
    for i in range(1, params['poly_order'].value+1):
        ref_y_rescaled += params['p'+str(i)] * ref_x**i
    return ref_y_rescaled

def reference_fit(params, ref_x, ref_y, target_x, target_y):
    # function for use with LMFIT minimize(), provides difference between target data and rescaled reference data
    # rescale ref_y using scale_factor parameter
    ref_y_rescaled = ref_y * params['scale_factor']
    # add polynomial
    for i in range(1, params['poly_order'].value+1):
        ref_y_rescaled += params['p'+str(i)] * ref_x**i
    # get ref - target difference in x,y
    dx = ref_x - target_x
    dy = ref_y_rescaled - target_y
    # return euclidean distance between ref and target
    return np.sqrt(dx**2 + dy**2)
            
def subtract_reference(ref, target, x_key='raman_shift', y_key='y_av_sub', alt_key='y_sub', output_key='y_refsub', x_list=[], add_polynomial=False, poly_order=5, save_plot=False, show_plot=True, plot_name=None, log=None, tabs=0, debug=False):
    # script for subtracting a scaled reference spectrum by fitting it to a set of points
    # argument summary:
    #   - ref:              Measurement object containing data for reference spectrum
    #   - target:           Measurement object containing data for target spectrum
    #   - x_key:            list of x positions to evaluate when fitting reference to target (list or 1D array)
    #   - y_key:            key for x axis values in spec dict (string)
    #   - y_type:           key for y axis values in spec dict (string)
    #   - output_key:       key for outputting results into spec dict (string)
    #   - debug:            print debug messages? (boolean)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(tabs) + "running reference subtraction on measurement %s using %s" % (target.title, ref.name))
    if type(x_list) != list or len(x_list) == 0:
        raise Exception("    "*(tabs+1) + "no x positions passed to subtract_reference function!")
    if isinstance(ref, Measurement) == False:
        raise Exception("    "*(tabs+1) + "ref argument passed to subtract_reference function must be a Measurement object!")
    elif hasattr(ref, x_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in ref data!" % x_key)
    elif hasattr(ref, 'y_av_sub') == False:
        raise Exception("    "*(tabs+1) + "y_av_sub not in ref data!")
    if isinstance(target, Measurement) == False:
        raise Exception("    "*(tabs+1) + "target argument passed to subtract_reference function must be a Measurement object!")
    elif hasattr(target, x_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % x_key)
    elif hasattr(target, y_key) == False and hasattr(target, alt_key) == False:
        raise Exception("    "*(tabs+1) + "%s and %s not in measurement data!" % (y_key, alt_key))
    if type(log) != str:
        log = '%s fitted and subtracted' % ref.name
        
    # ==================================================
    # check if plotting is required
    plot = False
    if save_plot == True or show_plot == True:
        plot = True
    if plot == True and plot_name == None:
        plot_name = ref['name']+"-sub"
        
    # ==================================================
    # get spectrum data for ref
    ref_x, ref_y = ref(x_key, 'y_av_sub')
    if ref_y.ndim > 1:
        ref_y = np.mean(ref_y, axis=1)
    ref_y /= np.amax(ref_y)
    
    # ==================================================
    # get spectrum data for ref
    if hasattr(target, y_key):
        key = y_key
    else:
        key = alt_key
    target_x, target_y = target(x_key, key)
    indices = target[key].indices
    old_log = target[key].log
    if debug == True:
        print("    "*(tabs+1) + "input ref arrays:", np.shape(ref_x), np.shape(ref_y))
        print("    "*(tabs+1) + "input target arrays:", np.shape(target_x), np.shape(target_y))
        print("    "*(tabs+1) + "previous y processing:", "\n".join(old_log))
        
    # ==================================================
    # smooth data for fitting
    if target_y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        target_y_s = smooth_spectrum(target_y, 5, 3)[:,np.newaxis]
    elif np.shape(target_y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        target_y_s = np.zeros_like(target_y)
        for i in range(np.shape(target_y)[1]):
            target_y_s[:,i] = smooth_spectrum(target_y[:,i], 5, 3)
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        target_y_s = np.zeros_like(target_y)
        for i in range(np.shape(target_y)[1]):
            target_y_s[:,i] = smooth_spectrum(target_y[:,i], 5, 3)
        if plot == True:
            print("    "*(tabs+1) + "cannot plot results of subtraction for multi-spec measurement!")
            plot = False
    if debug == True:
        print("    "*(tabs+1) + "smoothed y array:", np.shape(target_y_s))
            
    # ==================================================
    # trim x_list to points that fall within spectrum x range
    x_list = np.asarray(x_list)
    x_list = np.sort(x_list[np.where((np.amin(ref_x)+10 <= x_list) & (x_list <= np.amax(ref_x)-10) & (np.amin(target_x)+10 <= x_list) & (x_list <= np.amax(target_x)-10))])
    if debug == True:
        print("    "*(tabs+1) + "x_list for fitting:" % x_list)
        
    # ==================================================
    # get x,y values using x_list and fit reference to match
    target_y_refsub = np.zeros_like(target_y_s)
    # for each input target spectrum
    for i in range(np.shape(target_y)[1]):
        # create arrays of average values for each point +/-5 pixels
        ref_x_averages, ref_y_averages = average_list(ref_x, ref_y, x_list, 5, debug=debug)
        target_x_averages, target_y_averages = average_list(target_x, target_y[:,i], x_list, 5, debug=debug)
    
        # create LMFIT parameters object
        params = lmfit.Parameters()
        params.add('scale_factor', value=np.amax(target_y_averages)/np.amax(ref_y_averages), min=0.)
        params.add('poly_order', value=poly_order, vary=False)
        for i2 in range(1, poly_order+1):
            params.add('p'+str(i2), value=0., vary=add_polynomial)
            
        if debug == True:
            print("    "*(tabs+1) + "input parameters:")
            params.pretty_print()
        
        # fit target values using reference values
        fit_output = lmfit.minimize(reference_fit, params, args=(ref_x_averages, ref_y_averages, target_x_averages, target_y_averages))
        
        if debug == True:
            print("    "*(tabs+1) + "fitted parameters:")
            print("    "*(tabs+2) + "spec %s: rescale factor = %0.2f" % (indices[i], fit_output.params['scale_factor'].value))
            fit_output.params.pretty_print()
        
        # interpolate rescaled reference to match target x values and subtract
        ref_y_rescaled = reference_curve(fit_output.params, ref_x, ref_y)
        ref_y_rescaled_interp = np.interp(target_x, ref_x, ref_y_rescaled)
        target_y_refsub[:,i] = target_y[:,i] - ref_y_rescaled_interp
    
    # ==================================================
    # save subtracted spectrum to measurement object
    target.add_spectrum(output_key, target_y_refsub, label="Intensity After %s Subtraction (counts)" % ref.name, log=old_log + [log])
    if debug == True:
        print("    "*(tabs+1) + "output %s array:" % output_key, np.shape(target_y_refsub))
        print("    "*(tabs+1) + "output log:", old_log + [log])
    
    # ==================================================
    # create figure if required
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)  # ax1: original target spec overlaid with rescaled reference spec
        ax2 = plt.subplot(212, sharex=ax1)  # ax2: target spec after subtraction
        ax1.set_title("%s\n%s Subtraction" % (target.title, ref.name))
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
        ax1.set_xlim(target.x_start, target.x_end)
        # plot original spectrum in ax1
        ax1.plot(target_x, target_y, 'k', label='input spectrum')
        # plot points used for fitting in ax1
        ax1.plot(target_x_averages, target_y_averages, 'ro')
        # plot rescaled reference spectrum in ax1
        ax1.plot(ref_x, ref_y_rescaled, label='rescaled ref')
        # plot refsub spectrum in ax2
        ax2.plot(target_x, target_y_refsub, label='after sub')
        # finish figure
        ax1.legend()
        ax2.legend()
        plt.minorticks_on()
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_%s.png" % (target.fig_dir, target.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(tabs+1) + "end of subtract_reference function")

"""
# ==================================================
# functions for finding peaks
# ==================================================
"""

def find_max(x, y, start, end):
    # function for finding the maximum in a slice of input data
    x_slice = x[np.where((start <= x) & (x <= end))]        # create slice
    y_slice = y[np.where((start <= x) & (x <= end))]
    i = np.argmax(y_slice)
    return np.array([x_slice[i], y_slice[i]]) # return x,y position of the maximum

def find_min(x, y, start, end):
    # function for finding the minimum in a slice of input data
    x_slice = x[np.where((start <= x) & (x <= end))]        # create slice
    y_slice = y[np.where((start <= x) & (x <= end))]
    i = np.argmin(y_slice)
    return np.array([x_slice[i], y_slice[i]]) # return x,y position of the minimum

def find_maxima(x, y, min_sep, threshold, tabs=0, debug=False):
    # function for finding the maxima of input data. Each maximum will have the largest value within ±/- min_sep.
    index_list = argrelextrema(y, np.greater, order=min_sep)    # determines indices of all maxima
    all_maxima = np.asarray([x[index_list], y[index_list]]) # creates an array of x and y values for all maxima
    y_limit = threshold * np.amax(y)                        # set the minimum threshold for defining a 'peak'
    x_maxima = all_maxima[0, all_maxima[1] >= y_limit]      # records the x values for all valid maxima
    y_maxima = all_maxima[1, all_maxima[1] >= y_limit]      # records the y values for all valid maxima
    maxima = np.asarray([x_maxima, y_maxima])               # creates an array for all valid maxima
    if debug == True:
        print("    "*(tabs) + "maxima found:")
        print("    "*(tabs+1) + "x: ", maxima[0])
        print("    "*(tabs+1) + "y: ", maxima[1])
    return maxima

def get_noise(x, y, start=2000, end=2100, tabs=0, debug=False):
    # function for estimating background noise level
    if end > np.amax(x):
        start, end = (1800, 1900)
    if end > np.amax(x):
        start, end = (600, 700)
    if debug == True:
        print("    "*(tabs) + "estimating noise")
        print("    "*(tabs+1) + "noise region: %0.1f - %0.1f" % (start, end))
    # noise is calculated as st. dev. of a 100 cm-1 wide peak-free region, defaults to 2400-2500
    noise_x, noise_y = slice_spectrum(x, y, start=start, end=end, tabs=tabs+1, debug=debug)
    noise = np.std(noise_y)
    noise = np.std(noise_y)
    if debug == True:
        print("    "*(tabs+1) + "noise level: %0.1f" % noise)
    return noise

def detect_peaks(measurement, x_key='raman_shift', y_key='y_av_sub', output_key='detected_peaks', SNR_threshold=10, norm_threshold=0.05, min_sep=20, x_start=None, x_end=None, noise=None, noise_region=(2000,2100), save_plot=False, show_plot=True, plot_name=None, save_to_file=True, log=None, debug=False, tabs=0):
    # script for finding peaks in a spectrum
    # argument summary:
    #   - measurement:      Measurement object containing all data
    #   - SNR_threshold:    minimum signal:noise ratio for valid peaks (int or float)
    #   - norm_threshold:   minimum intensity for valid peaks relative to spectrum maximum (float, 0-1)
    #   - min_sep:          minimum separation between valid peaks, in x axis units (int or float)
    #   - x_start:          start of x axis range to be processed (int or float)
    #   - x_end:            end of x axis range to be processed (int or float)
    #   - noise:            user-defined noise level (int or float), or None for automatic noise estimation
    #   - noise_region:     user-defined region to calculate noise from (tuple of 2 ints or 2 floats)
    #   - x_key:            key for x axis values in spec dict (string)
    #   - y_key:            key for y axis values in spec dict (string)
    #   - output_key:       key for outputting results into spec dict (string)
    #   - debug:            print debug messages? (boolean)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    print()
    print("    "*tabs + "running peak detection on measurement %s" % measurement.title)
    if isinstance(measurement, Measurement) == False:
        raise Exception("    "*(tabs+1) + "first argument passed to detect_peaks() function must be a Measurement instance!")
    elif hasattr(measurement, x_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % x_key)
    elif hasattr(measurement, y_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % y_key)
    if noise != None and type(noise) not in [int, float, np.int16, np.int32, np.float32, np.float64]:
        raise Exception("    "*(tabs+1) + "noise argument passed to find_peaks must be integer or float!")
    elif type(noise) in [int, float, np.int16, np.int32, np.float32, np.float64]:
        if noise <= 0:
            raise Exception("    "*(tabs+1) + "noise argument passed to find_peaks must be greater than 0!")
    if type(log) != str:
        log = 'automatic peak detection'
        
    # ==================================================
    # check if plotting is required
    plot = False
    if save_plot == True or show_plot == True:
        plot = True
    if plot == True and plot_name == None:
        plot_name = "peak-detection"
    
    # ==================================================
    # get data from measurement object
    x, y = measurement(x_key, y_key)
    indices = measurement[y_key].indices
    old_log = measurement[y_key].log
    if debug == True:
        print("    "*(tabs+1) + "input arrays:", x_key, np.shape(x), y_key, np.shape(y))
        print("    "*(tabs+1) + "previous y processing:", "\n".join(old_log))
        
    # ==================================================
    # smooth data for fitting
    if y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        y_s = smooth_spectrum(y, 5, 3)[:,np.newaxis]
    elif np.shape(y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
        if plot == True:
            print("    "*(tabs+1) + "cannot plot results of peak detection for multi-spec measurement!")
            plot = False
    if x_start == None:
        x_start = np.amin(x)
    if x_end == None:
        x_end = np.amax(x)
    
    # ==================================================
    # search for maxima in spectra
    peak_data = []
    peak_data = {'spec_index': [], 'centers': [], 'heights': []}
    for i in range(np.shape(y)[1]):
        # for each input target spectrum
        
        noise = get_noise(x, y_s[:,i], start=noise_region[0], end=noise_region[1], tabs=tabs+1, debug=debug)

        # slice the data for peak detection
        x_slice, y_slice = slice_spectrum(x, y_s[:,i], start=x_start, end=x_end, tabs=tabs+1)
        
        # find local maxima above norm_threshold, produces two lists: x positions and y intensity values
        y_min = np.amin(y_slice)
        y_max = np.amax(y_slice)
        maxima = find_maxima(x_slice, y_slice, min_sep, norm_threshold)
        if debug == True:
            print("    "*(tabs+1) + "spec %s:" % indices[i])
            print("    "*(tabs+2) + "%s maxima found:" % len(maxima[0]), ", ".join(["%0.1f" % peak for peak in maxima[0]]))

        # only pass maxima that are above SNR threshold
        maxima_pass = [[],[]]
        for i2 in range(0, len(maxima[0])):
            if maxima[1,i2] > float(SNR_threshold) * noise:
                maxima_pass[0].append(maxima[0,i2])
                maxima_pass[1].append(maxima[1,i2])
        maxima_pass = np.asarray(maxima_pass)   # same structure as maxima
        if debug == True:
            print("    "*(tabs+2) + "%s pass SNR threshold:" % len(maxima_pass[0]), ", ".join(["%0.1f" % peak for peak in maxima_pass[0]]))
            print("    "*(tabs+2) + "%s did not pass:" % (len(maxima[0])-len(maxima_pass[0])), ", ".join(["%0.1f" % peak for peak in maxima[0] if peak not in maxima_pass[0]]))
        
        # add data to temp storage array
        peak_data['spec_index'] += list(np.full_like(maxima_pass[0], indices[i]))
        peak_data['centers'] += list(maxima_pass[0])
        peak_data['heights'] += list(maxima_pass[1])
    
    peak_data = pd.DataFrame(peak_data)
    print("    "*(tabs+1) + "total %s potential peaks found across %s spectra" % (len(peak_data['spec_index']), len(np.unique(peak_data['spec_index']))))
    
    # ==================================================
    # save detected peaks to Measurement instance
    setattr(measurement, output_key, peak_data)
    if debug == True:
        print("    "*(tabs+1) + "output %s array:" % output_key, np.shape(peak_data))
        print("    "*(tabs+1) + "output log:", old_log + [log])
        
    # save detected peaks to file
    if len(peak_data['centers']) > 0 and save_to_file == True:
        print()
        print("    "*(tabs) + "saving peak fit data to file")
        # save to file
        outdir = "%s%s_detected-peaks.csv" % (measurement.out_dir, measurement.title)
        if debug == True:
            print("    "*(tabs+1) + "detected peak properties dataframe:")
            print(peak_data.info())
            print("    "*(tabs+1) + "saving to %s" % outdir)
        # save data to output folder
        peak_data.to_csv(outdir)
    
    # ==================================================
    # create figure if required
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.set_title("%s\nAutomatic Peak Detection" % (measurement.title))
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Intensity (counts)")
        ax1.set_xlim(x_start, x_end)
        ax1.set_ylim(-0.2*y_max, 1.2*y_max)
        # add horizontal line for relative intensity threshold (blue)
        ax1.axhline(norm_threshold*y_max, color='b', linestyle=':', alpha=0.5, label='min. int.')
        # add horizontal line for SNR threshold (red)
        ax1.axhline(float(SNR_threshold)*noise, color='r', linestyle=':', alpha=0.5, label='min. SNR')
        # plot spectrum
        plt.plot(x_slice, y_slice, 'k', label='data')
        # plot maxima that fail
        ax1.plot(maxima[0], maxima[1], 'ro', label='fail (%d)' % (len(maxima[0])-len(maxima_pass[0])))
        # plot maxima that pass
        ax1.plot(maxima_pass[0], maxima_pass[1], 'bo', label='pass (%d)' % (len(maxima_pass[0])))
        if debug == True:
            print("    "*(tabs+1) + "detected peaks:")
        for i2 in range(len(maxima_pass[0])):
            # report detected positions
            ax1.text(maxima_pass[0,i2], maxima_pass[1,i2]+0.05*y_max, "%0.f" % maxima_pass[0,i2], rotation=90, va='bottom', ha='left')
            if debug == True:
                print("    "*(tabs+2) + "%0.f cm: %0.2f" % (maxima_pass[0,i2], maxima_pass[1,i2]/y_max))
        
        # create second y axis for SNR values
        ax2 = ax1.twinx()
        ax2.set_ylim(intensity2snr(ax1.get_ylim(), noise))
        ax2.set_ylabel("SNR")
        ax1.legend()
        plt.minorticks_on()
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_%s.png" % (measurement.fig_dir, measurement.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(tabs) + "end of detect_peaks() function")

"""
# ==================================================
# functions for fitting peaks
# ==================================================
"""

def multiG_curve(params, x, maxima):
    # function for generating a model gaussian curve using defined parameters
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiG_fit(params, x, y, maxima):
    # function for use with LMFIT minimize(), provides difference between input data and gaussian model
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiL_curve(params, x, maxima):
    # function for generating a model lorentzian curve using defined parameters
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        gamma = params['gamma_%s' % i]
        model += A * (gamma**2)/((x - mu)**2 + gamma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiL_fit(params, x, y, maxima):
    # function for use with LMFIT minimize(), provides difference between input data and lorentzian model
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        gamma = params['gamma_%s' % i]
        model += A * (gamma**2)/((x - mu)**2 + gamma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiPV_curve(params, x, maxima):
    # function for generating a model pseudo-voigt curve using defined parameters
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        gamma = np.sqrt(2.*np.log(2.)) * sigma
        eta = params['eta_%s' % i]
        model += A * (eta * (gamma**2)/((x - mu)**2 + gamma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiPV_fit(params, x, y, maxima):
    # function for use with LMFIT minimize(), provides difference between input data and pseudo-voigt model
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        gamma = np.sqrt(2.*np.log(2.)) * sigma
        eta = params['eta_%s' % i]
        model += A * (eta * (gamma**2)/((x - mu)**2 + gamma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def peak_fit_script(x, y, maxima, max_shift=10., min_fwhm=5., max_fwhm=150., function='g', vary_baseline=True, debug=False):
    # script for fitting a set of maxima with pre-defined peak functions
    if function.lower() in ['pv', 'pseudo-voigt', 'pseudovoigt', 'psuedo-voigt', 'psuedovoigt']:
        function = 'pv'
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac', 'fermidirac']:
        function = 'fd'
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        function = 'l'
    elif function.lower() in ['g', 'gauss', 'gaussian']:
        function = 'g'
    else:
        if debug == True:
            print("    specified peak function is not recognised, defaults to gaussian")
    # create LMFIT parameters object
    params = lmfit.Parameters()
    # add linear baseline parameters
    params.add('gradient', value=0., vary=vary_baseline)
    params.add('intercept', value=np.amin(y), vary=vary_baseline)
    for i in range(0, len(maxima)):
        # for each input peak, add necessary peak function parameters
        y_max = x[np.argmin(np.absolute(y - maxima[i]))]
        params.add('center_%s' % i, value=maxima[i], min=maxima[i]-max_shift, max=maxima[i]+max_shift)
        params.add('amplitude_%s' % i, value=y_max, min=0.)
        if function == 'pv':
            # pseudo-voigt functions use sigma parameter for width, eta for lorentzian factor
            params.add('sigma_%s' % i, value=10., min=min_fwhm/(2.*np.sqrt(2.*np.log(2))), max=max_fwhm/(2.*np.sqrt(2.*np.log(2))))
            params.add('eta_%s' % i, value=0.5, min=0., max=1.)
        elif function == 'fd':
            # symmetric fermi-dirac functions use width parameter for half-width, rounding for overall roundness
            params.add('width_%s' % i, value=10., min=min_fwhm/2., max=max_fwhm/2.)
            params.add('rounding_%s' % i, value=5., min=0.)
        elif function == 'l':
            # lorentzian functions use gamma for width
            params.add('gamma_%s' % i, value=10., min=2., max=max_fwhm/2.)
        else:
            # gaussian functions use sigma for width
            params.add('sigma_%s' % i, value=10., min=min_fwhm/(2.*np.sqrt(2.*np.log(2))), max=max_fwhm/(2.*np.sqrt(2.*np.log(2))))
    if debug == True:
        print("        initial parameters:")
        print(params.pretty_print())
    # run fit and generate fitted curve
    if function == 'pv':
        fit_output = lmfit.minimize(multiPV_fit, params, args=(x, y, maxima))
        fit_curve = multiPV_curve(x, fit_output.params, maxima)
    elif function == 'fd':
        fit_output = lmfit.minimize(multiFD_fit, params, args=(x, y, maxima))
        fit_curve = multiFD_curve(x, fit_output.params, maxima)
    elif function == 'l':
        fit_output = lmfit.minimize(multiL_fit, params, args=(x, y, maxima))
        fit_curve = multiL_curve(x, fit_output.params, maxima)
    else:
        fit_output = lmfit.minimize(multiG_fit, params, args=(x, y, maxima))
        fit_curve = multiG_curve(x, fit_output.params, maxima)
    if debug == True:
        print("        fit status: ", fit_output.message)
        print("        fitted parameters:")
        print(fit_output.params.pretty_print())
    return fit_output, fit_curve

def fwhm(params, i=None, function='g'):
    # returns the FWHM for any function using the appropriate calculation
    if function.lower() in ['v', 'voigt']:
        # voigt width is approximated by combination of lorentzian gamma and gaussian sigma parameters
        if i == None:
            fL = 2. * params['gamma']
            fG = np.sqrt(2. * np.log(2)) * params['sigma']
            return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
        else:
            fL = 2. * params['gamma_%s' % i]
            fG = np.sqrt(2. * np.log(2)) * params['sigma_%s' % i]
            return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
    elif function.lower() in ['pv', 'pseudo-voigt', 'pseudovoigt', 'psuedo-voigt', 'psuedovoigt']:
        # pseudo-voigt width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma']
        else:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i]
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac', 'fermidirac']:
        # fermi-dirac width is twice the width parameter
        if i == None:
            return 2. * params['width']
        else:
            return 2. * params['width_%s' % i]
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        # lorentzian width is twice the gamma parameter
        if i == None:
            return 2. * params['gamma']
        else:
            return 2. * params['gamma_%s' % i]
    else:
        # gaussian width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma']
        else:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i]
        
def fwhm_err(params, i=None, function='g'):
    # returns the FWHM standard error for any function using the appropriate calculation
    if function.lower() in ['v', 'voigt']:
        # voigt width is approximated by combination of lorentzian gamma and gaussian sigma parameters
        if i == None:
            if params['gamma'].stderr != None and params['sigma'].stderr != None:
                fL = 2. * params['gamma'].stderr
                fG = np.sqrt(2. * np.log(2)) * params['sigma'].stderr
                return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            else:
                return 0.
        else:
            if params['gamma_%s' % i].stderr != None and params['sigma_%s' % i].stderr != None:
                fL = 2. * params['gamma_%s' % i].stderr
                fG = np.sqrt(2. * np.log(2)) * params['sigma_%s' % i].stderr
                return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            else:
                return 0.
    elif function.lower() in ['pv', 'pseudo-voigt', 'pseudovoigt', 'psuedo-voigt', 'psuedovoigt']:
        # pseudo-voigt width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            if params['sigma'].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma'].stderr
            else:
                return 0.
        else:
            if params['sigma_%s' % i].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i].stderr
            else:
                return 0.
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac', 'fermidirac']:
        # fermi-dirac width is twice the width parameter
        if i == None:
            if params['width'].stderr != None:
                return 2. * params['width'].stderr
            else:
                return 0.
        else:
            if params['width_%s' % i].stderr != None:
                return 2. * params['width_%s' % i].stderr
            else:
                return 0.
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        # lorentzian width is twice the gamma parameter
        if i == None:
            if params['gamma'].stderr != None:
                return 2. * params['gamma'].stderr
            else:
                return 0.
        else:
            if params['gamma_%s' % i].stderr != None:
                return 2. * params['gamma_%s' % i].stderr
            else:
                return 0.
    else:
        # gaussian width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            if params['sigma'].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma'].stderr
            else:
                return 0.
        else:
            if params['sigma_%s' % i].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i].stderr
            else:
                return 0.
            
def fitting_regions(peak_list, start=400, end=1800, window=150., rounding=10, tabs=0, debug=False):
    # function for dividing a list of peak positions into separate regions for fitting
    regions = []
    peak_list = np.sort(peak_list)
    if len(peak_list) == 0:
        raise Exception("    "*(tabs) + "no peaks passed to fitting_regions() function!")
    else:
        peak_list_trim = peak_list[np.where((start <= peak_list) & (peak_list <= end))]
    if len(peak_list_trim) == 1:
        # single region centred around single peak, rounded to nearest 10.
        regions = [[rounding*np.floor((peak_list_trim[0] - window)/rounding), rounding*np.ceil((peak_list_trim[0] + window)/rounding)]]
    else:
        # at least two peaks present, split into 1+ regions
        temp = [rounding*np.floor((peak_list_trim[0] - window)/rounding), rounding*np.floor((peak_list_trim[0] + window)/rounding)]
        for peak in peak_list_trim:
            # check each peak to see if it falls within window of current region
            if rounding*np.floor((peak)/rounding) > temp[1]:
                # peak falls outside current region, save current region to regions list
                regions.append(temp)
                # create new region centered on peak
                temp = [rounding*np.floor((peak - window)/rounding), rounding*np.ceil((peak + window)/rounding)]
            else:
                # peak falls inside current region, extend region to include it
                temp[-1] = rounding*np.ceil((peak + window)/rounding)
        regions.append(temp)
    if debug == True:
        print("    "*(tabs+1), regions)
    return np.asarray(regions)

def fit_peaks(measurement, x_key='raman_shift', y_key='y_av_sub', output_key='fitted_peaks', function='pv', peak_positions=[], max_shift=20, max_fwhm=150, min_fwhm=5, vary_baseline=True, region_window=150., x_start=None, x_end=None, SNR_threshold=10, norm_threshold=0.05, min_sep=25, noise=None, noise_region=(2400,2500), save_plot=False, show_plot=True, plot_name=None, save_to_file=True, log=None, tabs=0, debug=False):
    # script for finding peaks in a spectrum
    # argument summary:
    #   - measurement:      Measurement object containing all data
    #   - function:         function to use for fitting peaks ('g', 'l', 'pv', or 'fd')
    #   - peak_positions:   list of peak positions to use, leave empty for automatic (list of ints or floats)
    #   - max_shift:        maximum change in position allowed during fitting (int or float)
    #   - max_fwhm:         maximum full-width-half-maximum (peak width) allowed during fitting (int or float)
    #   - vary_baseline:    allow fit to change baseline parameters as part of fit? (boolean)
    #   - region_window:    max separation between peaks for including in same fit (int or float)
    #   - x_start:          start of x axis range to be processed (int or float)
    #   - x_end:            end of x axis range to be processed (int or float)
    #   - SNR_threshold:    minimum signal:noise ratio for automatic peak detection (int or float)
    #   - norm_threshold:   minimum intensity for automatic peak detection (float, 0-1)
    #   - noise:            user-defined noise level (int or float), or None for automatic noise estimation
    #   - noise_region:     user-defined region to calculate noise in (tuple of 2 ints or 2 floats)
    #   - min_sep:          minimum separation between valid peaks, in x axis units (int or float)
    #   - x_key:            key for x axis values in spec dict (string)
    #   - y_key:            key for y axis values in spec dict (string)
    #   - output_key:       key for outputting results into spec dict (string)
    #   - debug:            print debug messages? (boolean)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    print()
    print("    "*(tabs) + "running peak fitting on measurement %s" % measurement.title)
    if isinstance(measurement, Measurement) == False:
        raise Exception("    "*(tabs+1) + "first argument passed to fit_peaks() function must be a Measurement instance!")
    elif hasattr(measurement, x_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % x_key)
    elif hasattr(measurement, y_key) == False:
        raise Exception("    "*(tabs+1) + "%s not in measurement data!" % y_key)
    if noise != None and type(noise) not in [int, float, np.int16, np.int32, np.float32, np.float64]:
        raise Exception("    "*(tabs+1) + "noise argument passed to find_peaks must be integer or float!")
    elif type(noise) in [int, float, np.int16, np.int32, np.float32, np.float64]:
        if noise <= 0:
            raise Exception("    "*(tabs+1) + "noise argument passed to find_peaks must be greater than 0!")
    if type(log) != str:
        log = 'automatic peak fitting'
        
    # ==================================================
    # check if plotting is required
    plot = False
    if save_plot == True or show_plot == True:
        plot = True
    if plot == True and plot_name == None:
        plot_name = "peak-fitting"
    
    # ==================================================
    # get data from measurement object
    x, y = measurement(x_key, y_key)
    old_log = measurement[y_key].log
    indices = measurement[y_key].indices
    if debug == True:
        print("    "*(tabs+1) + "input arrays:", x_key, np.shape(x), y_key, np.shape(y))
        print("    "*(tabs+1) + "previous y processing:", "\n".join(old_log))
    # smooth data for fitting
    if y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        y_s = y[:,np.newaxis]
    elif np.shape(y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        y_s = y
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        y_s = y
        if plot == True:
            print("    "*(tabs+1) + "cannot plot results of peak fitting for multi-spec measurement!")
            plot = False
    if x_start == None:
        x_start = np.amin(x)
    if x_end == None:
        x_end = np.amax(x)
        
    # ==================================================
    # determine which function to use
    if function.lower() in ['pv', 'pseudo-voigt', 'pseudo voigt']:
        function = 'pv'
        function_title = 'Pseudo-Voigt'
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
        function = 'fd'
        function_title = 'Fermi-Dirac'
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        function = 'l'
        function_title = 'Lorentzian'
    else:
        function = 'g'
        function_title = 'Gaussian'
    if debug == True:
        print("    "*(tabs+1) + "fitting function:", function_title)
    
    # ==================================================
    # create arrays to store results
    input_peaks = [[] for i in range(np.shape(y)[1])]
    peak_data = {'spec_index': [], 'function': [], 'centers': [], 'centers_err': [], 'amplitudes': [], 'amplitudes_err': [], 'fwhm': [], 'fwhm_err': [], 'heights': [],}
    # add more arrays depending on chosen function
    if function == 'pv':
        peak_data['sigmas'] = []
        peak_data['sigmas_err'] = []
        peak_data['etas'] = []
        peak_data['etas_err'] = []
    elif function == 'fd':
        peak_data['rounds'] = []
        peak_data['rounds_err'] = []
        peak_data['widths'] = []
        peak_data['widths_err'] = []
    elif function == 'l':
        peak_data['gammas'] = []
        peak_data['gammas_err'] = []
    else:
        peak_data['sigmas'] = []
        peak_data['sigmas_err'] = []
        
    # ==================================================
    # determine which input peak positions to use for fitting
    if len(peak_positions) > 0:
        # use manually specified peaks
        if debug == True:
            print("    "*(tabs+1) + "using manually specified peak positions:")
            print("    "*(tabs+2) + "all spectra:", ", ".join(["%0.f" % peak for peak in peak_positions]))
        # pass list of peaks that are inside range x_start - x_end
        temp = np.asarray([peak for peak in peak_positions if x_start <= peak and peak <= x_end])
        for i in range(np.shape(y)[1]):
            # ensures each input spectrum gets its own copy of the list
            input_peaks[i] = temp
    else:
        # use automatic peak detection to get peaks
        if hasattr(measurement, 'detected_peaks'):
            # use previously detected peaks
            if debug == True:
                print("    "*(tabs+1) + "no peaks specified, using previously detected peak positions:")
            for i in range(np.shape(y)[1]):
                # ensures each input spectrum gets its own list from detected_peaks
                sort = measurement['detected_peaks']['spec_index'].values == indices[i]
                temp = np.asarray(measurement['detected_peaks'].loc[sort,'centers'])
                print(temp)
                print(np.shape(temp))
                if debug == True:
                    print("    "*(tabs+2) + "spec %s:" % i, ", ".join(["%0.f" % peak for peak in temp]))
                input_peaks[i] = temp
        else:
            # no peak detection recorded, do it now
            if debug == True:
                print("    "*(tabs+1) + "no peaks specified, doing automatic peak detection")
            # run detect_peaks() using input variables
            detect_peaks(measurement, x_key, y_key, 'detected_peaks', SNR_threshold=SNR_threshold, norm_threshold=norm_threshold, min_sep=min_sep, noise=noise, noise_region=noise_region, x_start=x_start, x_end=x_end, save_plot=False, show_plot=show_plot, tabs=tabs+1, debug=debug)
            for i in range(np.shape(y)[1]):
                # ensures each input spectrum gets its own list from detected_peaks
                sort = measurement['detected_peaks']['spec_index'].values == indices[i]
                temp = np.asarray(measurement['detected_peaks'].loc[sort,'centers'])
                if debug == True:
                    print("    "*(tabs+2) + "spec %s:" % i, ", ".join(["%0.f" % peak for peak in temp]))
                input_peaks[i] = temp
    
    # ==================================================
    # estimate noise (for calculating SNR)
    noise = get_noise(x, y, start=noise_region[0], end=noise_region[1], tabs=tabs+1, debug=debug)
    
    # ==================================================
    # proceed with fitting
    for i in range(np.shape(y)[1]):
        # for each spectrum
        peaks = input_peaks[i]
        if debug == True:
            print()
            print("    "*(tabs+1) + "fitting spectrum %s..." % i)
        if len(peaks) == 0:
            print("    "*(tabs+2) + "no input peaks!")
        else:
            # at least one peak found, proceed
            
            # calculate noise for this spectrum
            noise = get_noise(x, y_s[:,i], start=noise_region[0], end=noise_region[1], tabs=tabs+2, debug=debug)
            if debug == True:
                print("    "*(tabs+2) + "noise level: %0.1f" % noise)

            # split into regions
            regions = fitting_regions(input_peaks[i], start=x_start, end=x_end, window=region_window, tabs=tabs+2)
            if debug == True:
                print("    "*(tabs+2) + "spec %s divided into %s regions:" % (i, len(regions)), ", ".join(["(%0.1f-%0.1f)" % (reg[0], reg[1]) for reg in regions]))
                
            for region in regions:
                # ==================================================
                # for each region in spectrum
                start, end = region
                if debug == True:
                    print()
                    print("    "*(tabs+2) + "fitting region %0.f - %0.f cm-1" % (start, end))

                # pick out only those peaks in this region
                reg_peaks = peaks[np.where((start < peaks) & (peaks < end))]
                if debug == True:
                    print("    "*(tabs+3) + "%s peaks in region:" % len(reg_peaks), reg_peaks)

                # slice spectrum to region
                x_slice, y_slice = slice_spectrum(x, y_s[:,i], start=start, end=end, tabs=tabs+3)

                # proceed with peak fit
                fit_output, fit_curve = peak_fit_script(x_slice, y_slice, reg_peaks, function=function, max_shift=max_shift, min_fwhm=min_fwhm, max_fwhm=max_fwhm)
                
                if debug == True and fit_output.params['center_0'].stderr == None:
                    print("    "*(tabs+3) + "could not estimate standard errors")

                # ==================================================
                # convert results to numpy arrays
                for i2, peak in enumerate(reg_peaks):
                    # add parameters to storage array
                    peak_data['spec_index'].append(indices[i])
                    peak_data['function'].append(function)
                    peak_data['heights'].append(fit_output.params['amplitude_%s' % i2].value + fit_output.params['gradient'].value * fit_output.params['center_%s' % i2].value + fit_output.params['intercept'].value)
                    for prop in ['center', 'amplitude', 'sigma', 'gamma', 'eta', 'width', 'round']:
                        key = prop+"_%s" % i2
                        if key in fit_output.params.keys():
                            peak_data["%ss" % prop].append(fit_output.params[key].value)
                            if fit_output.params[key].stderr != None:
                                peak_data[prop+"s_err"].append(fit_output.params[key].stderr)
                            else:
                                peak_data[prop+"s_err"].append(0.)
                    # get FWHM and error
                    peak_data['fwhm'].append(fwhm(fit_output.params, i2, function))
                    peak_data['fwhm_err'].append(fwhm_err(fit_output.params, i2, function))

                # ==================================================
                # create region fit figures if required
                if plot == True and i == 0:
                    # for each region in spectrum 0
                    # prepare figure
                    plt.figure(figsize=(8,6))
                    # ax1: results of fit
                    ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
                    ax1.set_title("%s\n%0.f-%0.f cm$^{-1}$ %s Peak Fitting" % (measurement.title, start, end, function_title))
                    ax1.set_ylabel("Average Intensity")
                    # ax2: residuals
                    ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
                    ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
                    ax2.set_ylabel("Residual")
                    # histogram of residuals
                    ax3 = plt.subplot2grid((4,5), (3,4))
                    ax3.set_yticks([])
                    # determine y limits for residual, hist plots
                    y_min = np.amin(y_slice-fit_curve)
                    y_max = np.amax(y_slice-fit_curve)
                    res_min = y_min - 0.1*(y_max-y_min)
                    res_max = y_max + 0.1*(y_max-y_min)
                    ax2.set_ylim(res_min, res_max)
                    # plot input data and residuals
                    ax1.plot(x_slice, y_slice, 'k')
                    ax2.plot(x_slice, y_slice-fit_curve, 'k')
                    ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
                    x_temp = np.linspace(start, end, 10*len(x_slice))
                    # plot total peak fit
                    if function == 'pv':
                        total_curve = multiPV_curve(x_temp, fit_output.params, reg_peaks)
                    elif function == 'fd':
                        total_curve = multiFD_curve(x_temp, fit_output.params, reg_peaks)
                    elif function == 'l':
                        total_curve = multiL_curve(x_temp, fit_output.params, reg_peaks)
                    else:
                        total_curve = multiG_curve(x_temp, fit_output.params, reg_peaks)
                    ax1.plot(x_temp, total_curve, 'b--')
                    # determine y axis limits
                    y_max = np.amax(y_slice)
                    y_min = np.amin([-0.2*y_max, np.amin(y_slice), np.amin(total_curve)])
                    ax1.set_xlim(start, end)
                    ax1.set_ylim(y_min, 1.2*y_max)
                    # plot individual peak fits
                    for i2, peak in enumerate(reg_peaks):
                        # plot and report peak positions
                        ax1.text(fit_output.params["center_%s" % i2]+0.01*(end-start), y_min+0.95*(1.2*y_max-y_min), i2+1)
                        plt.figtext(0.82, 0.93-0.08*i2, "Center %s: %.1f" % (i2+1, fit_output.params["center_%s" % i2]))
                        plt.figtext(0.82, 0.9-0.08*i2, "  FWHM %s: %.1f" % (i2+1, fwhm(fit_output.params, i2, function)))
                        ax1.axvline(fit_output.params["center_%s" % i2], color='k', linestyle=':')
                        # create function curve for plotting this specific peak
                        params = lmfit.Parameters()
                        params.add('gradient', value=fit_output.params["gradient"])
                        params.add('intercept', value=fit_output.params["intercept"])
                        params.add('amplitude_0', value=fit_output.params["amplitude_%s" % i2])
                        params.add('center_0', value=fit_output.params["center_%s" % i2])
                        if function == 'pv':
                            params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                            params.add('eta_0', value=fit_output.params["eta_%s" % i2])
                            peak_curve = multiPV_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        elif function == 'fd':
                            params.add('width_0', value=fit_output.params["width_%s" % i2])
                            params.add('round_0', value=fit_output.params["round_%s" % i2])
                            peak_curve = multiFD_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        elif function == 'l':
                            params.add('gamma_0', value=fit_output.params["gamma_%s" % i2])
                            peak_curve = multiL_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        else:
                            params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                            peak_curve = multiG_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        ax1.plot(x_temp, peak_curve, 'b:')
                    # finish figure
                    ax1.minorticks_on()
                    ax2.minorticks_on()
                    ax3.minorticks_on()
                    plt.tight_layout()
                    # save to file
                    figdir = "%s%s_%0.f-%0.fcm_%s-fit.png" % (measurement.fig_dir, measurement.title, start, end, function.upper())
                    if debug == True:
                        print("    "*(tabs+3) + "saving figure to %s" % figdir)
                    if save_plot == True:
                        plt.savefig(figdir, dpi=300)
                    if show_plot == True:
                        plt.show()
                    else:
                        plt.close()
                
    # ==================================================
    # convert spectrum results to pandas dataframe
    peak_data = pd.DataFrame(peak_data)
    
    print("    "*(tabs) + "total %s peaks fitted across %s spectra" % (len(peak_data['spec_index']), len(np.unique(peak_data['spec_index']))))
    
    # ==================================================
    # save fitted peaks to Measurement instance
    setattr(measurement, output_key, peak_data)
    if debug == True:
        print("    "*(tabs) + "output %s array:" % output_key)
        print(peak_data.info())
        print("    "*(tabs+1) + "output log:", old_log + [log])
        
    # save fitted peaks to file
    if len(peak_data['centers']) > 0 and save_to_file == True:
        print("    "*(tabs) + "saving peak fit data to file")
        # save to file
        outdir = "%s%s_fitted-peaks.csv" % (measurement.out_dir, measurement.title)
        if debug == True:
            print("    "*(tabs+1) + "saving to %s" % outdir)
        # save data to output folder
        peak_data.to_csv(outdir)
    
    # ==================================================
    # create summary figure if required
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.set_title("%s\nAutomatic Peak Fitting" % (measurement.title))
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Intensity (counts)")
        # plot spectrum
        x, y = measurement(x_key, y_key)
        y = measurement[y_key].mean()
        y_max = np.amax(y)
        y_min = np.amin(y)
        plt.plot(x, y, 'k', label=y_key)
        # plot fitted peak positions and intensities
        x = peak_data['centers']
        y = peak_data['heights']
        ax1.plot(x, y, 'bo', label='fitted peaks')
        if debug == True:
            print("    "*(tabs+1) + "fitted peaks:")
        for i2 in range(len(x)):
            # report detected positions
            ax1.text(x[i2], y[i2]+0.05*(y_max-y_min), "%0.f" % x[i2], rotation=90, va='bottom', ha='left')
            if debug == True:
                print("    "*(tabs+2) + "%0.f: %0.2f" % (x[i2], y[i2]/y_max))
        # get axis limits
        ax1.set_xlim(x_start, x_end)
        ax1.set_ylim(-0.2*y_max, 1.2*y_max)
        if noise > 0:
            # create second y axis for SNR values
            ax2 = ax1.twinx()
            ax2.set_ylim(intensity2snr(ax1.get_ylim(), noise))
            ax2.set_ylabel("SNR")
        # finish figure
        ax1.legend()
        plt.minorticks_on()
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_%s.png" % (measurement.fig_dir, measurement.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
    
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(tabs) + "end of fit_peaks() function")
        
"""
# ==================================================
# functions for handling maps
# ==================================================
"""



"""
# ==================================================
# functions for doing statistical analysis
# ==================================================
"""











"""
# ==================================================
# functions for doing dimensional reduction
# ==================================================
"""

def do_pca(measurements, x_key='raman_shift', y_key='y_av', pca_components=6, x_list=[], start=None, end=None, standardise=False, normalise=False, first_deriv=False, tabs=0, save_plot=False, show_plot=False, plot_components=3, colour_dict={}, debug=False):
    # function for doing Principal Component Analysis of several spectra to identify major variances
    # argument summary:
    #   - measurements:     list of Measurement instances to average
    #   - x_key:            key for x axis values to get from measurement (string)
    #   - y_key:            key for y axis values to get from measurement (string)
    #   - normalise:        normalise spectra? (boolean)
    #   - x_list:           optional list of x positions for interpolating data (list of ints/floats)
    #   - start:            trim x range to start here
    #   - end:              trim x range to end here
    
    # ==================================================
    # interpolate data
    x, y = interpolate_spectra(measurements, x_key=x_key, y_key=y_key, x_list=x_list, debug=True)

    # ==================================================
    # transform data (if required)
    model_data = np.copy(y)
    modifications = []
    if standardise == True:
        # standardisation: subtract mean and divide by standard deviation
        model_data = (y - np.mean(y, axis=1)[:, np.newaxis]) / np.std(y, axis=1)[:, np.newaxis]
        modifications.append('standardised')
    if normalise == True:
        # normalisation: reduce spectrum to range 0-1 (min-max)
        model_data = (y - np.amin(y, axis=1)[:, np.newaxis]) / (np.amax(y, axis=1)[:, np.newaxis] - np.amin(y, axis=1)[:, np.newaxis])
        modifications.append('normalised')
    if first_deriv == True:
        # first derivative: smooth spectrum and get change in y
        for i in range(np.size(y, axis=0)):
            model_data[i] = savgol_filter(y[i], 25, polyorder = 5, deriv=1)
        modifications.append('first-derivative')
    if debug == True:
        print("    "*(tabs), "transformed data array:", np.shape(model_data))
        print("    "*(tabs+1), "nan check:", np.any(np.isnan(model_data)))
            
    # ==================================================
    # get Principal Component coordinates
    if pca_components >= len(labels):
        # not enough spectra, reduce number of components to M-1
        if debug == True:
            print("    "*(tabs), "N=%s PCA components >= M=%s spectra. Reducing N to %s" % (pca_components, len(labels), len(labels)-1))
        pca_components = len(labels)-1
            
    pca_model = PCA(n_components=pca_components).fit(model_data)
    pca_coords = pca_model.transform(model_data)
    if debug == True:
        print("    "*(tabs), "PCA coordinate array:", np.shape(pca_coords))
        for i, (comp, var) in enumerate(zip(pca_model.components_, pca_model.explained_variance_)):
            print("    component %d: %0.3f (cumulative: %0.3f)" % (i+1, pca_model.explained_variance_ratio_[i], np.sum(pca_model.explained_variance_ratio_[:i+1])))
            
    if plot == True:
        plot_PCA_summary(x, model_data, pca_model, model_labels, modifications, plot_components=plot_components, colour_dict=colour_dict)
        
    return pca_model, pca_coords

def plot_PCA_summary(x, y, pca_model, labels, modifications=[], plot_components=3, colour_dict={}, save_plot=True, show_plot=True):
    # script for generating a PCA summary plot
    
    plot = False
    if save_plot == True or show_plot == True:
        plot = True
    
    labels = np.asarray(labels)
    
    # generate PCA coordinates for plotting in ax2
    pca_coords = pca_model.transform(y)
    n_components = len(pca_model.components_)
    
    # check plot_components does not exceed actual components
    if plot_components > n_components:
        plot_components = n_components
    
    # generate plot colour dict if needed
    if len(colour_dict.keys()) == 0:
        colour_dict = {label: Colour_List[i % len(Colour_List)] for i, label in enumerate(np.unique(labels))}
           
    # generate PCA plot figure
    plt.figure(figsize=(15,5))
    # ax1: all input spectra
    ax1 = plt.subplot2grid((plot_components,3), (0,0), rowspan=plot_components)
    ax1.set_title("Input Spectra")
    ax1.set_xlabel("Frequency (cm$^{-1}$)")
    ax1.set_ylabel("Intensity")
    ax1.set_xlim(np.amin(x), np.amax(x))
    spec_count = np.size(y, axis=0)
    for i in range(spec_count):
        c = colour_dict[labels[i]]
        ax1.plot(x, y[i], c, alpha=1./np.sqrt(spec_count))
    # ax2: plot PCA coordinates by label
    ax2 = plt.subplot2grid((plot_components,3), (0,1), rowspan=plot_components)
    ax2.set_title('PCA of %0.f-%0.f cm$^{-1}$' % (np.amin(x), np.amax(x)))
    ax2.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca_model.explained_variance_ratio_[0]))
    ax2.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca_model.explained_variance_ratio_[1]))
    for i, label in enumerate(np.unique(labels)):
        mec = colour_dict[label]
        mfc = colour_dict[label]
        result = labels == label
        ax2.scatter(pca_coords[result,0], pca_coords[result,1], facecolors=mfc, edgecolors=mec, alpha=1., label="%s (%d)" % (label, np.count_nonzero(result)))
    
    # remaining axes: plot loading spectra for each component up to N=plot_components
    for i in range(plot_components):
        ax = plt.subplot2grid((plot_components,3), (i,2))
        if i == 0:
            ax.set_title("PCA Loading Spectra")
        if i < plot_components-1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("PC%d Coefficient" % (i+1))
        ax.set_xlim(np.amin(x), np.amax(x))
        comp = pca_model.components_[i] * pca_model.explained_variance_[i]  # scale component by its variance explanation power
        if 'first-derivative' in modifications:
            ax.plot(x, np.cumsum(comp), label="PC %s" % (i+1))
        else:
            ax.plot(x, comp, label="PC %s" % (i+1))
    plt.tight_layout()
    ax2.legend()
    ax2.grid(zorder=3)
    if save_plot == True:
        plt.savefig("%sPCA/PCA.png" % (Fig_dir), dpi=300)
    if show_plot == True:
        plt.show()
    else:
        plt.close()

"""
# ==================================================
# functions for spectral classification by supervised machine learning models
# ==================================================
"""

def prepare_data_for_classification(x, y, labels, standardise=False, normalise=False, first_deriv=False, pca_components=6, tabs=0, plot=False, plot_components=3, colour_dict={}, debug=False):
    # script for preparing data for input into a training/testing model
    #   - data:             spectral y values with matching x values (2D numpy array, use interpolate_spectra function)
    #   - labels:           list of corresponding labels for classifying each spectrum (list or array of strings)
    #   - standardise:      standardise spectra? (boolean)
    #   - normalise:        normalise spectra? (boolean)
    #   - first_deriv:      take first derivative of spectra? (boolean)
    #   - pca_components:   number of principal components to calculate (int)
    
    # ==================================================
    # transform data (if required)
    model_data = np.copy(y)
    modifications = []
    if standardise == True:
        # standardisation: subtract mean and divide by standard deviation
        model_data = (y - np.mean(y, axis=1)[:, np.newaxis]) / np.std(y, axis=1)[:, np.newaxis]
        modifications.append('standardised')
    if normalise == True:
        # normalisation: reduce spectrum to range 0-1 (min-max)
        model_data = (y - np.amin(y, axis=1)[:, np.newaxis]) / (np.amax(y, axis=1)[:, np.newaxis] - np.amin(y, axis=1)[:, np.newaxis])
        modifications.append('normalised')
    if first_deriv == True:
        # first derivative: smooth spectrum and get change in y
        for i in range(np.size(y, axis=0)):
            model_data[i] = savgol_filter(y[i], 25, polyorder = 5, deriv=1)
        modifications.append('first-derivative')
    if debug == True:
        print("    "*(tabs), "transformed data array:", np.shape(model_data))
        print("    "*(tabs+1), "nan check:", np.any(np.isnan(model_data)))
            
    # ==================================================
    # get Principal Component coordinates
    if pca_components >= len(labels):
        # not enough spectra, reduce number of components to M-1
        if debug == True:
            print("    "*(tabs), "N=%s PCA components >= M=%s spectra. Reducing N to %s" % (pca_components, len(labels), len(labels)-1))
        pca_components = len(labels)-1
            
    pca_model = PCA(n_components=pca_components).fit(model_data)
    pca_coords = pca_model.transform(model_data)
    if debug == True:
        print("    "*(tabs), "PCA coordinate array:", np.shape(pca_coords))
        for i, (comp, var) in enumerate(zip(pca_model.components_, pca_model.explained_variance_)):
            print("    component %d: %0.3f (cumulative: %0.3f)" % (i+1, pca_model.explained_variance_ratio_[i], np.sum(pca_model.explained_variance_ratio_[:i+1])))
            
            
    if plot == True:
        plot_PCA_summary(x, model_data, pca_model, labels, modifications=modifications, plot_components=plot_components, colour_dict=colour_dict)
        
    return model_data, pca_coords

def train_classification_model(model_type, train_data, train_labels, pca_components=6, lda_components=3, knn_neighbours=3, tabs=0, debug=False):
    # script for training a supervised spectral classification model to identify spectra
    #   - model:            name of model, determines which method is used (string)
    #   - training_data:    spectral intensities for training (2D array)
    #   - training_labels:  corresponding spectrum labels for training (list of strings)
    #   - pca_components:   number of components to calculate for PCA models (int)
    #   - lda_components:   number of components to calculate for LDA models (int)
    #   - knn_neighbours:   number of closest neighbours to use for KNN models (int)
    
    classes = np.unique(train_labels)
    if debug == True:
        print("    "*(tabs), "model:", model_type)
        print("    "*(tabs+1), "training dataset:", np.shape(train_data))
        print("    "*(tabs+1), "training labels:")
        for label in classes:
            print("    "*(tabs+2), "%s: %s" % (label, np.count_nonzero(train_labels == label)))
    
    # run appropriate supervised machine learning model
    if 'SVM' in model_type:
        # support vector machine of PCA coordinates
        svm = SVC()
        # train SVM on training dataset
        trained_model = svm.fit(train_data, train_labels)
    elif 'LDA' in model_type:
        # linear discriminant analysis of spectra
        # determine how many LDA components to calculate
        if len(classes)-1 < lda_components:
            lda_components = len(classes)-1
        lda = LinearDiscriminantAnalysis(n_components=lda_components)
        # train lda fit on training dataset
        trained_model = lda.fit(train_data, train_labels)
    elif 'KNN' in model_type:
        # K-nearest neighbours of spectra
        knn = KNeighborsClassifier(n_neighbors=knn_neighbours)
        # train KNN on training dataset
        trained_model = knn.fit(train_data, train_labels)
    elif 'Euclid' in model_type:
        # nearest match in training data, by Euclidean distance (KNN, k=1)
        euclid = KNeighborsClassifier(n_neighbors=1)
        # train euclid on training dataset
        trained_model = euclid.fit(train_data, train_labels)
        
    return trained_model

def test_classification_model(model_type, trained_model, test_data, test_labels, tabs=0, debug=False):
    # test a supervised spectral classification model on known data to assess its efficacy
    #   - model:            name of model, determines which method is used (string)
    #   - trained_model:    trained model for testing, from train_classification_model() function (object)
    #   - test_data:        spectral intensities for testing (2D array)
    #   - test_labels:      corresponding spectrum labels for testing (list of strings)
    
    classes = np.unique(test_labels)
    if debug == True:
        print("    "*(tabs), "model:", model_type)
        print("    "*(tabs+1), "testing dataset:", np.shape(test_data))
        print("    "*(tabs+1), "testing labels:")
        for label in classes:
            print("    "*(tabs+2), "%s: %s" % (label, np.count_nonzero(test_labels == label)))
        
    # use model to predict classes for test data
    prediction = trained_model.predict(test_data)
    
    # report performance
    performance = {
        'model': model_type,
        'true labels': test_labels,
        'predicted labels': prediction,
        'matrix': confusion_matrix(test_labels, prediction, labels=classes, normalize='true'),
        'accuracy': accuracy_score(test_labels, prediction),
        'precision': precision_score(test_labels, prediction, average='weighted'),
        'recall': recall_score(test_labels, prediction, average='weighted'),
        'F1': f1_score(test_labels, prediction, average='weighted')
    }
        
    return performance


def train_and_test_classification_model(model_type, model_data, model_labels, classes=[], test_fraction=0.1, min_samples_per_class=10, n_folds=1, pca_components=6, lda_components=3, knn_neighbours=3, seed=None, tabs=0, debug=False):
    # script for training and testing a supervised classification model on a given (prepared) dataset
    #   - model_type:               name of model, determines which method is used (string)
    #   - model_data:               spectral intensity data for training/testing model (2D array from prepare_data_for_classification() function)
    #   - model_labels:             corresponding spectrum labels for testing (list of strings)
    #   - classes:                  class labels to train model on (list of strings, leave blank for all available)
    #   - test_faction:             fraction of spectra to put in test dataset (float)
    #   - min_samples_per_class:    minimum number of spectra required for a class label to be included (int)
    #   - n_folds:                  number of data folds for K-fold cross-validation
    #   - pca_components:           number of components to calculate for PCA models (int)
    #   - lda_components:           number of components to calculate for LDA models (int)
    #   - knn_neighbours:           number of closest neighbours to use for KNN models (int)
    #   - seed:                     seed for random number calculation
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    print()
    print("    "*(tabs) + "training and testing classification model...")
    if type(model_type) != str:
        raise Exception("    "*(tabs+1) + "first argument passed to train_and_test_classification_model() must be a string!")
    else:
        print("    "*(tabs+1), "input model:", model_type)
    if model_data.ndim != 2:
        raise Exception("    "*(tabs+1) + "model data must be a 2D array!")
    elif len(model_labels) != np.size(model_data, axis=0):
        raise Exception("    "*(tabs+1) + "label list length does not match data array size!")
    
    if debug == True:
        print("    "*(tabs+1), "input data array:", np.shape(model_data))
        print("    "*(tabs+1), "input labels:", np.shape(model_labels))
        print("    "*(tabs+2), model_labels)
    
    model_labels = np.asarray(model_labels)
        
    # create random number generator using seed
    rng = np.random.default_rng(seed=seed)
    
    # ==================================================
    # determine classes to use for classification
        
    if len(classes) == 0:
        # no classes specified, get from label list instead
        classes = np.unique(model_labels)
    temp = []
    for label in classes:
        # check if there are enough examples of this class label in dataset
        result = np.ravel(np.where(model_labels == label))
        count = len(result)
        if debug == True:
            print("    "*(tabs), label, count, round(test_fraction*count))
        if count > 0:
            if count < min_samples_per_class:
                # not enough spectra, skip this label
                print("    "*(tabs+1), "warning: %s %s spectra found, does not meet threshold of %s - will not be included in model" % (count, label, min_samples_per_class))
            else:
                # add class label to temp array
                temp.append(label)
    print()
    print("class labels passed to model:", ", ".join(temp))
    
    # ==================================================
    # divide data into training & testing datasets
    indices = np.arange(len(model_labels))
    temp = []
    valid_indices = []
    test_indices = []
    for label in classes:
        # get indices for this class label
        result = np.ravel(np.where(model_labels == label))
        count = len(result)
        # add indices to valid_indices list
        valid_indices.append(result)
        if round(test_fraction*count) > 1:
            # select appropriate fraction of indices for test dataset
            test_indices.append(rng.choice(result, round(test_fraction*count), replace=False))
        else:
            # select minimum 1 index for test dataset
            test_indices.append(rng.choice(result, 1, replace=False))
    # concatenate results
    classes = np.asarray(temp)
    valid_indices = np.concatenate(valid_indices)
    test_indices = np.concatenate(test_indices)
    # all valid indices not in test dataset go to train dataset
    train_indices = np.setdiff1d(valid_indices, test_indices)
    
    # ==================================================
    # train and test model
    if n_folds > 1:
        # do cross-validation
        
        # split training dataset into N folds for training/validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        crossval_performance = []
        for i, (train_index, validate_index) in enumerate(kf.split(model_data[train_indices])):
            # for each fold, train and test model
            
            print("    "*(tabs), "Fold %s: training model on %s spectra, validating on %s" % (i, len(train_index), len(validate_index)))

            # train model on training dataset
            trained_model = train_classification_model(model_type, model_data[train_index], model_labels[train_index], pca_components=pca_components, lda_components=lda_components, knn_neighbours=knn_neighbours, tabs=tabs+1, debug=debug)
            
            # test model on validation dataset
            crossval_performance.append(test_classification_model(model_type, trained_model, model_data[validate_index], model_labels[validate_index], tabs=tabs+1, debug=debug))
        
        # after cross-validation, combine results from folds
        performance = {prop: [crossval_performance[i][prop] for i in range(n_folds)] for prop in ['accuracy', 'precision', 'recall', 'F1']}
        
        # report variance in performance
        print("    "*(tabs), "%s cross-validation results:" % model_type)
        for prop in ['accuracy', 'precision', 'recall', 'F1']:
            print("    "*(tabs+1), "mean %s = %0.2f (IQR: %0.2f)" % (prop, np.mean(performance[prop]), np.subtract(*np.percentile(performance[prop], [75, 25]))))
        
        
    else:
        # no cross-validation, do single train and test
        print("    "*(tabs), "training model on %s spectra, testing on %s" % (len(train_indices), len(test_indices)))
        
        # train model on training dataset
        trained_model = train_classification_model(model_type, model_data[train_indices], model_labels[train_indices], pca_components=pca_components, lda_components=lda_components, knn_neighbours=knn_neighbours, tabs=tabs+1, debug=debug)
    
        # test model
        performance = test_classification_model(model_type, trained_model, model_data[test_indices], model_labels[test_indices], tabs=tabs+1, debug=debug)
        
        print()
        print("    "*(tabs), "%s test results" % model_type)
        for prop in ['accuracy', 'precision', 'recall', 'F1']:
            print("    "*(tabs+1), "%s = %0.2f" % (prop, performance[prop]))
    
    # ==================================================
    # end of function
    if debug == True:
        print()
        print("    "*(tabs) + "end of train_and_test_classification_model() function")

"""
# ==================================================
# functions for creating generic plots
# ==================================================
"""

def get_plot_data(measurement, x_key='x', y_key='y_av', start=None, end=None, indices=[], average=False, normalise=False, tabs=0, debug=False):
    # returns all necessary information for plotting a spectrum
    # argument summary:
    #   - measurements:     Measurement instance
    #   - x_key:            key for x axis values to get from measurement (string)
    #   - y_key:            key for y axis values to get from measurement (string)
    #   - indices:          list of indices to use
    #   - norm:             normalise spectrum before averaging? (boolean)
    #   - start:            trim x range to start here
    #   - end:              trim x range to end here
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(tabs) + "getting data to plot %s v %s for %s" % (x_key, y_key, measurement.title))
    if hasattr(measurement, x_key) == False:
        raise Exception("    "*(tabs+1) + "measurement does not have %s data!" % x_key)
    if hasattr(measurement, y_key) == False:
        raise Exception("    "*(tabs+1) + "measurement does not have %s data!" % y_key)
    if type(indices) in [int, float]:
        indices = [int(indices)]
        
    if start == None:
        start = np.amin(measurement[x_key])
    if end == None:
        end = np.amax(measurement[x_key])
    
    # get x,y data
    x, y = measurement(x_key, y_key)
    if debug == True:
        print("    "*(tabs), "output x,y arrays:", np.shape(x), np.shape(y))
    
    # trim to start-end
    sliced = np.where((start <= x) & (x <= end))
    x = np.asarray(x)[sliced]
    y = np.asarray(y)[sliced]
    
    # normalise if needed
    if len(indices) > 0:
        y = y[:,indices]
    if average == True:
        y = np.mean(y, axis=1)
    if normalise == True:
        y /= np.amax(y)
        
    return x, y
    
"""
# ==================================================
# functions for quantitative Raman analysis of concentrations
# ==================================================
"""

class QuantParameter:
    # class containing all info about a quantitative model parameter
    def __init__(self, **kwargs):
        # method for setting up new instance from input values
        # check if required inputs were passed to __init__
        check = ['value', 'std']
        check = [s for s in check if s not in kwargs]
        if len(check) > 0:
            raise Exception("QuantParameter instance requires the following keyword arguments:", ", ".join(check))
        
        # add essential properties
        self.value = np.float64(kwargs['value'])
        self.std = np.float64(kwargs['std'])
        if self.value > 0:
            self.relstd = self.std / self.value
        else:
            self.relstd = 0.
            
        if 'bootstrap' in kwargs:
            self.pc1 = np.percentile(kwargs['bootstrap'], 1)
            self.pc99 = np.percentile(kwargs['bootstrap'], 99)
        else:
            self.pc1 = np.nan
            self.pc99 = np.nan
        
        # add truth-related properties
        if 'truth' in kwargs:
            self.truth = kwargs['truth']
        else:
            self.truth = np.nan
        if np.isnan(self.truth) or self.truth == None:
            self.error = np.nan
            self.relerror = np.nan
        else:
            self.error = self.value - self.truth
            self.relerror = self.error / self.truth
            
        # add any additional properties passed to __init__
        for key in kwargs:
            if key not in ['value', 'std', 'truth']:
                setattr(self, key, kwargs[key])
            
    def add(self, **kwargs):
        # generic function for adding properties to instance
        for k in kwargs:
            if k in ['name', 'value', 'std', 'relstd', 'truth', 'error', 'relerror']:
                raise Exception("%s is a protected property of QuantModel instance, cannot overwrite!" % k)
            else:
                setattr(self, k, kwargs[k])
    
    # class methods for returning values
            
    def __iter__(self):
        # returns value
        yield self.value
                
    def __getitem__(self, item):
        # method called when subscripting the instance, e.g. model1[property]
        return getattr(self, item)  # return specified property
                
    def to_dict(self):
        return {key: getattr(self, key, np.nan) for key in ['value', 'mean', 'median', 'std', 'relstd', 'truth', 'error', 'relerror', 'bootstrap']}
                
    def __str__(self):
        # return string
        if self.std > 0:
            return "%0.2E +/- %0.2E" % (self.value, self.std)
        else:
            return "%0.2E" % self.value
        
    def _getval(self):
        # return value
        return self.value
                
    # class methods for handling mathematical operations involving the QuantParameter
    
    def __array__(self):
        # array
        return np.array(np.float64(self._getval()))
    
    def __abs__(self):
        # absolute
        return abs(self._getval())

    def __neg__(self):
        # negative
        return -self._getval()

    def __pos__(self):
        # positive
        return +self._getval()

    def __bool__(self):
        # boolean
        return self._getval() != 0

    def __int__(self):
        # integer
        return int(self._getval())

    def __float__(self):
        # float
        return np.float64(self._getval())

    def __trunc__(self):
        # truncated
        return self._getval().__trunc__()

    def __add__(self, other):
        # add (left)
        return self._getval() + other

    def __radd__(self, other):
        # add (right)
        return other + self._getval()

    def __sub__(self, other):
        # subtract (left)
        return self._getval() - other

    def __rsub__(self, other):
        # subtract (right)
        return other - self._getval()

    def __truediv__(self, other):
        # divide (left)
        return self._getval() / other

    def __rtruediv__(self, other):
        # divide (right)
        return other / self._getval()

    def __floordiv__(self, other):
        # floor divide (left)
        return self._getval() // other

    def __rfloordiv__(self, other):
        # floor divide (right)
        return other // self._getval()

    def __divmod__(self, other):
        # divide and modulo (left)
        return divmod(self._getval(), other)

    def __rdivmod__(self, other):
        # divide and modulo (right)
        return divmod(other, self._getval())

    def __mod__(self, other):
        # modulo (left)
        return self._getval() % other

    def __rmod__(self, other):
        # modulo (right)
        return other % self._getval()

    def __mul__(self, other):
        # multiply (left)
        return self._getval() * other

    def __rmul__(self, other):
        # multiply (right)
        return other * self._getval()

    def __pow__(self, other):
        # raise to power
        return self._getval() ** other

    def __rpow__(self, other):
        # raise other to this power
        return other ** self._getval()

    def __gt__(self, other):
        # greater than
        return self._getval() > other

    def __ge__(self, other):
        # greater than or equal to
        return self._getval() >= other

    def __le__(self, other):
        # less than or equal to
        return self._getval() <= other

    def __lt__(self, other):
        # less than
        return self._getval() < other

    def __eq__(self, other):
        # equal to
        return self._getval() == other

    def __ne__(self, other):
        # not equal to
        return self._getval() != other
    
    # end of QuantParameter class

class QuantModel:
    # class containing all relevant info for a trained quantitative intensity/concentration model
    def __init__(self, **kwargs):
        # method for setting up new instance from input parameter values and covariance matrix (true values optional)
        
        # first check if params object contains QuantParameter instances
        success = False
        if 'params' in kwargs:
            params = kwargs['params']
            check = [key in params for key in ['FXJ', 'FXB', 'JA']]
            if np.all(check):
                # all 3 parameters must be in params dict
                check = [isinstance(params[key], QuantParameter) for key in ['FXJ', 'FXB', 'JA']]
                if np.all(check):
                    # all 3 parameters must be QuantParameter instances
                    for key in ['FXJ', 'FXB', 'JA']:
                        # add parameter data directly
                        setattr(self, key, params[key])
                    # add FXJ/JA
                    if 'FXJ_JA' in params:
                        setattr(self, 'FXJ_JA', params['FXJ_JA'])
                    else:
                        setattr(self, 'FXJ_JA', QuantParameter(
                            value = self['FXJ'] / self['JA'],
                            std = np.sqrt(self['FXJ'].relstd**2 + self['JA'].relstd**2)
                        ))
                    success = True
        # if not, proceed with individual data input
        if success == False:
            # get values and stdevs for input parameters
            temp = {'FXJ': {}, 'FXB': {}, 'JA': {}, 'FXJ_JA': {}}
            if 'params' in kwargs:
                params = kwargs['params']
                if isinstance(params, QuantParameter):
                    # handle input params as QuantParameter instance
                    for key in ['FXJ', 'FXB', 'JA']:
                        temp[key] = {}
                        for prop in params[key]:
                            temp[key][prop] = params[key][prop]
                    if 'FXJ_JA' in params.keys():
                        temp['FXJ_JA'] = {}
                        for prop in params[key]:
                            temp['FXJ_JA'][prop] = params['FXJ_JA'][prop]
                    else:
                        temp['FXJ_JA'] = {
                            'value': params['FXJ'].value / params['JA'].value,
                            'std': np.sqrt((params['FXJ'].std/params['FXJ'].value)**2 + (params['JA'].std/params['JA'].value)**2)
                        }

                elif isinstance(params, lmfit.Parameters):
                    # handle input params as LMFIT Parameters object
                    for key in ['FXJ', 'FXB', 'JA']:
                        temp[key] = {'value': params[key].value, 'std': params[key].stderr}
                    if 'FXJ_JA' in params:
                        temp['FXJ_JA'] = {
                            'value': params['FXJ_JA'].value,
                            'std': params['FXJ_JA'].stderr
                        }
                    else:
                        temp['FXJ_JA'] = {
                            'value': params['FXJ'].value / params['JA'].value,
                            'std': np.sqrt((params['FXJ'].stderr/params['FXJ'].value)**2 + (params['JA'].stderr/params['JA'].value)**2)
                        }

                elif isinstance(params, dict):
                    # handle input params as dict
                    check = [s for s in ['FXJ', 'FXB', 'JA'] if s not in params.keys()]
                    if len(check) > 0:
                        raise Exception("input parameters are missing:", ",".join(check))
                    else:
                        for key in ['FXJ', 'FXB', 'JA']:
                            temp[key] = {'value': params[key]}
                            if key+"_std" in params.keys():
                                temp[key]['std'] = params[key+"_std"]
                            else:
                                temp[key]['std'] = 0.
                        if 'FXJ_JA' in params.keys():
                            temp['FXJ_JA']['value'] = params['FXJ_JA']
                            if "FXJ_JA_std" in params.keys():
                                temp['FXJ_JA']['std'] = params["FXJ_JA_std"]
                            else:
                                temp['FXJ_JA']['std'] = 0.
                        else:
                            temp['FXJ_JA']['value'] = params['FXJ'] / params['JA']
                            if "FXJ_std" in params.keys() and "JA_std" in params.keys():
                                temp["FXJ_JA"]['std'] = np.sqrt((params["FXJ_std"]/params['FXJ'])**2 + (params["JA_std"]/params['JA'])**2)
                            else:
                                temp[key]['std'] = 0.

                else:
                    # handle input params as list (order: FXJ, FXB, JA)
                    if len(params) < 3:
                        raise Exception("input params must contain at least 3 values!")
                    else:
                        for i, key in enumerate(['FXJ', 'FXB', 'JA']):
                            temp[key] = {'value': params[key], 'std': 0.}

            elif 'FXJ' in kwargs and 'FXB' in kwargs and 'JA' in kwargs:
                # handle each input individually
                for key in ['FXJ', 'FXB', 'JA']:
                    temp[key] = {'value': kwargs[key]}
                    if key+"_std" in kwargs:
                        temp[key]['std'] = kwargs[key+"_std"]
                    else:
                        temp[key]['std'] = 0.
                if 'FXJ_JA' in kwargs:
                    temp['FXJ_JA']['value'] = kwargs['FXJ_JA']
                    if "FXJ_JA_std" in kwargs:
                        temp['FXJ_JA']['std'] = kwargs["FXJ_JA_std"]
                    else:
                        temp['FXJ_JA']['std'] = np.sqrt((kwargs["FXJ_std"]/kwargs['FXJ'])**2 + (kwargs["JA_std"]/kwargs['JA'])**2)
                else:
                    temp['FXJ_JA']['value'] = kwargs['FXJ'] / kwargs['JA']
                    if "FXJ_std" in kwargs and "JA_std" in kwargs:
                        temp["FXJ_JA"]['std'] = np.sqrt((kwargs["FXJ_std"]/kwargs['FXJ'])**2 + (kwargs["JA_std"]/kwargs['JA'])**2)
                    else:
                        temp[key]['std'] = 0.

            else:
                # no valid inputs were passed to __init__, raise Exception
                raise Exception("QuantModel instance must receive args for either params or individual parameters!")

            # add each parameter to QuantModel instance
            for key in ['FXJ', 'FXB', 'JA', 'FXJ_JA']:
                if 'truths' in kwargs and 'truth' not in temp[key].keys():
                    if key in kwargs['truths'].keys():
                        temp[key]['truth'] = kwargs['truths'][key]
                    else:
                        temp[key]['truth'] = np.nan
                else:
                    temp[key]['truth'] = np.nan

                setattr(self, key, QuantParameter(
                    value = np.float64(temp[key]['value']),
                    std = np.float64(temp[key]['std']),
                    truth = np.float64(temp[key]['truth'])
                ))
            
        # get covariance matrix (assumes matrix is constructed in order ['FXJ', 'FXB', 'JA'])
        temp = np.zeros((3,3), dtype=np.float64)
        if 'cov_matrix' in kwargs:
            # use values from input
            cov_matrix = kwargs['cov_matrix']
            if isinstance(cov_matrix, np.ndarray):
                # handle input as numpy array
                if np.size(cov_matrix, axis=0) < 3 or np.size(cov_matrix, axis=1) < 3:
                    raise Exception("covariance matrix must be at least 3x3!")
                else:
                    temp = cov_matrix[:3,:3]
            elif isinstance(cov_matrix, pd.DataFrame):
                # handle input as pandas dataframe
                if np.size(cov_matrix, axis=0) < 3 or np.size(cov_matrix, axis=1) < 3:
                    raise Exception("covariance matrix must be at least 3x3!")
                else:
                    temp = np.asarray(cov_matrix)[:3,:3]
            else:
                # handle input as list of lists
                if len(cov_matrix) < 3:
                    raise Exception("covariance matrix must be at least 3x3!")
                elif np.any([len(col) for col in cov_matrix[:3]]) < 3:
                    raise Exception("covariance matrix must be at least 3x3!")
                else:
                    temp = np.asarray(cov_matrix)[:3,:3]
        
        # add covariance matrix to QuantModel instance
        setattr(self, 'cov_matrix', temp)
        
        # add any additional input kwargs to instance
        for k in kwargs:
            if k not in ['FXJ', 'FXB', 'JA', 'FXJ_JA', 'params', 'cov_matrix']:
                setattr(self, k, kwargs[k])
        
        # __init__ complete
        del temp
        
    def add(self, **kwargs):
        # generic function for adding properties to instance
        for k in kwargs:
            if k in ['FXJ', 'FXB', 'JA', 'FXJ_JA', 'cov_matrix']:
                raise Exception("%s is a protected property of QuantModel instance, cannot overwrite!" % k)
            else:
                setattr(self, k, kwargs[k])
                
    # methods for getting values from instance
    
    def __getitem__(self, item):
        # method called when subscripting the instance, e.g. model1[property]
        if isinstance(getattr(self, item), QuantParameter):
            return getattr(self, item)
        else:
            return getattr(self, item)  # return specified property
        
    def predict_concentration(self, IR, IR_err=None):
        prediction, uncertainty = predict_concentration(self, IR, IR_err)
        return prediction, uncertainty
    
    def predict_intensity(self, Cr, CR_err=None):
        prediction, uncertainty = predict_intensity(self, CR, CR_err)
        return prediction, uncertainty
    
    def predict_detection_limits(self, limit='lower', **kwargs):
        if limit == 'lower':
             return detection_limit(self, limit='upper', **kwargs)
        else:
             return detection_limit(self, limit='lower', **kwargs)
    
    # end of QuantModel class
    
# quality of life functions

def parameter2string(param, n=2, e=False):
    # convert mean and uncertainty to string representation as follows: (mean±std)x10^mag
    if isinstance(param, QuantParameter):
        median = param.median
    else:
        median = param
        
    if e == True:
        mag_joiner = "E"
    else:
        mag_joiner = "x10^"
        
    if np.isnan(median):
        text = "NaN"
    elif np.isinf(median):
        text = "inf"
    elif median == 0:
        text = "0"
    else:
        mag = int(np.floor(np.log10(median)))
        c = median / 10**mag
        if e == True:
            text = f"{c:.{n}f}{mag_joiner}{mag:.0f}"
        else:
            text = f"{c:.{n}f}{mag_joiner}{mag:.0f}"
    return text

def percentiles2string(param=None, percentiles=(16, 84), n=2, include_median=False, e=False, **kwargs):
    # convert median and percentiles to string representation as follows: (median, lower - upper)x10^mag
    
    if isinstance(param, QuantParameter):
        median = param.median
        lower = np.nanpercentile(param.bootstrap, percentiles[0])
        upper = np.nanpercentile(param.bootstrap, percentiles[1])
    elif 'median' in kwargs and 'lower' in kwargs and 'upper' in kwargs:
        median = kwargs['median']
        lower = kwargs['lower']
        upper = kwargs['upper']
    elif isinstance(param, np.ndarray):
        # handle input as distribution 
        median = np.nanmedian(param)
        lower = np.nanpercentile(param, percentiles[0])
        upper = np.nanpercentile(param, percentiles[1])
    else:
        raise Exception("percentiles2string() must receive either QuantParameter object, or median,lower,upper kwargs, or distribution array!")
        
    mag = None
    if include_median == True:
        if np.isnan(median):
            mtext = "NaN"
        elif np.isinf(median):
            mtext = "inf"
        elif median == 0:
            mtext = "0"
        else:
            mag = np.floor(np.log10(np.abs(median)))
            c = median / 10**mag
            mtext = f"{c:.{n}f}"
        
    text = []
    for val in [lower, upper]:
        if np.isnan(val):
            text.append("NaN")
        elif np.isinf(val):
            text.append("inf")
        elif val == 0:
            text.append("0")
        else:
            if mag == None:
                mag = np.floor(np.log10(np.abs(val)))
            c = val / 10**mag
            text.append(f"{c:.{n}f}")
    if mag == None:
        mag = 0
        
    if e == True:
        mag_joiner = "E"
    else:
        mag_joiner = "x10^"
    
    if include_median == True:
        return f"({mtext}, {'-'.join(text)}){mag_joiner}{mag:.0f}"
    else:
        return f"({'-'.join(text)}){mag_joiner}{mag:.0f}"
    
def range2string(lower, upper, n=2, e=False):
    
    if e == True:
        mag_joiner = "E"
    else:
        mag_joiner = "x10^"
    
    text = []
    for val in [lower, upper]:
        if np.isnan(val):
            text.append("NaN")
        elif np.isinf(val):
            text.append("inf")
        elif val == 0:
            text.append("0")
        else:
            mag = np.floor(np.log10(np.abs(val)))
            c = val / 10**mag
            if e == True:
                text.append(f"{c:.{n}f}{mag_joiner}{mag:.0f}")
            else:
                text.append(f"{c:.{n}f}{mag_joiner}{mag:.0f}")
        
    return "-".join(text)
        
# functions for bootstrapping

def bootstrap_resample_indices(indices, seed=None):
    # set up random number generator
    rng = np.random.default_rng(seed)
    # get random resample of indices
    resampled = rng.choice(indices, size=len(indices), replace=True)
    return resampled

def bootstrap_resample_data(data, n_bootstraps, seed=None):
    # function for bootstrap resampling of a dataset
    
    # set up storage array
    data_length = np.size(data, axis=0)
    resampled_means = np.zeros((data_length, n_bootstraps), dtype=np.float64)
    resampled_errs = np.zeros((data_length, n_bootstraps), dtype=np.float64)
    for i1 in range(data_length):
        # for each input data point
        # remove nans
        temp = data[i1][~np.isnan(data[i1])]
        for i2 in range(n_bootstraps):
            # set up random number generator
            if seed == None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(int(seed) + 10*i1 + 100*i2)
            # get random resample
            resample = rng.choice(range(len(temp)), size=len(temp), replace=True)
            resampled_means[i1,i2] = np.nanmean(temp[resample])
            resampled_errs[i1,i2] = np.nanstd(temp[resample])
    return resampled_means, resampled_errs

def get_covariance_matrix(data, return_data=False):
    # function for getting a covariance matrix from estimated parameter distributions
    keys = ['FXJ', 'FXB', 'JA']
    for i, key in enumerate(keys):
        if key not in data.keys():
            raise Exception('%s not in input data dict!' % key)
    # add composite data to dict
    data['FXJ_JA'] = data['FXJ'] / data['JA']
    # generate covariance matrix
    keys = list(data.keys())
    temp = np.asarray([val for key, val in data.items()])
    cov_matrix = pd.DataFrame(np.cov(temp), columns=keys, index=keys)
    if return_data == True:
        return cov_matrix, data
    else:
        return cov_matrix

# functions for getting concentration curves

def concentration_curve(model, IR, debug=False):
    # function for predicting concentration from intensity using quantitative model
    curve = (IR - model["FXB"]) / (model["FXJ"] - model["JA"] * IR)
    return curve

def concentration_curves(model, IR, debug=False):
    # function for predicting intensity distribution using all models from bootstrapping
    params = {prop: model[prop].bootstrap[:,np.newaxis] for prop in ['FXJ', 'FXB', 'JA']}
    curve = (IR[np.newaxis,:] - params["FXB"]) / (params["FXJ"] - params["JA"] * IR[np.newaxis,:])
    return curve

def concentration_error(model, IR, IR_err=0., debug=False):
    # function for estimating uncertainty in concentration prediction from input error and model covariance matrix
    if isinstance(IR_err, np.ndarray):
        if np.shape(IR_err) != np.shape(IR):
            IR_err = np.zeros_like(IR)
    else:
        IR_err = np.zeros_like(IR)
    if debug == True:
        print(np.shape(IR), np.shape(IR_err))
    variances = []
    for ir, err in zip(np.ravel(IR), np.ravel(IR_err)):
        # calculate Jacobian matrix terms
        dCR_dIR = (model['FXJ'] - model['JA'] * model['FXB']) / (model['FXJ'] - model['JA'] * ir)**2
        dCR_dFXJ = -(ir - model['FXB']) / (model['FXJ'] - model['JA'] * ir)**2
        dCR_dFXB = -1. / (model['FXJ'] - model['JA'] * ir)
        dCR_dJA = ir * (ir - model['FXB']) / (model['FXJ'] - model['JA'] * ir)**2
        jacobian_matrix = np.asarray([dCR_dFXJ, dCR_dFXB, dCR_dJA])
        if debug == True:
            print("IR=%0.1E +/- %0.1E :" % (ir, err))
            print("    dCR/dIR:  %0.1E" % dCR_dIR)
            print("    dCR/dFXJ: %0.1E" % dCR_dFXJ)
            print("    dCR/dFXB: %0.1E" % dCR_dFXB)
            print("    dCR/dJA:  %0.1E" % dCR_dJA)
        # calculate variance using numerical matrix method (Tellinghuisen 2001) 
        variances.append(dCR_dIR**2 * err**2 + jacobian_matrix.T @ model['cov_matrix'][:3,:3] @ jacobian_matrix)
    variances = np.reshape(np.asarray(variances), np.shape(IR))
    return np.sqrt(variances)

# functions for getting intensity curves

def intensity_curve(model, CR, debug=False):
    # function for predicting intensity from concentration using quantitative model
    curve = (model["FXJ"] * CR + model["FXB"]) / (model["JA"] * CR + 1)
    return curve

def intensity_curves(model, CR, debug=False):
    # function for predicting intensity distribution using all models from bootstrapping
    params = {prop: model[prop].bootstrap[:,np.newaxis] for prop in ['FXJ', 'FXB', 'JA']}
    curve = (params["FXJ"] * CR[np.newaxis,:] + params["FXB"]) / (params["JA"] * CR[np.newaxis,:] + 1)
    return curve

def intensity_error(model, CR, CR_err=0., debug=False):
    # function for estimating uncertainty in intensity prediction from input error and model covariance matrix
    if isinstance(CR_err, np.ndarray):
        if np.shape(CR_err) != np.shape(CR):
            CR_err = np.zeros_like(CR)
    else:
        CR_err = np.zeros_like(CR)
    if debug == True:
        print(np.shape(CR), np.shape(CR_err))
    variances = []
    for cr, err in zip(np.ravel(CR), np.ravel(CR_err)):
        # calculate Jacobian matrix terms
        dIR_dCR = (model['FXJ'] - model['JA'] * model['FXB']) / (model['JA'] * cr + 1)**2
        dIR_dFXJ = cr / (model['JA'] * cr + 1)
        dIR_dFXB = 1. / (model['JA'] * cr + 1)
        dIR_dJA = - cr * (model['FXJ'] * cr + model['FXB']) / (model['JA'] * cr + 1)**2
        jacobian_matrix = np.asarray([dIR_dFXJ, dIR_dFXB, dIR_dJA])
        if debug == True:
            print("CR=%0.1E +/- %0.1E :" % (cr, err))
            print("    dIR/dCR:  %0.1E" % dIR_dCR)
            print("    dIR/dFXJ: %0.1E" % dIR_dFXJ)
            print("    dIR/dFXB: %0.1E" % dIR_dFXB)
            print("    dIR/dJA:  %0.1E" % dIR_dJA)
        # calculate variance using input error and numerical matrix method (Tellinghuisen 2001)
        variances.append(dIR_dCR**2 * err**2 + jacobian_matrix.T @ model['cov_matrix'][:3,:3] @ jacobian_matrix)
    variances = np.reshape(np.asarray(variances), np.shape(CR))
    return np.sqrt(variances)

# functions for training a model on data

def intensity_fit(model, CR, IR, weights):
    # objective function for fitting an intensity/concentration curve using quantitative model
    curve = (model["FXJ"] * CR + model["FXB"]) / (model["JA"] * CR + 1)
    if np.any(np.isnan(curve)):
        curve[np.isnan(curve)] = 10**-20
    if np.any(np.isinf(curve)):
        curve[np.isinf(curve)] = 10**-20
    if np.any(curve <= 0):
        curve[curve <= 0] = 10**-20
    
    if model["log"] == 1:
        return (np.log10(IR) - np.log10(curve)) * weights
    else:
        return (IR - curve) * weights

def intensity_fit_script(CR, IR, IR_err=None, A=0.01, B=0.01, J=1., F=1., X=1., noise=0., truths={}, log=True, vary_A=True, vary_B=True, vary_FX=True, outlier_threshold=2., report_threshold=0.2, title="", plot_variance=False, fig_dir="./figures/", tabs=0, return_quantmodel=False, save_plot=False, show_plot=True, debug=False):
    # function for setting up and fitting the quantitative model on an input dataset of concentration and intensity ratios
    # argument summary:
    #   - CR:               array of concentration ratios for fitting
    #   - IR:               array of intensity ratios for fitting
    #   - IR_err:           array of intensity ratio uncertainties
    #   - A:                initial value for parameter A (float)
    #   - B:                initial value for parameter B (float)
    #   - J:                initial value for parameter J (float)
    #   - F:                initial value for parameter F (float)
    #   - X:                initial value for parameter X (float)
    #   - noise:            estimated background noise level (float)
    #   - log:              is the fitting in log space or not? (boolean)
    #   - vary_A:           is the A parameter allowed to vary? (boolean)
    #   - vary_B:           is the B parameter allowed to vary? (boolean)
    #   - vary_FX:          is the FX parameter allowed to vary? (boolean)
    #   - outlier_threshold:    minimum factor for defining outliers based on st.dev. of 1st fit residuals (float)
    #   - report_threshold:     minimum fraction of data points that are outliers before a report is triggered
    #   - title:                name to put on generated figures
    #   - plot_variance:        generate a figure showing co-variances? (boolean)
    #   - fig_dir:              directory for saving figure files
    #   - tabs:                 number of tabs to put before debug messages (str)
    #   - debug:                if True print all debug messages in viewer (boolean)
    
    # clean up input data
    CR = CR.astype(np.float64)
    IR = IR.astype(np.float64)
    A = np.float64(A)                   # internal cross-section ratio for molecule A at v2 / v1
    B = np.float64(B)                   # internal cross-section ratio for molecule B at v1 / v2
    J = np.float64(J)                   # cross-section ratio for A at v1 / B at v2
    FX = np.float64(F) * np.float64(X)  # F: instrument sensitivity ratio for v1 / v2 ; X: transmission ratio of sample for v1 / v2
    
    # generate data weighting based on signal:noise, where 1 is fully weighted and 0 is ignored
    weights = np.ones_like(IR)
        
    # remove data points that will mess up the maths
    notnan_check = np.logical_and(~np.isnan(CR), ~np.isnan(IR))
    notinf_check = np.logical_and(~np.isinf(CR), ~np.isinf(IR))
    positive_check = np.logical_and(CR > 0, IR > 0)
    clean = np.logical_and(notnan_check, notinf_check)
    if log == True:
        # also trim any IR data that cannot be log'd
        clean = np.logical_and(clean, positive_check)
    CR = CR[clean]
    IR = IR[clean]
    weights = weights[clean]
        
    # set up initial parameters
    params = lmfit.Parameters()
    params.add("FXJ", value=FX*J, min=10**-40)
    params.add("FXB", value=FX*B, min=0)
    params.add("JA", value=J*A, min=0)
    params.add("log", value=log, min=0, max=1, vary=False)
    if debug == True:
        print()
        print("    "*(tabs), "input data:")
        print(np.shape(CR), np.shape(IR))
        print("    "*(tabs+1), "nans:", np.count_nonzero(~notinf_check))
        print("    "*(tabs+1), "infs:", np.count_nonzero(~notnan_check))
        print("    "*(tabs+1), "<=0: ", np.count_nonzero(~positive_check))
        print("    "*(tabs+1), "points trimmed:", np.size(CR) - np.count_nonzero(clean))
        print("    "*(tabs), "input parameters:")
        print(params.pretty_print())
        print("    "*(tabs), "proceeding with fit")
        
    # do fitting
    repeat = False
    fit_output = lmfit.minimize(intensity_fit, params, args=(CR, IR, weights), method='nelder')
    if debug == True:
        print("    "*(tabs), "fit status: ", fit_output.message)
        print("    "*(tabs), "fitted parameters:")
        print(fit_output.params.pretty_print())
    
    # check for outliers (difference in log >2 st. devs. and difference in log > 1)
    fit_curve = intensity_curve(fit_output.params, CR, debug=False)
    threshold = outlier_threshold * np.std(np.log10(fit_curve) - np.log10(IR))
    outlier_check = np.logical_and(np.abs(np.log10(IR) - np.log10(fit_curve)) >= threshold, np.abs(np.log10(IR) - np.log10(fit_curve)) > 1)
    bad_points = np.ravel(np.where(outlier_check == True))
    
    if np.size(bad_points) > report_threshold * np.size(CR):
        # too many outliers, print report and generate outlier summary figure
        repeat = True
        debug = True
        show_plot = True
        print("    "*(tabs), "error check: %s/%s" % (np.count_nonzero(outlier_check), np.size(outlier_check)))
        print("    "*(tabs), "at least one error found, report:")
        print()
        print("    "*(tabs), "input data:", np.shape(CR))
        print("    "*(tabs+1), "nans:", np.count_nonzero(~notinf_check))
        print("    "*(tabs+1), "infs:", np.count_nonzero(~notnan_check))
        print("    "*(tabs+1), "<=0: ", np.count_nonzero(~positive_check))
        print("    "*(tabs+1), "points trimmed:", np.count_nonzero(~clean))
        print()
        print("    "*(tabs), "data vs model:")
        print("    "*(tabs+1), "log-space threshold: +/-%0.1f" % threshold)
        for i in bad_points:
            print("    point %s: %0.2E vs %0.2E (%0.1f)" % (i, IR[i], fit_curve[i], np.log10(fit_curve[i])-np.log10(IR[i])))
        
    if np.all(outlier_check) == True:
        # all points registered as outliers, cannot proceed with 2nd fit
        print()
        print("    "*(tabs), "MAJOR ERROR: all points classed as outliers!")
        print("    "*(tabs), "fitted parameters:")
        print(fit_output.params.pretty_print())
        print("    "*(tabs), "data range:  %0.1E - %0.1E" % (np.amin(IR), np.amax(IR)))
        print("    "*(tabs), "model range: %0.1E - %0.1E" % (np.amin(fit_curve), np.amax(fit_curve)))
        plt.figure(figsize=(6,4))
        plt.title("Outlier Report\n%s" % title)
        plt.plot(CR[~outlier_check], IR[~outlier_check], 'ko', label='good (%s)' % np.count_nonzero(~outlier_check))
        plt.plot(CR[outlier_check], IR[outlier_check], 'ro', label='bad (%s)' % np.count_nonzero(outlier_check), zorder=4)
        plt.fill_between(CR, 10**(np.log10(fit_curve)-threshold), 10**(np.log10(fit_curve)+threshold), color='b', alpha=0.1)
        plt.plot(CR, fit_curve, 'b', label='fit 1')
        x_min = 10**np.floor(np.log10(0.9*np.amin(CR[IR > 0])))
        x_max = 10**np.ceil(np.log10(1.1*np.amax(CR[IR > 0])))
        plt.xscale('log')
        plt.xlim(x_min, x_max)
        y_min = 10**np.floor(np.log10(0.9*np.amin(IR[IR > 0])))
        y_max = 10**np.ceil(np.log10(1.1*np.amax(IR[IR > 0])))
        plt.yscale('log')
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()
        plt.savefig("%s%s_outliers.png" % (Fig_dir, title), dpi=300)
        plt.close()
        
    elif np.any(outlier_check) == True:
        # rerun fit without outliers
        repeat = True
        new_output = lmfit.minimize(intensity_fit, params, args=(CR[~outlier_check], IR[~outlier_check], weights[~outlier_check]), method='nelder')
        new_curve = intensity_curve(new_output.params, CR, debug=False)
        if debug == True:
            print("    "*(tabs), "fit status: ", new_output.message)
            print("    "*(tabs), "fitted parameters:")
            print(new_output.params.pretty_print())
        
    if show_plot == True or save_plot == True:
        plt.figure(figsize=(6,4))
        plt.title("Outlier Report\n%s" % title)
        plt.plot(CR[~outlier_check], IR[~outlier_check], 'ko', label='exp (%s)' % np.count_nonzero(~outlier_check))
        plt.plot(CR[outlier_check], IR[outlier_check], 'ro', label='bad (%s)' % np.count_nonzero(outlier_check), zorder=4)
        plt.fill_between(CR, 10**(np.log10(fit_curve)-threshold), 10**(np.log10(fit_curve)+threshold), color='b', alpha=0.1)
        plt.plot(CR, fit_curve, 'b', label='fit 1')
        if repeat == True:
            plt.plot(CR, new_curve, 'm', label='fit 2')
        x_min = 10**np.floor(np.log10(0.9*np.amin(CR[IR > 0])))
        x_max = 10**np.ceil(np.log10(1.1*np.amax(CR[IR > 0])))
        plt.xscale('log')
        plt.xlim(x_min, x_max)
        y_min = 10**np.floor(np.log10(0.9*np.amin(IR[IR > 0])))
        y_max = 10**np.ceil(np.log10(1.1*np.amax(IR[IR > 0])))
        plt.yscale('log')
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_outliers.png" % (fig_dir, title), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        
    if repeat == True:
        result = new_output
    else:
        result = fit_output
    
    # add outlier info to output dict
    result.clean_indices = np.ravel(np.where(clean))
    result.outlier_indices = np.arange(np.size(clean))[clean][outlier_check]
    result.outlier_rate = np.count_nonzero(outlier_check) / np.size(outlier_check)
        
    # report fitted parameters
    if debug == True:
        for prop in ['FXJ', 'FXB', 'JA']:
            if result.params[prop].stderr == None or np.isnan(result.params[prop].stderr) == True:
                print("    "*(tabs), "%3s = %0.1E (no err)" % (prop, result.params[prop].value))
            else:
                print("    "*(tabs), "%3s = %0.1E +/- %0.1E" % (prop, result.params[prop].value, result.params[prop].stderr))
    
    if return_quantmodel == True:
        # generate QuantModel
        model = QuantModel(
            params = result.params,
            corr_matrix = np.full((3,3), np.nan),
            cov_matrix = np.full((3,3), np.nan),
            truths = truths,
            training_data = {},
            bootstrap_data = {}
        )
        return model
    else:
        # return fitted model
        return result.params


def bootstrap_intensity_fit_script(CR, IR, n_bootstraps=1000, sample_size=1., report_threshold=1., truths={}, title="", seed=None, plot_covariance=False, fig_dir="./figures/", x_lims={}, plot_linear=False, show_text=True, save_plot=False, show_plot=True, debug=False, tabs=0, **kwargs):
    # function for doing bootstrap training
    # argument summary:
    #   - 
    #   - tabs:             number of tabs to put before debug messages (str)
    #   - debug:            if True print all debug messages in viewer (boolean)
    
    # check input data
    n_bootstraps = int(n_bootstraps)
    if IR.ndim == 1:
        # force array into 2D
        IR = IR[:,np.newaxis]
    if np.size(IR, axis=1) == 1:
        print("caution: only 1 value per data point! Bootstrapping will not generate meaningful uncertainties")
    if np.size(IR, axis=1) <= 5:
        print("caution: only %s values per data point! Bootstrapping may not generate meaningful uncertainties" % np.size(IR, axis=1))
    if isinstance(x_lims, dict) == False:
        x_lims = {'FXJ': (None, None), 'FXB': (None, None), 'JA': (None, None), 'FXJ_JA': (None, None)}

    # prepare bootstrap results storage array
    bootstrap_results = {key: np.zeros(n_bootstraps, dtype=np.float64) for key in ['FXJ', 'FXB', 'JA', 'FXJ_JA']}
    
    # generate bootstrapped data
    IR_bstrp_av, IR_bstrp_std = bootstrap_resample_data(IR, n_bootstraps, seed=seed)
    if debug == True:
        print()
        print("    "*(tabs), "bootstrapping input data, N = %d" % n_bootstraps)
        print("    "*(tabs+1), " input data array:", np.shape(IR))
        print("    "*(tabs+1), "output data array:", np.shape(IR_bstrp_av))

    if debug == True:
        print()
        print("    "*(tabs), "proceeding with bootstrap modelling, N = %d" % n_bootstraps)
    count = 0
    
    for i in range(n_bootstraps):
        # for each bootstrap resample

        # fit model to intensity curve
        if i == 0 and debug == True:
            debug1=True
        else:
            debug1=False
        fit_output = intensity_fit_script(CR, IR_bstrp_av[:,i], IR_err=IR_bstrp_std[:,i], report_threshold=report_threshold, show_plot=False, save_plot=False, debug=debug1)

        # add fitted parameter values to storage array
        for prop in ['FXJ', 'FXB', 'JA']:
            bootstrap_results[prop][i] = fit_output[prop].value
        bootstrap_results['FXJ_JA'][i] = fit_output['FXJ'].value / fit_output['JA'].value

        if 10.*(i+1) / n_bootstraps >= count + 1 and debug == True:
            count += 1
            print("    "*(tabs), "%s/%s (%3.f%%) models fitted" % (i+1, n_bootstraps, 10*count))

    # get means and standard deviations for each parameter
    mean_params = {}
    for prop in ['FXJ', 'FXB', 'JA']:
        # create QuantParameter instance for this parameter
        if isinstance(truths, dict):
            if prop in truths:
                truth = truths[prop]
            else:
                truth = np.nan
        mean_params[prop] = QuantParameter(
            bootstrap = bootstrap_results[prop],
            value = np.nanmean(bootstrap_results[prop]),
            mean = np.nanmean(bootstrap_results[prop]),
            median = np.nanmedian(bootstrap_results[prop]),
            std = np.nanstd(bootstrap_results[prop]),
            truth = truth
        )
        
    # do the same for FXJ/JA
    prop = 'FXJ_JA'
    if isinstance(truths, dict):
        if prop in truths:
            truth = truths[prop]
        else:
            truth = np.nan
    # calculate std directly from medians and std of FXJ, JA to avoid infs
    std = (mean_params['FXJ'].median / mean_params['JA'].median) * np.sqrt(mean_params['FXJ'].relstd**2 + mean_params['JA'].relstd**2 - 2*np.cov([bootstrap_results['FXJ'], bootstrap_results['JA']])[0,1]/(mean_params['FXJ'].mean * mean_params['JA'].mean))
    
    mean_params[prop] = QuantParameter(
        bootstrap = bootstrap_results['FXJ'] / bootstrap_results['JA'],
        value = mean_params['FXJ'].median / mean_params['JA'].median,
        mean = mean_params['FXJ'].mean / mean_params['JA'].mean,
        median = mean_params['FXJ'].median / mean_params['JA'].median,
        std = std,
        truth = truth
    )

    if debug == True:
        for prop in ['FXJ', 'FXB', 'JA', 'FXJ_JA']:
            # report results for each parameter
            print()
            print("    "*(tabs), "%s:" % prop)
            if np.isnan(mean_params[prop].truth) == False:
                print(""*(tabs), "          truth: %0.2E" % mean_params[prop].truth)
            print("    "*(tabs), "           mean: %0.2E" % mean_params[prop].mean)
            print("    "*(tabs), "         median: %0.2E" % mean_params[prop].median)
            print("    "*(tabs), "          stdev: %0.2E (%3.3f%%)" % (mean_params[prop].std, 100. * mean_params[prop].relstd))
            y = mean_params[prop].bootstrap.copy()
            check = np.logical_or(y <= 0, np.isinf(y))
            y[check] = np.nan
            print("    "*(tabs), "  min-max range: %0.1E - %0.1E" % (np.nanmin(y), np.nanmax(y)))
            if np.isnan(mean_params[prop].error) == False:
                    print("    "*(tabs), "          error: %0.1E (%3.3f%%)" % (mean_params[prop].error, 100. * mean_params[prop].relerror))
                    
            
    # calculate correlation coefficients
    corr = np.corrcoef([value for key, value in bootstrap_results.items()])
    cov_matrix = np.cov([value for key, value in bootstrap_results.items()])
    
    # convert results into QuantModel instance
    mean_model = QuantModel(
        params = mean_params,
        corr_matrix = corr,
        cov_matrix = cov_matrix,
        truths = truths,
        n_bootstraps = n_bootstraps,
        n_training = np.size(IR),
        training_data = {'CR': CR, 'IR_av': np.nanmean(IR, axis=1), 'IR_std': np.nanstd(IR, axis=1)},
        bootstrap_data = {'CR': CR, 'IR_av': IR_bstrp_av, 'IR_std': IR_bstrp_av}
    )
    
    # calculate inherent limits of detection
    LL_model = detection_limit(mean_model, threshold=3., limit='lower', return_all=True, debug=False)['model']
    UL_model = detection_limit(mean_model, threshold=3., limit='upper', return_all=True, debug=False)['model']
    
    setattr(mean_model, 'LLoD_model', LL_model)
    setattr(mean_model, 'ULoD_model', UL_model)
    
    # calculate inherent limits of quantitation
    LL_model = detection_limit(mean_model, threshold=10., limit='lower', return_all=True, debug=False)['model']
    UL_model = detection_limit(mean_model, threshold=10., limit='upper', return_all=True, debug=False)['model']
    
    setattr(mean_model, 'LLoQ_model', LL_model)
    setattr(mean_model, 'ULoQ_model', UL_model)

    # plot summary figures for bootstrapping, if required
    if show_plot == True or save_plot == True:
        plot_quantmodel_summary(mean_model, fig_dir=fig_dir, title=title, x_lims=x_lims, plot_covariance=plot_covariance, seed=seed,  plot_linear=plot_linear, show_text=show_text, show_plot=show_plot, save_plot=save_plot, tabs=tabs, debug=debug)
        
    return mean_model

def plot_quantmodel_summary(model, CR=None, IR=None, plot_covariance=False, title="", fig_dir="./figures/", x_lims={'FXJ': (None, None), 'FXB': (None, None), 'JA': (None, None), 'FXJ_JA': (None, None)}, plot_linear=False, seed=None, show_text=True, save_plot=False, show_plot=True, debug=False, tabs=0):
    # function for easy plotting of model summary figure
    
    #   - tabs:             number of tabs to put before debug messages (str)
    #   - debug:            if True print all debug messages in viewer (boolean)
    
    # check model contains necessary parameter data
    props = ['FXJ', 'FXB', 'JA', 'FXJ_JA']
    n_bootstraps = model.n_bootstraps
    check = [prop for prop in props if hasattr(model, prop) == False]
    if len(check) > 0:
        raise Exception("cannot generate summary figure for model missing data for parameters %s" % ", ".join(check))
    if isinstance(x_lims, dict) == False:
        x_lims = {prop: (None, None) for prop in props}
    
    # check for valid input data
    input_data = {'CR': [], 'IR_av': [], 'IR_std': []}
    if hasattr(model, 'training_data'):
        check = [prop for prop in ['CR', 'IR_av', 'IR_std'] if prop not in model['training_data'].keys()]
        if len(check) > 0:
            raise Exception("input training_data does not contain %s" % ", ".join(check))
        else:
            input_data = model['training_data']
    elif hasattr(model, 'input_data'):
        check = [prop for prop in ['CR', 'IR_av', 'IR_std'] if prop not in model['input_data'].keys()]
        if len(check) > 0:
            raise Exception("input training_data does not contain %s" % ", ".join(check))
        else:
            input_data = model['input_data']
    elif isinstance(CR, np.ndarray) and isinstance(IR, np.ndarray):
        if IR.ndim > 1:
            input_data = {'CR': CR, 'IR_av': np.nanmean(IR, axis=1), 'IR_std': np.nanstd(IR, axis=1)}
        else:
            input_data = {'CR': CR, 'IR_av': IR, 'IR_std': np.zeros_like(IR)}
    
    if save_plot == False and show_plot == False:
        print("    "*(tabs), "plot_quantmodel_summary() received show_plot=False and save_plot=False, will not generate anything")
    else:
        # ==================================================
        # set up single-row figure to plot individual parameter distributions
        if plot_covariance == True and hasattr(model, 'cov_matrix'):
            fig, axs = plt.subplots(2, 4, figsize=(16,8))
        else:
            fig, axs = plt.subplots(1, 4, figsize=(16,4))
        
        # determine alpha (transparency) for data
        alpha = float(n_bootstraps)**-0.5
        
        # first ax: intensity curves
        ax = axs.flat[0]
        ax.set_title(title)
        ax.set_xlabel("Concentration Ratio")
        ax.set_ylabel("Intensity Ratio")
        
        # determine x,y axis limits
        neg_check = input_data['IR_av'] > 0
        x_min = 10**np.floor(np.log10(0.9*np.nanmin(input_data['CR'])))
        x_max = 10**np.ceil(np.log10(1.1*np.nanmax(input_data['CR'])))
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        y_min = 10**np.floor(np.log10(0.9*np.nanmin(input_data['IR_av'][neg_check])))
        y_max = 10**np.ceil(np.log10(1.1*np.nanmax(input_data['IR_av'][neg_check])))
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)

        if len(input_data['CR']) > 0:
            # add input data with errorbars
            ax.errorbar(input_data['CR'], input_data['IR_av'], yerr=input_data['IR_std'], fmt='none', ecolor='k', capsize=3, zorder=6)
            ax.scatter(input_data['CR'], input_data['IR_av'], c='k', label='data', zorder=5)
            
        # add mean model curve as dashed black line, confidence interval band as red region
        x_temp = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        pred, upper_conf, lower_conf = predict_intensity(model, x_temp, percentiles=(2.5, 97.5), seed=seed, tabs=tabs+1, debug=False)
        ax.plot(x_temp, pred, 'r', label='mean model', zorder=4)
        ax.fill_between(x_temp, lower_conf, upper_conf, color='r', linewidth=0., alpha=0.2, label='95% CI')
        
        if plot_linear == True:
            ax.plot(x_temp, model['FXJ'].median * x_temp, 'r:', label='linear')
        
        # finish ax
        ax.legend(loc='upper left')
        
        # plot each parameter distribution in its own ax
        for ax, prop in zip(axs.flat[1:4], ['FXJ', 'FXB', 'JA']):
            
            # get x_lims
            check = False
            if isinstance(x_lims, dict):
                if prop in x_lims:
                    if len(x_lims[prop]) == 2:
                        x_min, x_max = x_lims[prop]
                        check = True
            if check == False:
                # no suitable ax limits defined for this property, get best results based on distribution of values
                mean = model[prop].mean
                std = model[prop].std
                if model[prop].mean == 0. and model[prop].std == 0.:
                    x_min, x_max = (-0.001, 0.001)
                else:
                    x_min = np.amin([model[prop].mean - 5 * model[prop].std, 0.99*model[prop].mean])
                    x_max = np.amax([model[prop].mean + 5 * model[prop].std, 1.01*model[prop].mean])
            
            # initialise ax
            ax.set_title("%s Distribution" % prop)
            ax.set_xlabel(prop)
            
            # plot mean value
            ax.axvline(model[prop].mean, c='k', linestyle=':', label="mean")
            ax.axvline(model[prop].median, c='k', linestyle='--', label="median")
            
            # set up CR range for plotting predicted intensity curves
            x_temp = np.logspace(np.log10(x_min), np.log10(x_max), 100)
            
            # plot histogram
            hist, bins, patches = ax.hist(model[prop].bootstrap, range=(x_min, x_max), bins=50, color='steelblue')
                
            # add text for distribution properties 
            text = [
                "Median: %0.1E" % model[prop].median,
                " 1st %%: %0.1E" % model[prop].pc1,
                "99th %%: %0.1E" % model[prop].pc99
            ]
                
            if np.all(~np.isnan(hist)) and model[prop].std > 0 and bins[-1]-bins[0] > 0:
                # plot normal distribution based on mean, std
                params = lmfit.Parameters()
                params.add('gradient', value=0., vary=False)
                params.add('intercept', value=0., vary=False)
                params.add('center_%s' % 0, value=model[prop].mean, vary=False)
                params.add('amplitude_%s' % 0, value=np.nanmax(hist))
                params.add('sigma_%s' % 0, value=model[prop].std, vary=False)
                # fit amplitude to best match hist data
                fit = lmfit.minimize(multiG_fit, params, args=((bins[1:]+bins[:-1])/2, hist, [0]))
                # plot normal distribution
                x_temp = np.logspace(np.log10(x_min), np.log10(x_max), 100)
                curve = multiG_curve(fit.params, x_temp, [0])
                ax.plot(x_temp, curve, 'k')
                
            # add truth if available
            if np.isnan(model[prop].truth) == False:
                ax.axvline(model[prop].truth, c='k', linestyle='-', label="truth")
                if np.isnan(model[prop].error) == False:
                    text.append("")
                    text.append("err: %0.1f$\sigma$" % (np.abs(model[prop].error)/model[prop].std))
                ax.legend(loc='upper left')
            
            # add text to figure
            if show_text == True:
                ax.text(0.97, 0.97, "\n".join(text), ha='right', va='top', transform=ax.transAxes)
            
            # finish ax
            ax.set_xlim(x_min, x_max)
            
        # ==================================================
        # add covariance plots if needed

        if plot_covariance == True and hasattr(model, 'cov_matrix'):
            cov_matrix = model['cov_matrix']
            # get all possible pairs of parameters
            pairs = combinations(range(3), 2)
            count = 0
            for pair in pairs:
                # for each pair of parameters
                ax = axs.flat[5+count]
                xi, yi = pair
                x_prop = props[xi]
                y_prop = props[yi]
                
                # get Pearson correlation coefficient
                pcorr_coeff = cov_matrix[xi,yi]/(np.sqrt(cov_matrix[xi,xi]) * np.sqrt(cov_matrix[yi,yi]))
                
                # determine limits for axes
                x_min, x_max = x_lims[x_prop]
                y_min, y_max = x_lims[y_prop]
                
                # plot 2 parameters as 2D scatter
                ax.set_title("%s v %s" % (y_prop, x_prop))
                ax.scatter(model[x_prop].bootstrap, model[y_prop].bootstrap, c='steelblue', linewidth=0., alpha=float(n_bootstraps)**-0.25)
                ax.axvline(model[x_prop].mean, c='k', linestyle=':')
                ax.axhline(model[y_prop].mean, c='k', linestyle=':')
                ax.text(0.05, 0.95, "P=%0.2f" % pcorr_coeff, ha='left', va='top', transform=ax.transAxes)
                ax.set_xlabel(x_prop)
                ax.set_ylabel(y_prop)
                
                # finish ax
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                count += 1
                
            # plot limits of detection, quantitation in spare ax
            ax = axs.flat[4]
            
            ax.set_xlabel("Concentration Ratio")
            ax.set_ylabel("Intensity Ratio")
            neg_check = input_data['IR_av'] > 0
            x_min = 10**np.floor(np.log10(0.9*np.nanmin(input_data['CR'])))
            x_max = 10**np.ceil(np.log10(1.1*np.nanmax(input_data['CR'])))
            ax.set_xscale('log')
            ax.set_xlim(x_min, x_max)
            y_min = 10**np.floor(np.log10(0.9*np.nanmin(input_data['IR_av'][neg_check])))
            y_max = 10**np.ceil(np.log10(1.1*np.nanmax(input_data['IR_av'][neg_check])))
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)

            if len(input_data['CR']) > 0:
                # add input data with errorbars
                ax.errorbar(input_data['CR'], input_data['IR_av'], yerr=input_data['IR_std'], fmt='none', ecolor='k', capsize=3, zorder=6)
                ax.scatter(input_data['CR'], input_data['IR_av'], c='k', label='data', zorder=5)

            # add mean model curve, confidence interval band
            x_temp = np.logspace(np.log10(x_min), np.log10(x_max), 100)
            pred, upper_conf, lower_conf = predict_intensity(model, x_temp, percentiles=(2.5, 97.5), seed=seed, tabs=tabs+1, debug=False)
            ax.plot(x_temp, pred, 'r', label='mean model', zorder=4)
            ax.fill_between(x_temp, lower_conf, upper_conf, color='r', linewidth=0., alpha=0.2, label='95% CI')

            # plot limits of detection
            LL = 0
            UL = 0
            print("    LLoD: %0.1E" % LL)
            print("    ULoD: %0.1E" % UL)
            if LL > 0 and ~np.isinf(LL):
                ax.fill(
                        [x_min, x_min, LL, LL],
                        [y_min, intensity_curve(model, LL), intensity_curve(model, LL), y_min],
                        color='k', alpha=0.1, linewidth=0., hatch='////'
                )
            if UL > 0 and ~np.isinf(UL):
                ax.fill(
                        [x_min, x_min, x_max, x_max, UL, UL],
                        [intensity_curve(model, UL), y_max, y_max, y_min, y_min, intensity_curve(model, UL)],
                        color='k', alpha=0.1, linewidth=0., hatch='////'
                )
                    
            # plot limits of quantitation
            LL = 0
            UL = 0
            print("    LLoQ: %0.1E" % LL)
            print("    ULoQ: %0.1E" % UL)
            if LL > 0 and ~np.isinf(LL):
                ax.fill(
                        [x_min, x_min, LL, LL],
                        [y_min, intensity_curve(model, LL), intensity_curve(model, LL), y_min],
                        color='k', alpha=0.1, linewidth=0.,
                )
            if UL > 0 and ~np.isinf(UL):
                ax.fill(
                        [x_min, x_min, x_max, x_max, UL, UL],
                        [intensity_curve(model, UL), y_max, y_max, y_min, y_min, intensity_curve(model, UL)],
                        color='k', alpha=0.1, linewidth=0.,
                )
                
            # finish ax
            ax.legend(loc='upper left')
        
        # ==================================================
        # finish figure
        
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_model-summary.png" % (fig_dir, title), dpi=300)
            plt.savefig("%s%s_model-summary.svg" % (fig_dir, title), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
        if debug == True:
            print()
            print("    "*(tabs+1), "end of plot_quantmodel_summary() function")

# functions for predicting values using a trained model

def predict_intensity(model, CR_in, CR_std=0., percentiles=(16,84), seed=None, normal_samples=1000, return_all=False, clean_output=True, tabs=0, debug=False):
    # function for bootstrapped prediction of concentration ratio distribution for a given input intensity ratio
    # argument summary:
    #   - model:        QuantModel object containing all bootstrapped models
    #   - IR_in:        input intensity ratio (float, or 1D array of floats, or 2D array of floats if 2nd dim is)
    #   - IR_std:       input uncertainty in intensity ratio (float, or array of floats)
    #   - percentiles   percentiles to evaluate if return_all == False (tuple of 2 values between 1-99)
    #   - return_all    return entire distribution, or just summary values? (boolean)
    #   - rng_seed      seed for random sampling of normally distributed uncertainty (when IR_std > 0)
    # if return_all == True, this function returns the following:
    #   - if CR_in is a float, returns a 1D (N) array of predicted intensity ratios from N models
    #   - if CR_in is a 1D (M) array of M datapoints, returns a 2D (M,N) array of predicted intensity ratios
    #   - if CR_in is a 2D (M,P) array of P observations of M datapoints, returns a 2D (M,N*P) array of predicted intensity ratios
    # if return_all == False, this function returns the following:
    #   - if CR_in is a float, returns 3 single values for median, lower percentile, upper percentile
    #   - if CR_in is a 1D (M) array of M datapoints, returns 3 1D (M) arrays for median, lower, upper
    #   - if CR_in is a 2D (M,P) array of P observations of M datapoints, returns 3 1D (M) arrays for median, lower, upper
    
    # clean up input data, determine ndim
    if type(seed) == float:
        seed = int(seed)
    normal_samples = int(normal_samples)
    if len(percentiles) != 2:
        raise Exception("predict_concentration() must receive exactly two percentile values!")
    elif percentiles[0] > 100 or percentiles[1] > 100:
        raise Exception("predict_concentration() must receive percentile values between 0 and 100!")
    elif percentiles[0] < 0 or percentiles[1] < 0:
        raise Exception("predict_concentration() must receive percentile values between 0 and 100!")
    if isinstance(CR_in, (int, float, float, np.float64)):
        # received single scalar
        CR = np.asarray([CR_in], dtype=np.float64)
        ndim = 1
        ndshape = (1)
        ndsize = 1
    elif isinstance(CR_in, (list, np.ndarray)):
        # received list or N-dimensional array, convert and expand to ndim+1
        CR = np.asarray(CR_in, dtype=np.float64)
        ndim = CR.ndim
        ndshape = np.shape(CR)
        ndsize = np.size(CR, axis=0)
    else:
        raise Exception("predict_concentration() must receive a float or array of floats!")
        
    if debug == True:
        print("input CR:", np.shape(CR_in))
        print("clean CR:", np.shape(CR))
    
    # set up random number generator for uncertainty estimation
    rng = np.random.default_rng(seed)
    
    # handle IR uncertainty
    std_check = False
    if np.all(CR_std == None) or np.all(CR_std == 0):
        # no uncertainty, ignore
        pass
    elif isinstance(CR_std, (int, float, np.float64)):
        # received a single uncertainty value, generate normally distributed values
        CR = np.repeat(np.expand_dims(CR, axis=-1), normal_samples, axis=-1)
        CR = rng.normal(CR, CR_std, np.shape(CR))
        std_check = True
    elif isinstance(CR_std, (list, np.ndarray)):
        # received multiple uncertainty values, generate normally distributed values for each
        if np.shape(CR_std) == np.shape(CR):
            CR = np.repeat(np.expand_dims(CR, axis=-1), normal_samples, axis=-1)
            CR = rng.normal(CR, CR_std, np.shape(CR))
            std_check = True
        else:
            raise Exception("predict_concentration() IR_std array shape must match IR_in shape!")
    else:
        raise Exception("predict_concentration() IR_std argument must be None, float or array of floats!")
    
    # check for infinities
    inf_check = np.isinf(CR)
    
    # add extra dim to IR array for broadcasting with params
    CR = np.expand_dims(CR, axis=-1)
    
    if debug == True:
        print("final CR:", np.shape(CR))
        print("    median: %0.1E" % np.nanmedian(CR))
    
    # generate bootstrapped parameter matrix with extra dims to broadcast with IR
    if ndim == 0:
        params = {prop: model[prop].bootstrap for prop in ['FXJ', 'FXB', 'JA']}
    else:
        params = {prop: np.expand_dims(model[prop].bootstrap, axis=tuple(range(CR.ndim-1))) for prop in ['FXJ', 'FXB', 'JA']}
    
    # predict concentration ratio for each input IR (with uncertainty) for each model
    IR = (params["FXJ"] * CR + params["FXB"]) / (params["JA"] * CR + 1)
    
    # handle infinities
    if np.any(inf_check) == True:
        IR[inf_check] = model['FXJ'].bootstrap / model['JA'].bootstrap
    
    if debug == True:
        print("  params:", np.shape(params['FXJ']))
        print("pred. IR:", np.shape(IR))
    
    
    if IR.ndim > 2:
        # flatten uncertainty distributions together
        IR = np.reshape(IR, (ndsize, -1))
        
    if debug == True:
        print("final IR:", np.shape(IR))
        print("nan check: %3.2f%%" % (100.*np.count_nonzero(np.isnan(IR))/np.size(IR)))
        print("inf check: %3.2f%%" % (100.*np.count_nonzero(np.isinf(IR))/np.size(IR)))
        print(" <0 check: %3.2f%%" % (100.*np.count_nonzero(IR < 0)/np.size(IR)))
        
    # clean up output
    if clean_output == True:
        check = np.logical_or(IR < 0, np.isinf(IR))
        IR[check] = np.nan
        
    if return_all == True:
        # return predicted CRs
        return IR
    else:
        # return only median, low and high percentiles for each input IR
        median = np.nanmedian(IR, axis=-1)
        pc_low = np.nanpercentile(IR, percentiles[0], axis=-1)
        pc_high = np.nanpercentile(IR, percentiles[1], axis=-1)
        return median, pc_low, pc_high

def predict_concentration(model, IR_in, IR_std=0., percentiles=(16,84), seed=None, normal_samples=1000, return_all=False, clean_output=True, tabs=0, debug=False):
    # function for bootstrapped prediction of concentration ratio distribution for a given input intensity ratio
    # argument summary:
    #   - model:            QuantModel object containing all bootstrapped models
    #   - IR_in:            input intensity ratio (float, or array of floats)
    #   - IR_std:           input uncertainty in intensity ratio (float, or array of floats)
    #   - percentiles       percentiles to evaluate if return_all == False (tuple of 2 values between 1-99)
    #   - seed              seed for random sampling of normally distributed uncertainty (when IR_std > 0)
    #   - normal_samples    number of samples to draw from normal distribution when approximating noisy data (int)
    #   - return_all        if True return entire distribution, if False return just summary values (boolean)
    #   - tabs:             number of tabs to put before debug messages (str)
    #   - debug:            if True print all debug messages in viewer (boolean)
    # if return_all == True, this function returns the following:
    #   - if IR_in is a float, returns a 1D (N) array of predicted concentration ratios from N models
    #   - if IR_in is a 1D array, returns a 2D (M,N) array of concentration ratios for N models of M data points
    #   - if IR_in is a 2D array, returns a 2D (M,N*P) array of concentration ratios for N models of M data points (with P observations)
    # if return_all == False, this function returns the following:
    #   - if IR_in is a float, returns 3 single values for median, lower percentile, upper percentile of predicted concentration ratios
    #   - if IR_in is a 1D array, returns 3 1D (M) arrays for median, lower percentile, upper percentile for M data points
    #   - if IR_in is a 2D array, returns 3 1D (M) arrays for median, lower percentile, upper percentile for M data points
    
    # clean up input data, determine ndim
    if type(seed) == float:
        seed = int(seed)
    normal_samples = int(normal_samples)
    if len(percentiles) != 2:
        raise Exception("predict_concentration() must receive exactly two percentile values!")
    elif percentiles[0] > 100 or percentiles[1] > 100:
        raise Exception("predict_concentration() must receive percentile values between 0 and 100!")
    elif percentiles[0] < 0 or percentiles[1] < 0:
        raise Exception("predict_concentration() must receive percentile values between 0 and 100!")
    if isinstance(IR_in, (int, float, float, np.float64)):
        # received single scalar
        IR = np.asarray([IR_in], dtype=np.float64)
        ndim = 1
        ndshape = (1)
        ndsize = 1
    elif isinstance(IR_in, (list, np.ndarray)):
        # received list or N-dimensional array, convert and expand to ndim+1
        IR = np.asarray(IR_in, dtype=np.float64)
        ndim = IR.ndim
        ndshape = np.shape(IR)
        ndsize = np.size(IR, axis=0)
    else:
        raise Exception("predict_concentration() must receive a float or array of floats!")
    
    # set up random number generator for uncertainty estimation
    rng = np.random.default_rng(seed)
    
    # handle IR uncertainty
    std_check = False
    if np.all(IR_std == None) or np.all(IR_std == 0):
        # no uncertainty, ignore
        pass
    elif isinstance(IR_std, (int, float, np.float64)):
        # received a single uncertainty value, generate normally distributed values
        IR = np.repeat(np.expand_dims(IR, axis=-1), normal_samples, axis=-1)
        IR = rng.normal(IR, IR_std, np.shape(IR))
        std_check = True
    elif isinstance(IR_std, (list, np.ndarray)):
        # received multiple uncertainty values, generate normally distributed values for each
        if np.shape(IR_std) == np.shape(IR):
            IR = np.repeat(np.expand_dims(IR, axis=-1), normal_samples, axis=-1)
            IR = rng.normal(IR, IR_std, np.shape(IR))
            std_check = True
        else:
            raise Exception("predict_concentration() IR_std array shape must match IR_in shape!")
    else:
        raise Exception("predict_concentration() IR_std argument must be None, float or array of floats!")
    
    # add extra dim to IR array for broadcasting with params
    IR = np.expand_dims(IR, axis=-1)
    
    # generate bootstrapped parameter matrix with extra dims to broadcast with IR
    if ndim == 0:
        params = {prop: model[prop].bootstrap for prop in ['FXJ', 'FXB', 'JA']}
    else:
        params = {prop: np.expand_dims(model[prop].bootstrap, axis=tuple(range(IR.ndim-1))) for prop in ['FXJ', 'FXB', 'JA']}
    
    # predict concentration ratio for each input IR (with uncertainty) for each model
    CR = (IR - params["FXB"]) / (params["FXJ"] - params["JA"] * IR)
    
    if CR.ndim > 2:
        # flatten uncertainty distributions together
        CR = np.reshape(CR, (ndsize, -1))
        
    if debug == True:
        print("final CR:", np.shape(CR))
        print("nan check: %3.2f%%" % (100.*np.count_nonzero(np.isnan(CR))/np.size(CR)))
        print("inf check: %3.2f%%" % (100.*np.count_nonzero(np.isinf(CR))/np.size(CR)))
        print(" <0 check: %3.2f%%" % (100.*np.count_nonzero(CR < 0)/np.size(CR)))
        
    # clean up output
    if clean_output == True:
        check = np.logical_or(CR < 0, np.isinf(CR))
        CR[check] = np.nan
        
    if return_all == True:
        # return predicted CRs
        return CR
    else:
        # return only median, low and high percentiles for each input IR
        median = np.nanmedian(CR, axis=-1)
        pc_low = np.nanpercentile(CR, percentiles[0], axis=-1)
        pc_high = np.nanpercentile(CR, percentiles[1], axis=-1)
        return median, pc_low, pc_high
    
def predict_intensity_noise(model, CR, CR_std=0., I1=0., I2=0., noise=0., percentiles=(16,84), n_samples=1000, seed=None, return_all=False):
    # function for adding noise to an intensity prediction from the model, using a reference peak intensity and known background noise
    rng = np.random.default_rng(seed)
    
    # predict standard IR for CR, CR_std using model
    IR = predict_intensity(model, CR, CR_std, return_all=True)
    ndshape = np.shape(IR)
    
    if I1 > 0 and noise > 0:
        # add noise based on relative intensity of I1
        temp_I2 = rng.normal(I1 / IR[...,np.newaxis], noise, ndshape+(n_samples,))
        temp_I1 = rng.normal(I1, noise, ndshape+(1000,))
        IR = np.reshape(temp_I1 / temp_I2, (np.size(CR), -1))
    elif I2 > 0 and noise > 0:
        # add noise based on relative intensity of I2
        temp_I1 = rng.normal(I2 * IR[...,np.newaxis], noise, ndshape+(n_samples,))
        temp_I2 = rng.normal(I2, noise, ndshape+(1000,))
        IR = np.reshape(temp_I1 / temp_I2, (np.size(CR), -1))
    
    if return_all == True:
        return IR
    else:
        if isinstance(CR, (int, float, np.float64)) == True:
            # return scalars, not single-element arrays
            median = np.nanmedian(IR)
            pc_low = np.nanpercentile(IR, percentiles[0])
            pc_high = np.nanpercentile(IR, percentiles[1])
        else:
            median = np.nanmedian(IR, axis=-1)
            pc_low = np.nanpercentile(IR, percentiles[0], axis=-1)
            pc_high = np.nanpercentile(IR, percentiles[1], axis=-1)
        return median, pc_low, pc_high

def get_detection_limit(model, limit='lower', I1=0., I2=0., noise=0., threshold=1, seed=None, return_all=False, show_plot=False, save_plot=False, title="", fig_dir="./figures/", tabs=0, debug=False):
    # function for determining the limit of detection (upper or lower) for a given model, evaluated for both intrinsic uncertainty (model) and measurement error (noise)
    # argument summary:
    #   - model:            QuantModel object containing all bootstrapped models
    #   - limit:            which limit to determine, either 'lower' or 'upper' (str)
    #   - I1:               intensity of peak I1, required to evaluate upper noise limit (float)
    #   - I2:               intensity of peak I2, required to evaluate lower noise limit (float)
    #   - noise:            st.dev. of background noise, required to evaluate either noise limit (float)
    #   - threshold:        percentile to evaluate
    #   - return_all:       if True return both limits, if False return most conservative limit (boolean)
    #   - show_plot:        if True the summary figure will be shown in viewer (boolean)
    #   - save_plot:        if True the summary figure will be saved to file (boolean)
    #   - title:            figure title to put at beginning of filename (str)
    #   - fig_dir:          directory to save figure file in (str)
    #   - tabs:             number of tabs to put before debug messages (str)
    #   - debug:            if True print all debug messages in viewer (boolean)
    
    
    # clean up input
    limit = limit.lower()
    if limit not in ['upper', 'lower']:
        raise Exception("get_detection_limit() limit argument must be either 'upper' or 'lower'!")
        
    # set up random number generator
    rng = np.random.default_rng(seed)
    
    # determine basic IR values
    if limit == 'upper':
        percentiles = (threshold, 100-threshold)
        blank_CR = np.inf
        blank_model_median, blank_model_lower, blank_model_upper = predict_intensity(model, blank_CR, percentiles=percentiles, return_all=False)
        model_target = blank_model_lower
        # handle noise
        temp_I1 = I1
        temp_I2 = 0.
        if I1 > 0 and noise > 0:
            blank_noise_median, blank_noise_lower, blank_noise_upper = predict_intensity_noise(model, blank_CR, I1=temp_I1, I2=temp_I2, noise=noise, seed=seed, percentiles=percentiles, return_all=False)
            noise_target = blank_noise_lower
        else:
            noise_target = np.inf
    else:
        percentiles = (threshold, 100-threshold)
        blank_CR = 0.
        blank_model_median, blank_model_lower, blank_model_upper = predict_intensity(model, blank_CR, percentiles=percentiles, return_all=False)
        model_target = blank_model_upper
        temp_I1 = 0.
        temp_I2 = I2
        if I2 > 0 and noise > 0:
            blank_noise_median, blank_noise_lower, blank_noise_upper = predict_intensity_noise(model, blank_CR, I1=temp_I1, I2=temp_I2, noise=noise, seed=seed, percentiles=percentiles, return_all=False)
            noise_target = blank_noise_upper
        else:
            noise_target = 0.
        
    if isinstance(model_target, np.ndarray):
        model_target = model_target[0]
    if isinstance(noise_target, np.ndarray):
        noise_target = noise_target[0]
                
    if debug == True:
        print()
        print("    "*(tabs), "getting P=%s thresholds..." % threshold)
        print("    "*(tabs), "Estimated %s IR thresholds:" % limit.title())
        print("    "*(tabs+1), "model: %0.1E" % model_target)
        print("    "*(tabs+1), "noise: %0.1E" % noise_target)
            
    # for model_target, evaluate model-only distributions and get CR where median IR equals target
    if model_target == 0.:
        LD_model = np.nan
    else:
        CR_target = concentration_curve(model, model_target)
        CR_min = 10**np.floor(np.log10(0.5*CR_target))
        CR_max = 10**np.ceil(np.log10(2*CR_target))
        model_CR_range = np.logspace(np.log10(CR_min), np.log10(CR_max), 100)
        model_medians, model_lower, model_upper = predict_intensity(model, model_CR_range, percentiles=percentiles, return_all=False)
        LD_model = np.interp(model_target, model_medians, model_CR_range)
        
    if debug == True:
        print("    "*(tabs), "evaluating model limit...")
        print("    "*(tabs+1), "CR range: %0.1E - %0.1E" % (CR_min, CR_max))
        print("    "*(tabs+1), "IR range: %0.1E - %0.1E" % (model_medians[0], model_medians[-1]))
        print("    "*(tabs+1), "   model: %0.1E" % LD_model)
        if LD_model <= CR_min or LD_model >= CR_max:
            print("    "*(tabs+2), "caution! predicted model limit is outside evaluated range!")
    
    # for noise_target, evaluate model+noise distributions and get CR where median IR equals target
    if noise_target == 0.:
        LD_noise = np.nan
    else:
        CR_target = concentration_curve(model, noise_target)
        CR_min = 10**np.floor(np.log10(0.5*CR_target))
        CR_max = 10**np.ceil(np.log10(2*CR_target))
        noise_CR_range = np.logspace(np.log10(CR_min), np.log10(CR_max), 100)
        noise_medians, noise_lower, noise_upper = predict_intensity_noise(model, noise_CR_range, I1=temp_I1, I2=temp_I2, noise=noise, seed=seed, percentiles=percentiles, return_all=False)
        LD_noise = np.interp(noise_target, noise_medians, noise_CR_range)
    
    if debug == True:
        print("    "*(tabs), "evaluating noise limit...")
        print("    "*(tabs+1), "CR range: %0.1E - %0.1E" % (CR_min, CR_max))
        print("    "*(tabs+1), "IR range: %0.1E - %0.1E" % (noise_medians[0], noise_medians[-1]))
        print("    "*(tabs+1), "   noise: %0.1E" % LD_noise)
        if LD_noise <= CR_min or LD_noise >= CR_max:
            print("    "*(tabs+2), "caution! predicted noise limit is outside evaluated range!")
    
    # clean up output
    if LD_model < 0:
        LD_model = np.nan
    if LD_noise < 0:
        LD_noise = np.nan
    
    if limit == 'upper':
        LD_overall = np.nanmin([LD_model, LD_noise])
    else:
        LD_overall = np.nanmax([LD_model, LD_noise])
    
    # plot results if required
    if show_plot == True or save_plot == True:
        # set up figure
        plt.figure(figsize=(6,6))
        ax1 = plt.subplot(111)
        ax1.set_title("%s Detect. Limits" % (limit.title()))
        
        # plot FXB or FXJ/JA
        ax1.axhline(blank_model_median, c='k', linestyle='-', label='limit')
        ax1.axhspan(blank_model_lower, blank_model_upper, facecolor='k', edgecolor='k', linewidth=0., alpha=0.2, hatch='////')
        if noise > 0:
            ax1.axhspan(blank_noise_lower, blank_noise_upper, facecolor='k', linewidth=0., alpha=0.2)
        
        if debug == True:
            print("    "*(tabs), "CR range for plotting: %0.1E - %0.1E" % (CR_min, CR_max))
        
        # plot model-only data
        ax1.plot(model_CR_range, model_medians, c='r', label='model')
        ax1.fill_between(model_CR_range, model_lower, model_upper, facecolor='r', edgecolor='r', linewidth=0., alpha=0.2, hatch='////', label='model %s-%s%%' % (threshold, 100-threshold))
        
        # plot model-driven limits
        ax1.axhline(model_target, c='r', linestyle=':', label='LD (model)')
        ax1.axvline(LD_model, c='r', linestyle=':')
        
        if np.isnan(LD_noise) == False:
            # plot model+noise data
            ax1.plot(noise_CR_range, noise_medians, c='b', label='noise')
            ax1.fill_between(noise_CR_range, noise_lower, noise_upper, facecolor='b', linewidth=0., alpha=0.2, label='noise %s-%s%%' % (threshold, 100-threshold))
            
            # plot noise-driven limits
            ax1.axhline(noise_target, c='b', linestyle=':', label='LD (noise)')
            ax1.axvline(LD_noise, c='b', linestyle=':')
        
        # finish figure
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Theor. CR")
        ax1.set_ylabel("Theor. IR")
        if limit == 'upper':
            ax1.legend(loc='lower right')
        else:
            ax1.legend(loc='upper left')
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_%s-detection-limits.png" % (fig_dir, title, limit), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
    
    # return results
    if return_all == True:
        return {'model': LD_model, 'noise': LD_noise, 'overall': LD_overall}
    else:
        return LD_overall
    
def get_quantification_limit(model, limit='lower', I1=0., I2=0., noise=0., threshold=1, seed=None, return_all=False, show_plot=False, save_plot=False, title="", fig_dir="./figures/", tabs=0, debug=False):
    # function for determining the limit of quantification (upper or lower) for a given model, evaluated for both intrinsic uncertainty (model) and measurement error (noise)
    # argument summary:
    #   - model:            QuantModel object containing all bootstrapped models
    #   - limit:            which limit to determine, either 'lower' or 'upper' (str)
    #   - I1:               intensity of peak I1, required to evaluate upper noise limit (float)
    #   - I2:               intensity of peak I2, required to evaluate lower noise limit (float)
    #   - noise:            st.dev. of background noise, required to evaluate either noise limit (float)
    #   - threshold:        percentile to evaluate
    #   - return_all:       if True return both limits, if False return most conservative limit (boolean)
    #   - show_plot:        if True the summary figure will be shown in viewer (boolean)
    #   - save_plot:        if True the summary figure will be saved to file (boolean)
    #   - title:            figure title to put at beginning of filename (str)
    #   - fig_dir:          directory to save figure file in (str)
    #   - tabs:             number of tabs to put before debug messages (str)
    #   - debug:            if True print all debug messages in viewer (boolean)
    
    
    # clean up input
    limit = limit.lower()
    if limit not in ['upper', 'lower']:
        raise Exception("get_quantification_limit() limit argument must be either 'upper' or 'lower'!")
        
    # set up random number generator
    rng = np.random.default_rng(seed)
    
    # determine basic IR values
    if limit == 'upper':
        percentiles = (threshold, 100-threshold)
        blank_CR = np.inf
        blank_model_median, blank_model_lower, blank_model_upper = predict_intensity(model, blank_CR, percentiles=percentiles, return_all=False)
        model_target = blank_model_lower
        # handle noise
        temp_I1 = I1
        temp_I2 = 0.
        if I1 > 0 and noise > 0:
            blank_noise_median, blank_noise_lower, blank_noise_upper = predict_intensity_noise(model, blank_CR, I1=temp_I1, I2=temp_I2, noise=noise, seed=seed, percentiles=percentiles, return_all=False)
            noise_target = blank_noise_lower
        else:
            noise_target = np.inf
    else:
        percentiles = (threshold, 100-threshold)
        blank_CR = 0.
        blank_model_median, blank_model_lower, blank_model_upper = predict_intensity(model, blank_CR, percentiles=percentiles, return_all=False)
        model_target = blank_model_upper
        temp_I1 = 0.
        temp_I2 = I2
        if I2 > 0 and noise > 0:
            blank_noise_median, blank_noise_lower, blank_noise_upper = predict_intensity_noise(model, blank_CR, I1=temp_I1, I2=temp_I2, noise=noise, seed=seed, percentiles=percentiles, return_all=False)
            noise_target = blank_noise_upper
        else:
            noise_target = 0.
        
    if isinstance(model_target, np.ndarray):
        model_target = model_target[0]
    if isinstance(noise_target, np.ndarray):
        noise_target = noise_target[0]
                
    if debug == True:
        print()
        print("    "*(tabs), "getting P=%s thresholds..." % threshold)
        print("    "*(tabs), "Estimated %s IR thresholds:" % limit.title())
        print("    "*(tabs+1), "model: %0.1E" % model_target)
        print("    "*(tabs+1), "noise: %0.1E" % noise_target)
            
    # for model_target, evaluate model-only distributions and get CR where median IR equals target
    if model_target == 0.:
        LD_model = np.nan
    else:
        CR_target = concentration_curve(model, model_target)
        CR_min = 10**np.floor(np.log10(0.25*CR_target))
        CR_max = 10**np.ceil(np.log10(4*CR_target))
        model_CR_range = np.logspace(np.log10(CR_min), np.log10(CR_max), 100)
        model_medians, model_lower, model_upper = predict_intensity(model, model_CR_range, percentiles=percentiles, return_all=False)
        if limit == 'upper':
            LD_model = np.interp(model_target, model_upper, model_CR_range)
        else:
            LD_model = np.interp(model_target, model_lower, model_CR_range)
        
    if debug == True:
        print("    "*(tabs), "evaluating model limit...")
        if np.isnan(LD_model):
            print("    "*(tabs+1), "   model: %0.1E" % LD_model)
        else:
            print("    "*(tabs+1), "CR range: %0.1E - %0.1E" % (CR_min, CR_max))
            print("    "*(tabs+1), "IR range: %0.1E - %0.1E" % (model_medians[0], model_medians[-1]))
            print("    "*(tabs+1), "   model: %0.1E" % LD_model)
    if np.isnan(LD_model) == False:
        if LD_model <= CR_min or LD_model >= CR_max:
            raise Exception("caution! predicted model limit is outside evaluated range!")
    
    # for noise_target, evaluate model+noise distributions and get CR where median IR equals target
    if noise_target == 0.:
        LD_noise = np.nan
    else:
        CR_target = concentration_curve(model, noise_target)
        CR_min = 10**np.floor(np.log10(0.25*CR_target))
        CR_max = 10**np.ceil(np.log10(4*CR_target))
        noise_CR_range = np.logspace(np.log10(CR_min), np.log10(CR_max), 100)
        noise_medians, noise_lower, noise_upper = predict_intensity_noise(model, noise_CR_range, I1=temp_I1, I2=temp_I2, noise=noise, seed=seed, percentiles=percentiles, return_all=False)
        if limit == 'upper':
            LD_noise = np.interp(noise_target, noise_upper, noise_CR_range)
        else:
            LD_noise = np.interp(noise_target, noise_lower, noise_CR_range)
    
    if debug == True:
        print("    "*(tabs), "evaluating noise limit...")
        if np.isnan(LD_noise):
            print("    "*(tabs+1), "   noise: %0.1E" % LD_noise)
        else:
            print("    "*(tabs+1), "CR range: %0.1E - %0.1E" % (CR_min, CR_max))
            print("    "*(tabs+1), "IR range: %0.1E - %0.1E" % (noise_medians[0], noise_medians[-1]))
            print("    "*(tabs+1), "   noise: %0.1E" % LD_noise)
    if np.isnan(LD_noise) == False:
        if LD_noise <= CR_min or LD_noise >= CR_max:
            raise Exception("caution! predicted noise limit is outside evaluated range!")
    
    # clean up output
    if LD_model < 0:
        LD_model = np.nan
    if LD_noise < 0:
        LD_noise = np.nan
    
    if limit == 'upper':
        LD_overall = np.nanmin([LD_model, LD_noise])
    else:
        LD_overall = np.nanmax([LD_model, LD_noise])
    
    # plot results if required
    if show_plot == True or save_plot == True:
        # set up figure
        plt.figure(figsize=(6,6))
        ax1 = plt.subplot(111)
        ax1.set_title("%s Quant. Limits" % (limit.title()))
        
        # plot FXB or FXJ/JA
        ax1.axhline(blank_model_median, c='k', linestyle='-', label='limit')
        ax1.axhspan(blank_model_lower, blank_model_upper, facecolor='k', edgecolor='k', linewidth=0., alpha=0.2, hatch='////')
        if noise > 0:
            ax1.axhspan(blank_noise_lower, blank_noise_upper, facecolor='k', linewidth=0., alpha=0.2)
        
        if debug == True:
            print("    "*(tabs), "CR range for plotting: %0.1E - %0.1E" % (CR_min, CR_max))
        
        # plot model-only data
        ax1.plot(model_CR_range, model_medians, c='r', label='model')
        ax1.fill_between(model_CR_range, model_lower, model_upper, facecolor='r', edgecolor='r', linewidth=0., alpha=0.2, hatch='////', label='model %s-%s%%' % (threshold, 100-threshold))
        
        # plot model-driven limits
        ax1.axhline(model_target, c='r', linestyle=':', label='LD (model)')
        ax1.axvline(LD_model, c='r', linestyle=':')
        
        if np.isnan(LD_noise) == False:
            # plot model+noise data
            ax1.plot(noise_CR_range, noise_medians, c='b', label='noise')
            ax1.fill_between(noise_CR_range, noise_lower, noise_upper, facecolor='b', linewidth=0., alpha=0.2, label='noise %s-%s%%' % (threshold, 100-threshold))
            
            # plot noise-driven limits
            ax1.axhline(noise_target, c='b', linestyle=':', label='LD (noise)')
            ax1.axvline(LD_noise, c='b', linestyle=':')
        
        # finish figure
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Theor. CR")
        ax1.set_ylabel("Theor. IR")
        if limit == 'upper':
            ax1.legend(loc='lower right')
        else:
            ax1.legend(loc='upper left')
        plt.tight_layout()
        if save_plot == True:
            plt.savefig("%s%s_%s-quantification-limits.png" % (fig_dir, title, limit), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
    
    # return results
    if return_all == True:
        return {'model': LD_model, 'noise': LD_noise, 'overall': LD_overall}
    else:
        return LD_overall
    
    
    
    
    
    
    

def detection_limit(model, noise=0., I1=None, I2=None, threshold=5., limit='lower', return_all=False, debug=False):
    # function for estimating the limits of detection for a given measurement and model
    
    # calculate IR from noise level
    I_noise = float(threshold) * noise
    # calculate IR from appropriate model asymptote
    if limit.lower() in ['low', 'lower']:
        # calculating lower limit from I2
        mode = 'lower'
        # calculate IR from model asymptote
        IR_model = model['FXB'].value + float(threshold) * model['FXB'].std
        if np.all(I2 == None):
            I2 = 0.
            IR_noise = 0.
        else:
            IR_noise = I_noise / I2
    elif limit.lower() in ['up', 'upper']:
        # calculating upper limit from I1
        mode = 'upper'
        # calculate IR from model asymptote
        IR_model = model['FXJ_JA'].value - float(threshold) * model['FXJ_JA'].std
        if np.all(I1 == None):
            I1 = 0.
            IR_noise = 0.
        else:
            IR_noise = I1 / I_noise
    else:
        raise Exception("limit argument passed to detection_limits() function must be 'upper' or 'lower' str!")
        
    if debug == True:
        print("IR noise limit: %0.1E" % IR_noise)
        print("IR model limit: %0.1E" % IR_model)
        
    # calculate concentrations from intensity ratios
    CR_noise = concentration_curve(model, IR_noise, debug=debug)
    CR_model = concentration_curve(model, IR_model, debug=debug)
    
    if debug == True:
        print("pred CR noise limit: %0.1E" % CR_noise)
        print("pred CR model limit: %0.1E" % CR_model)
    
    # mask nonsensical values
    if CR_noise <= 0 or np.isnan(CR_noise) == True or np.isinf(CR_noise) == True:
        if mode == 'lower':
            CR_noise = 0.
        else:
            CR_noise = np.inf
    if CR_model <= 0 or np.isnan(CR_model) == True or np.isinf(CR_model) == True:
        if mode == 'lower':
            CR_model = 0.
        else:
            CR_model = np.inf
        
    temp = np.asarray([CR_noise, CR_model])
    # calculate most conservative limit
    if mode == 'lower':
        i = np.nanargmax(temp)
    else:
        i = np.nanargmin(temp)
    CR_limit = temp[i]
    
    # return values
    if return_all == True:
        return {'noise': CR_noise, 'model': CR_model, 'limit': CR_limit}
    else:
        return CR_limit

def confidence_band(model, CR, CR_err=0., alpha=0.05, seed=None):
    # function for calculating simultaneous confidence bands
    # code based on Confidence- and prediction intervals example from https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html
    
    # clean up input
    if isinstance(CR_err, np.ndarray):
        if np.shape(CR_err) != np.shape(CR):
            CR_err = np.zeros_like(CR)
    else:
        CR_err = np.zeros_like(CR)
    
    # determine scale for interval
    prb = 1. - alpha/2              # critical probability (for alpha=0.05: 0.975)
    dof = model.n_bootstraps - 3     # degrees of freedom: size of training data - number of parameters
    tval = stats.t.ppf(prb, dof)    # critical factor
    
    # get mean predictions, uncertainties from model
    y, std = predict_intensity(model, CR, CR_err)
    
    # get confidence intervals
    lower = y - tval * std
    upper = y + tval * std
        
    # return mean, and upper and lower bounds of band
    return y, lower, upper

"""
# ==================================================
# functions for saving measurements to file
# ==================================================
"""

def save_measurement(measurement, keys=[], headers=[], start=None, end=None, do_not_include=[], out_dir=None, file_name=None, tabs=0, metadata=True, save_plot=False, show_plot=False, debug=False):
    # script for saving a given set of spectra from a measurement
    # argument summary:
    #   - measurement:      Measurement instance containing data
    #   - *args:            series of spectrum keys to add to save file (strings)
    #   - save_name:        suffix to use when naming figure file (string)
    global Spectrum
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(tabs) + "saving measurement %s to file" % measurement.title)
        print("    "*(tabs+1) + "spectra to include:", ", ".join(keys))
    check = [hasattr(measurement, key) for key in keys]
    if np.any(check) == False:
        raise Exception("    "*(tabs+1) + "the following keys are missing from measurement:", np.asarray(keys)[check])
    if len(keys) != len(headers):
        raise Exception("    "*(tabs+1) + "header list passed to save_measurement() does not match key list length!")
    if out_dir == None:
        out_dir = measurement.out_dir
    if file_name == None:
        file_name = "%s_av-spectrum" % measurement.title
    if debug == True:
        print("    "*(tabs+1) + "file suffix: %s.csv" % save_name)
        
    # ==================================================
    # check if plotting is required
    plot = False
    if save_plot == True or show_plot == True:
        plot = True
    
    # ==================================================
    # gather spectral data
    header_info = []
    save_data = []
    for key, header in zip(keys, headers):
        # for each key in list, get array from matching property
        if key in ['x', 'raman_shift', 'wavelength', 'frequency']:
            # ==================================================
            # desired array is a direct property of Measurement object
            header_info.append(header)
            save_data.append(measurement[key][:,np.newaxis])
            if debug == True:
                print("    "*(tabs+2), header, np.shape(save_data[-1]))
        elif key[-5:] == '_norm' and hasattr(measurement[key[:-5]], 'y') == True:
            # ==================================================
            # desired array is a normalised Spectrum object
            x, y = measurement('x', key[:-5], normalise=True)
            indices = measurement[key[:-5]].indices
            if np.shape(y)[1] > 1:
                # more than 1 spectrum, each needs its own header
                for i in range(np.shape(y)[1]):
                    header_info.append("%s spec %s" % (header, indices[i]))
            else:
                # single spectrum, single header
                header_info.append(header)
            save_data.append(y)
            if debug == True:
                print("    "*(tabs+2), header, np.shape(save_data[-1]))
        elif hasattr(measurement[key], 'y') == True:
            # ==================================================
            # desired array is inside a Spectrum object within Measurement
            x,y = measurement('x', key)
            indices = measurement[key].indices
            if np.shape(y)[1] > 1:
                # more than 1 spectrum, each needs its own header
                for i in range(np.shape(y)[1]):
                    header_info.append("%s spec %s" % (header, indices[i]))
            else:
                # single spectrum, single header
                header_info.append(header)
            save_data.append(y)
            if debug == True:
                print("    "*(tabs+2), header, np.shape(save_data[-1]))
        else:
            # ==================================================
            # key is not a valid data type, skip
            if debug == True:
                print("    "*(tabs+2) + "%s is not a valid key for measurement %s" % (key, measurement.ID))
            pass
        
    if plot == True:
        plt.figure(figsize=(8,4))
        plt.title("Spectrum saved to file\n%s" % measurement.title)
        plt.xlabel(headers[0])
        plt.ylabel(headers[-1])
        plt.plot(save_data[0], save_data[-1])
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        
    # ==================================================
    # generate output folder if missing
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # ==================================================
    # convert save_data to pandas DataFrame and save to CSV
    save_data = pd.DataFrame(np.hstack(save_data), columns=header_info)
    if debug == True:
        print("    "*(tabs) + "resulting dataframe:")
        print(save_data.info())
    save_data.to_csv("%s%s.csv" % (out_dir, file_name), index=False)
    
    # ==================================================
    # generate metadata frame and save to CSV
    if metadata == True:
        if debug == True:
            print()
            print("    "*(tabs) + "generating metadata file for %s" % measurement.title)
        headers = []
        values = []
        # convert measurement info to metadata
        for header in ['ID', 'title', 'sample', 'subsample', 'notes', 'measurement_date', 'filename', 'technique']:
            if hasattr(measurement, header):
                if isinstance(measurement[header], (float, int, str)) == True and header not in headers and header not in do_not_include:
                    # single value of suitable data type, add to array
                    headers.append(header)
                    values.append(measurement[header])
                    if debug == True:
                        print("    "*(tabs+1), header, "=", measurement[header])
                else:
                    # all other data types get skipped
                    if debug == True:
                        print("    "*(tabs+1), header, "skipping %s" % type(measurement[header]))
        for header in measurement.__dict__.keys():
            # for all other attributes of measurement
            if isinstance(measurement[header], (float, int, str)) == True and header not in headers and header not in do_not_include:
                # single value of suitable data type, add to array
                headers.append(header)
                values.append(measurement[header])
                if debug == True:
                    print("    "*(tabs+1), header, "=", measurement[header])
            else:
                # all other data types get skipped
                if debug == True:
                    print("    "*(tabs+1), header, "skipping %s" % type(measurement[header]))
                
        metadata = np.asarray([headers, values])
        if debug == True:
            print(np.shape(metadata))
        metadata = pd.DataFrame(metadata.transpose(), columns=['Parameter', 'Value'])
        if debug == True:
            print("    "*(tabs) + "resulting dataframe:")
            print(metadata.info())
        metadata.to_csv("%s%s_metadata.csv" % (out_dir, file_name), index=False)
    
    # ==================================================
    # end of function
    if debug == True:
        print("    end of save_measurement()")
    return out_dir