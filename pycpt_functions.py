import os
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from cartopy import feature
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def download_data(url, authkey, outfile, force_download=False):
    """A smart function to download data from IRI Data Library
    If the data can be read in and force_download is False, will read from file
    Otherwise will download from IRIDL and then read from file
    
    PARAMETERS
    ----------
        url: the url pointing to the data.nc file
        authkey: the authentication key for IRI DL (see above)
        outfile: the data filename
        force_download: False if it's OK to read from file, True if data *must* be re-downloaded
    """
    
    if not force_download:
        try:
            model = xr.open_dataset(outfile, decode_times=False)
        except:
            force_download = True
        
    if force_download:
        # calls curl to download data
        command = "curl -C - -k -b '__dlauth_id={}' '{}' > {}".format(authkey, url, outfile)
        get_ipython().system(command)
        # open the data
        model = xr.open_dataset(outfile, decode_times=False)
        
    return model

def pltdomain(loni,lone,lati,late,title):
    """A simple plot function for the geographical domain
    
    PARAMETERS
    ----------
        loni: wester longitude
        lone: eastern longitude
        lati: southern latitude
        late: northern latitude
        title: title
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([loni,lone,lati,late])

    # Put a background image on for nice sea rendering.
    ax.stock_img()

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = feature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50ml',
        facecolor='none')

    ax.add_feature(feature.LAND)
    ax.add_feature(feature.COASTLINE)
    #ax.add_feature(states_provinces, edgecolor='gray')
    ax.set_title(title)
    pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    pl.xlabels_top = False
    pl.ylabels_left = False
    pl.xformatter = LONGITUDE_FORMATTER
    pl.yformatter = LATITUDE_FORMATTER

    plt.show()

def pltscore(score,loni,lone,lati,late):
    """A simple plot function for ploting the statistical score
    
    PARAMETERS
    ----------
        score: the score
        loni: wester longitude
        lone: eastern longitude
        lati: southern latitude
        late: northern latitude
        title: title
    """
    H = 9
    W = 5
    Record = np.dtype(('float32', H*W))
    A = np.fromfile('../output/RFREQ_'+score+'_Jun-Aug_wk1.dat',dtype=Record, count=1).astype('float')
    var = A[0].reshape((H, W), order='F')
    var = np.transpose(var)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([loni,lone,lati,late])

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    #states_provinces = feature.NaturalEarthFeature(
    #    category='cultural',
    #    name='admin_1_states_provinces_lines',
    #    scale='50ml',
    #    facecolor='none')

    ax.add_feature(feature.LAND)
    ax.add_feature(feature.COASTLINE)
    #ax.add_feature(states_provinces, edgecolor='gray')
    ax.set_title(score)
    pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    pl.xlabels_top = False
    pl.ylabels_right = False
    pl.xformatter = LONGITUDE_FORMATTER
    pl.yformatter = LATITUDE_FORMATTER
    
    #add if for levels, depending on each score
    if score == '2AFC':
        levels = np.linspace(0, 100, 9)
    if score == 'RocAbove':
        levels = np.linspace(0, 1, 9)
    if score == 'RocBelow':
        levels = np.linspace(0, 1, 9)
    if score == 'Spearman':
        levels = np.linspace(-1, 1, 9)
        
    CS=plt.contourf(np.linspace(loni, lone, num=H), np.linspace(lati, late, num=W), var, 
            levels = levels,
            cmap=plt.cm.bwr,
            extend='both') #,transform=proj)

    cbar = plt.colorbar(CS)
    plt.show()

def GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, mon, os, key, week, nlag, training_season, hstep):  
    # Download hindcasts (NCEP) 
    # v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    print("\n Hindcasts URL: \n\n "+url)
    get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_precip_"+mon+"_wk"+str(week)+".tsv")
    #! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv
    return url
    
def GetObs(day1, day2, mon, nday, key, week, nlag, training_season, hstep):
    # Download IMD observations  
    # v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v4 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v6 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv' 
    # v6 precip: (just omits 3./flagge/T)
    url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    
    print("\n Obs data URL: \n\n "+url)
    get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_precip_"+mon+"_wk"+str(week)+".tsv")
    #curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv
    return url
    
def GetObs_RFREQ(day1, day2, mon, nday, key, week, wetday_threshold, nlag, training_season, hstep):
    # Download IMD observations  
    # v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v4 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    # v6 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv' 
    # v6 precip: (just omits 3./flagge/T)
    url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    
    print("\n Obs data URL: \n\n "+url)
    get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_RFREQ_"+mon+"_wk"+str(week)+".tsv")
    #curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv
    return url
    
def GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, key, week, nlag):
    # Download forecast file  
    # v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/70/100/RANGE/Y/0/40/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/(1800%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/RANGE/%5BM%5Daverage/%5BL%5D1/0.0/boxAverage/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/2001/ensotime/12.0/1/Jan/2001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    #url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/2001/ensotime/12.0/1/Jan/2001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
    print("\n Forecast URL: \n\n "+url)
    get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv")
    #curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv
    return url
    
def CPTscript(mon,fday,wk,nla1,sla1,wlo1,elo1,nla2,sla2,wlo2,elo2,fprefix,training_season,ntrain,rainfall_frequency,grads_plot):
        """Function to write CPT namelist file
    
        PARAMETERS
        ----------
            loni: wester longitude
            lone: eastern longitude
            lati: southern latitude
            late: northern latitude
            title: title
        """
        # Set up CPT parameter file
        f=open("params","w+")
        # Opens CCA
        f.write("611\n")
        
        # Opens X input file
        f.write("1\n")
        file='../input/model_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
        f.write(file)
        # Nothernmost latitude
        f.write(str(nla1)+'\n')
        # Southernmost latitude
        f.write(str(sla1)+'\n')
        # Westernmost longitude
        f.write(str(wlo1)+'\n')
        # Easternmost longitude
        f.write(str(elo1)+'\n')
        # Minimum number of X modes
        f.write("1\n")
        # Maximum number of X modes
        f.write("10\n")

        # Opens forecast (X) file
        f.write("3\n")
        file='../input/modelfcst_precip_'+mon+'_fday'+str(fday)+'_wk'+str(wk)+'.tsv\n'
        f.write(file)

        # Opens Y input file
        f.write("2\n")
        if rainfall_frequency:
            file='../input/obs_RFREQ_'+mon+'_wk'+str(wk)+'.tsv\n'
        else:
            file='../input/obs_precip_'+mon+'_wk'+str(wk)+'.tsv\n'         
        f.write(file)
        # Nothernmost latitude
        f.write(str(nla2)+'\n')
        # Southernmost latitude
        f.write(str(sla2)+'\n')
        # Westernmost longitude
        f.write(str(wlo2)+'\n')
        # Easternmost longitude
        f.write(str(elo2)+'\n')
        # Minimum number of Y modes
        f.write("1\n")
        # Maximum number of Y modes
        f.write("10\n")

        # Minimum number of CCA modes
        f.write("1\n")
        # Maximum number of CCAmodes
        f.write("5\n")

        # X training period
        f.write("4\n")
        # First year of X training period
        f.write("1901\n")
        # Y training period
        f.write("5\n")
        # First year of Y training period
        f.write("1901\n")

        # Goodness index
        f.write("531\n")
        # Kendall's tau
        f.write("3\n")

        # Option: Length of training period
        f.write("7\n")
        # Length of training period 
        f.write(str(ntrain)+'\n')
        #   %store 55 >> params
        # Option: Length of cross-validation window
        f.write("8\n")
        # Enter length
        f.write("3\n")

        # Turn ON Transform predictand data
        f.write("541\n")
        # Turn ON zero bound for Y data  (automatically on if variable is precip)
        #%store 542 >> params
        # Turn ON synchronous predictors
        f.write("545\n")
        # Turn ON p-values for masking maps
        #%store 561 >> params

        ### Missing value options
        f.write("544\n")
        # Missing value X flag:
        blurb='-999\n'
        f.write(blurb)
        # Maximum % of missing values
        f.write("10\n")
        # Maximum % of missing gridpoints
        f.write("10\n")
        # Number of near-neighbors
        f.write("1\n")
        # Missing value replacement : best-near-neighbors
        f.write("4\n")
        # Y missing value flag
        blurb='-999\n'
        f.write(blurb)
        # Maximum % of missing values
        f.write("10\n")
        # Maximum % of missing stations
        f.write("10\n")
        # Number of near-neighbors
        f.write("1\n")
        # Best near neighbor
        f.write("4\n")

        #554 # Transformation seetings
        #1   #Empirical distribution

        #######BUILD MODEL AND VALIDATE IT  !!!!!

        if grads_plot:
        # select output format
            f.write("131\n")
        # GrADS format
            f.write("3\n")
        # NB: Default output format is CPTv10 format

        # save goodness index
        f.write("112\n")
        file='../output/'+fprefix+'_Kendallstau_'+training_season+'_wk'+str(wk)+'\n'
        f.write(file)

        # Cross-validation
        f.write("311\n")

        # cross-validated skill maps
        f.write("413\n")
        # save Spearmans Correlation
        f.write("2\n")
        file='../output/'+fprefix+'_Spearman_'+training_season+'_wk'+str(wk)+'\n'
        f.write(file)

        # cross-validated skill maps
        f.write("413\n")
        # save 2AFC score
        f.write("3\n")
        file='../output/'+fprefix+'_2AFC_'+training_season+'_wk'+str(wk)+'\n'
        f.write(file)

        # cross-validated skill maps
        f.write("413\n")
        # save RocBelow score
        f.write("10\n")
        file='../output/'+fprefix+'_RocBelow_'+training_season+'_wk'+str(wk)+'\n'
        f.write(file)

        # cross-validated skill maps
        f.write("413\n")
        # save RocAbove score
        f.write("11\n")
        file='../output/'+fprefix+'_RocAbove_'+training_season+'_wk'+str(wk)+'\n'
        f.write(file)

        #######FORECAST(S)  !!!!!
        # Probabilistic (3 categories) maps
        f.write("455\n")
        # Output results
        f.write("111\n")
        # Forecast probabilities
        f.write("501\n")
        file='../output/'+fprefix+'_CCAFCST_PROB_train_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
        f.write(file)
        #502 # Forecast odds
        #Exit submenu
        f.write("0\n")
    
        # Compute deterministc values and prediction limits
        f.write("454\n")
        # Output results
        f.write("111\n")
        # Forecast values
        f.write("511\n")
        file='../output/'+fprefix+'_CCAFCST_values_train_'+training_season+'_'+mon+'_'+str(fday)+'_wk'+str(wk)+'\n'
        f.write(file)
        #502 # Forecast odds
        #Exit submenu
        f.write("0\n")   

        # Stop saving  (not needed in newest version of CPT)

        # Exit
        f.write("0\n")
        f.write("0\n")
        f.write("0\n")

        f.close()