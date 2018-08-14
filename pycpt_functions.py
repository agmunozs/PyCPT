#This is PyCPT_functions.py (version1.1) -- 11 Aug 2018
#Authors: AG Muñoz (agmunoz@iri.columbia.edu) and AW Robertson (awr@iri.columbia.edu)
#Notes: be sure it matches version of PyCPT 
#To Do: (as Aug 15, 2018 -- AGM)
#	+ Add a percentile threshold for dry days
#	+ Deterministic forecast should re-adjust its colorbar range/units depending on the predictand used
#	+ Simplify download functions: just one function, with the right arguments and dictionaries.
#	+ Check Hindcasts and Forecast_RFREQ
import os
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from cartopy import feature
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
warnings.filterwarnings("ignore")

def lines_that_equal(line_to_match, fp):
	return [line for line in fp if line == line_to_match]

def lines_that_contain(string, fp):
	return [line for line in fp if string in line]

def lines_that_start_with(string, fp):
	return [line for line in fp if line.startswith(string)]

def lines_that_end_with(string, fp):
	return [line for line in fp if line.endswith(string)]

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
	
def PrepFiles(rainfall_frequency,wlo1, wlo2,elo1, elo2,sla1, sla2,nla1, nla2, day1, day2, fday, nday, fyr, mon, os, authkey, wk, wetday_threshold, nlag, training_season, hstep, model, force_download):
	"""Function to download (or not) the needed files"""
	if rainfall_frequency: 
		GetObs_RFREQ(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, authkey, wk, wetday_threshold, nlag, training_season, hstep, model, force_download)
		print('Obs:rfreq file ready to go')
		print('----------------------------------------------')
		GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, authkey, wk, nlag, training_season, hstep, model, force_download)
		#GetHindcasts_RFREQ(wlo1, elo1, sla1, nla1, day1, day2, nday, fyr, mon, os, authkey, wk, wetday_threshold, nlag, training_season, hstep, model, force_download)
		print('Hindcasts file ready to go')
		print('----------------------------------------------')
		#GetForecast_RFREQ(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, authkey, wk, wetday_threshold, nlag, model, force_download)
		GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, authkey, wk, nlag, model, force_download)
		print('Forecasts file ready to go')
		print('----------------------------------------------')
	else:
		GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, authkey, wk, nlag, training_season, hstep, model, force_download)
		print('Hindcasts file ready to go')
		print('----------------------------------------------')
		GetObs(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, authkey, wk, nlag, training_season, hstep, model, force_download)
		print('Obs:precip file ready to go')
		print('----------------------------------------------')
		GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, authkey, wk, nlag, model, force_download)
		print('Forecasts file ready to go')
		print('----------------------------------------------')
	
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
	fig, ax = plt.subplots(figsize=(8,6), subplot_kw=dict(projection=ccrs.PlateCarree()))
	ax.set_extent([loni,lone,lati,late])

	# Put a background image on for nice sea rendering.
	ax.stock_img()

	#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
	states_provinces = feature.NaturalEarthFeature(
		category='cultural',
		name='admin_1_states_provinces_shp',
		scale='10m',
		facecolor='none')
							 
	ax.add_feature(feature.LAND)
	ax.add_feature(feature.COASTLINE)
	ax.set_title(title)
	pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=2, color='gray', alpha=0.5, linestyle='--')
	pl.xlabels_top = False
	pl.ylabels_left = False
	pl.xformatter = LONGITUDE_FORMATTER
	pl.yformatter = LATITUDE_FORMATTER
	ax.add_feature(states_provinces, edgecolor='gray')
	
	return ax

def pltmap(score,loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk):
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
	
	plt.figure(figsize=(20,15))
	
	for L in range(nwk):
		wk=L+1
		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				
		#Prepare to read grads binary file, prepare figure
		Record = np.dtype(('float32', H*W))
		
		ax = plt.subplot(2, 2, wk, projection=ccrs.PlateCarree())
		ax.set_extent([loni,lone,lati,late], ccrs.PlateCarree())

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
			name='admin_1_states_provinces_shp',
			scale='10m',
			facecolor='none')
							 
		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)
		ax.set_title(score+' for Week '+str(wk))
		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=2, color='gray', alpha=0.5, linestyle='--')
		pl.xlabels_top = False
		pl.ylabels_left = True
		pl.ylabels_right = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
		ax.set_ybound(lower=lati, upper=late)
	
		#define colorbars, depending on each score	--This can be easily written as a function
		if score == '2AFC':
			levels = np.linspace(0, 100, 9)
			A = np.fromfile('../output/'+fprefix+'_'+mpref+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat',dtype=Record, count=1).astype('float')
			var = A[0].reshape((W, H), order='F')
			var = np.transpose(var)
			CS=plt.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
				levels = levels,
				cmap=plt.cm.bwr,
				extend='both') #,transform=proj)
			label = '2AFC (%)'

		if score == 'RocAbove':
			levels = np.linspace(0, 1, 9)
			A = np.fromfile('../output/'+fprefix+'_'+mpref+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat',dtype=Record, count=1).astype('float')
			var = A[0].reshape((W, H), order='F')
			var = np.transpose(var)
			CS=plt.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
				levels = levels,
				cmap=plt.cm.bwr,
				extend='both') #,transform=proj)
			label = 'ROC area'

		if score == 'RocBelow':
			levels = np.linspace(0, 1, 9)
			A = np.fromfile('../output/'+fprefix+'_'+mpref+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat',dtype=Record, count=1).astype('float')
			var = A[0].reshape((W, H), order='F')
			var = np.transpose(var)
			CS=plt.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
				levels = levels,
				cmap=plt.cm.bwr,
				extend='both') #,transform=proj)
			label = 'ROC area'

		if score == 'Spearman':
			levels = np.linspace(-1, 1, 9)
			A = np.fromfile('../output/'+fprefix+'_'+mpref+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat',dtype=Record, count=1).astype('float')
			var = A[0].reshape((W, H), order='F')
			var = np.transpose(var)
			CS=plt.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
				levels = levels,
				cmap=plt.cm.bwr,
				extend='both') #,transform=proj)
			label = 'Correlation'


		if score == 'Pearson':
			levels = np.linspace(-1, 1, 9)
			A = np.fromfile('../output/'+fprefix+'_'+mpref+'_'+score+'_'+training_season+'_wk'+str(wk)+'.dat',dtype=Record, count=1).astype('float')
			var = A[0].reshape((W, H), order='F')
			var = np.transpose(var)
			CS=plt.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
				levels = levels,
				cmap=plt.cm.bwr,
				extend='both') #,transform=proj)
			label = 'Correlation'

		if score == 'CCAFCST_V':
			levels = np.linspace(-2, 2, 9)
			A = np.fromfile('../output/'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat',dtype=Record, count=1).astype('float')
			var = A[0].reshape((W, H), order='F')
			var = np.transpose(var)
			CS=plt.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
				levels = levels,
				cmap=plt.cm.bwr,
				extend='both') #,transform=proj)
			ax.set_title("Deterministic forecast for Week "+str(wk))
			label = 'Freq Rainy Days (days)'

		plt.subplots_adjust(hspace=0)
		#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		#cbar_ax = plt.add_axes([0.85, 0.15, 0.05, 0.7])
		#plt.tight_layout()
		plt.subplots_adjust(bottom=0.15, top=0.9)
		cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
		cbar.set_label(label) #, rotation=270)
		plt.gca().set_rasterization_zorder(-1)
		
		
def pltmapProb(loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk):
	"""A simple plot function for ploting probabilistic forecasts
	
	PARAMETERS
	----------
		score: the score
		loni: wester longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
		title: title
	"""
	score = 'CCAFCST_P'

	plt.figure(figsize=(15,15))
	
	for L in range(nwk):
		wk=L+1
		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
		with open('../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				
		#Prepare to read grads binary file, prepare figure
		Record = np.dtype(('float32', H*W))
		

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
			name='admin_1_states_provinces_shp',
			scale='10m',
			facecolor='none')
							 
		levels = np.linspace(0, 100, 31)
		B = np.fromfile('../output/'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat',dtype=Record, count=-1).astype('float32')
		tit=['Below Normal','Normal','Above Normal']
		for i in range(3):
				var = np.transpose(B[i].reshape((W, H), order='F'))
				ax2=plt.subplot(nwk, 3, (L*3)+(i+1),projection=ccrs.PlateCarree())
				ax2.set_title("Category: "+tit[i]+" - Week "+str(wk))
				ax2.add_feature(feature.LAND)
				ax2.add_feature(feature.COASTLINE)
				ax2.set_ybound(lower=lati, upper=late)
				pl2=ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					linewidth=2, color='gray', alpha=0.5, linestyle='--')
				pl2.xlabels_top = False
				pl2.ylabels_left = True
				pl2.ylabels_right = False
				pl2.xformatter = LONGITUDE_FORMATTER
				pl2.yformatter = LATITUDE_FORMATTER
				ax2.add_feature(states_provinces, edgecolor='gray')
				#ax2.set_adjustable('box')
				#ax2.set_aspect('auto',adjustable='datalim',anchor='C')
				CS=ax2.contourf(np.linspace(loni, lone, num=W), np.linspace(lati, late, num=H), var, 
					levels = levels,
					cmap=plt.cm.bwr,
					extend='both')
				#plt.show(block=False)
				
	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
	cbar.set_label('Probability (%)') #, rotation=270)
	plt.gca().set_rasterization_zorder(-1)

		
def GetHindcasts(wlo1, elo1, sla1, nla1, day1, day2, fyr, mon, os, key, week, nlag, training_season, hstep,model, force_download):  
	# Download hindcasts (NCEP) 
	# v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	if not force_download:
		try:
			ff=open("model_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("OS error: {0}".format(err))
			print("Hindcasts file doesn't exist --downloading")
			force_download = True	 
	if force_download:
		#dictionary:
		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/('+mon+'%20'+str(fyr)+')/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
		}
		# calls curl to download data
		url=dic[model]
		print("\n Hindcasts URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f model_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv

def GetHindcasts_RFREQ(wlo1, elo1, sla1, nla1, day1, day2, nday, fyr, mon, os, key, week, wetday_threshold, nlag, training_season, hstep,model, force_download):  
	# Download hindcasts (NCEP) 
	# v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/:grid/use_as_grid/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	if not force_download:
		try:
			ff=open("model_RFREQ_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("OS error: {0}".format(err))
			print("Hindcasts file doesn't exist --downloading")
			force_download = True	 
	if force_download:
		#dictionary:
		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/dup/S/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.S/replaceGRID/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/c://name//water_density/def/998/%28kg/m3%29/:c/div//mm/unitconvert//name/(tp)/def/grid://name/%28T%29/def//units/%28months%20since%201960-01-01%29/def//standard_name/%28time%29/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/:grid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/('+mon+'%20'+str(fyr)+')/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
		}
		# calls curl to download data
		url=dic[model]
		print("\n Hindcasts URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > model_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f model_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		#! curl -g -k -b '__dlauth_id='$key'' ''$url'' > model_precip_${mo}.tsv
	
def GetObs(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, nlag, training_season, hstep, model, force_download):
	# Download IMD observations	 
	# v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v4 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v6 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv' 
	# v6 precip: (just omits 3./flagge/T)
	#IMD: last version: url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	if not force_download:
		try:
			ff=open("obs_precip_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("OS error: {0}".format(err))
			print("Obs precip file doesn't exist --downloading")
			force_download = True	 
	if force_download:
		#dictionary:
		dic = {'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
			   'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/('+mon+'%20'+str(fyr)+')/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/0./1.5/360./GRID/Y/-50/1.5/50/GRID/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/dup%5BT%5Daverage/sub/-999/setmissing_value/nip/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Obs (Rainfall) data URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f obs_precip_"+mon+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv
	
def GetObs_RFREQ(day1, day2, mon, fyr, wlo2, elo2, sla2, nla2, nday, key, week, wetday_threshold, nlag, training_season, hstep,model, force_download):
	# Download IMD observations	 
	# v3 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+mon+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/1960/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v4 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/7/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2068/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	# v6 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/3./flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv' 
	# v6 precip: (just omits 3./flagge/T)
	# IMD: last version: url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.IMD/.NCC1-2005/.v4p0/.rf/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	if not force_download:
		try:
			ff=open("obs_RFREQ_"+mon+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("OS error: {0}".format(err))
			print("Obs freq-rainfall file doesn't exist --downloading")
			force_download = True	 
	if force_download:
		#dictionary:
		dic = { 'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/S/(0000%201%20Jan%201999)/(0000%2031%20Dec%202010)/RANGEEDGES/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/S/('+training_season+')/VALUES/S/'+str(hstep)+'/STEP/L1/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL1/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/0./1.5/360./GRID/Y/-50/1.5/50/GRID/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201999)/(31%20Dec%202011)/RANGEEDGES/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/('+mon+'%20'+str(fyr)+')/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/-999/setmissing_value/hdate/('+str(fyr-20)+')/('+str(fyr-1)+')/RANGE/dup/%5Bhdate%5Daverage/sub/%5BM%5Daverage/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20'+str(fyr)+'-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/0./1.5/360./GRID/Y/-50/1.5/50/GRID/Y/'+str(sla2)+'/'+str(nla2)+'/RANGE/X/'+str(wlo2)+'/'+str(elo2)+'/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/2060/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Obs (Freq) data URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > obs_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f obs_RFREQ_"+mon+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > obs_precip_${mo}.tsv
	
def GetForecast(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, key, week, nlag, model, force_download):
	# Download forecast file  
	# v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/70/100/RANGE/Y/0/40/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/(1800%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/RANGE/%5BM%5Daverage/%5BL%5D1/0.0/boxAverage/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/2001/ensotime/12.0/1/Jan/2001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	#url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/2001/ensotime/12.0/1/Jan/2001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	if not force_download:
		try:
			ff=open("modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("OS error: {0}".format(err))
			print("Forecasts file doesn't exist --downloading")
			force_download = True	 
	if force_download:
		#dictionary:
		dic = {	'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/%5BM%5Daverage/%5Bhdate%5Daverage/sub/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Forecast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f modelfcst_precip_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv
		
def GetForecast_RFREQ(day1, day2, fday, mon, fyr, nday, wlo1, elo1, sla1, nla1, key, week, wetday_threshold, nlag, model, force_download):
	# Download forecast file  
	# v5 url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/%5BM%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/add/4./div/X/70/100/RANGE/Y/0/40/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/(1800%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/RANGE/%5BM%5Daverage/%5BL%5D1/0.0/boxAverage/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/2001/ensotime/12.0/1/Jan/2001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	#url='http://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/2001/ensotime/12.0/1/Jan/2001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(tp)/def//units/(mm)/def//long_name/(precipitation_amount)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
	if not force_download:
		try:
			ff=open("modelfcst_RFREQ_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv", 'r')
			s = ff.readline()
		except OSError as err:
			print("OS error: {0}".format(err))
			print("Forecasts file doesn't exist --downloading")
			force_download = True	 
	if force_download:
		#dictionary:  #CFSv2 needs to be transformed to RFREQ!
		dic = {	'CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/3./mul/SOURCES/.ECMWF/.S2S/.NCEP/.reforecast/.control/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag%5Daverage/add/4./div/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/L1/'+str(day1-1)+'/'+str(day2)+'/VALUES/%5BL1%5Ddifferences/L1/removeGRID/S/(0000%20'+str(fday)+'%20'+mon+')/VALUES/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.6_hourly_rotating/.FLXF/.surface/.PRATE/%5BL%5D1/0.0/boxAverage/S/-'+str(nlag-1)+'/1/0/shiftdatashort/%5BS_lag/M%5Daverage/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')VALUE/%5BX/Y%5DregridLinear/L/'+str(day1)+'/'+str(day2)+'/RANGEEDGES/%5BL%5Daverage/%5BS%5Daverage/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div/(mm/day)/unitconvert/'+str(nday)+'/mul//units/(mm)/def/exch/sub/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
				'ECMWF': 'https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.forecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/SOURCES/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/'+str(sla1)+'/'+str(nla1)+'/RANGE/X/'+str(wlo1)+'/'+str(elo1)+'/RANGE/L/('+str(day1-1)+')/('+str(day2)+')/VALUES/S/(0000%20'+str(fday)+'%20'+mon+'%20'+str(fyr)+')/VALUE/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert/'+str(wetday_threshold)+'/flagge/T/'+str(nday)+'/runningAverage/'+str(nday)+'.0/mul/%5Bhdate%5Daverage/sub/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/1/Jan/3001/ensotime/12.0/1/Jan/3001/ensotime/%3Agrid/addGRID/T//pointwidth/0/def/pop//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz'
			  }
		# calls curl to download data
		url=dic[model]
		print("\n Forecast URL: \n\n "+url)
		get_ipython().system("curl -g -k -b '__dlauth_id="+key+"' '"+url+"' > modelfcst_RFREQ_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		get_ipython().system("gunzip -f modelfcst_RFREQ_"+mon+"_fday"+str(fday)+"_wk"+str(week)+".tsv.gz")
		#curl -g -k -b '__dlauth_id='$key'' ''$url'' > modelfcst_precip_fday${fday}.tsv
	
def CPTscript(mon,fday,wk,nla1,sla1,wlo1,elo1,nla2,sla2,wlo2,elo2,fprefix,mpref,training_season,ntrain,rainfall_frequency,MOS):
		"""Function to write CPT namelist file
	
		"""
		# Set up CPT parameter file
		f=open("params","w")
		if MOS=='CCA':
			# Opens CCA
			f.write("611\n")
		elif MOS=='None': 
			# Opens GCM
			f.write("614\n")
		else:
			print ("MOS option is invalid")

		# First, ask CPT to stop if error is encountered
		f.write("571\n")
		f.write("3\n")
		
		# Opens X input file
		f.write("1\n")
		if rainfall_frequency:
			file='../input/model_precip_'+mon+'_wk'+str(wk)+'.tsv\n'
		else:
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
		if MOS=='CCA':
			# Minimum number of X modes
			f.write("1\n")
			# Maximum number of X modes
			f.write("10\n")

			# Opens forecast (X) file
			f.write("3\n")
			if rainfall_frequency:
				file='../input/modelfcst_precip_'+mon+'_fday'+str(fday)+'_wk'+str(wk)+'.tsv\n'
			else:
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
		if MOS=='CCA':
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
		#	%store 55 >> params
		# Option: Length of cross-validation window
		f.write("8\n")
		# Enter length
		f.write("3\n")

		# Turn ON Transform predictand data 
		f.write("541\n")
		# Turn ON zero bound for Y data	 (automatically on by CPT if variable is precip)
		#f.write("542\n")
		# Turn ON synchronous predictors
		f.write("545\n")
		# Turn ON p-values for masking maps
		#f.write("561\n")

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

		# Transformation settings
		#f.write("554\n")
		# Empirical distribution
		#f.write("1\n")	 

		#######BUILD MODEL AND VALIDATE IT	!!!!!

		# NB: Default output format is GrADS format
		# select output format
		f.write("131\n")
		# GrADS format
		f.write("3\n")

		# save goodness index
		f.write("112\n")
		file='../output/'+fprefix+'_'+mpref+'_Kendallstau_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# Cross-validation
		f.write("311\n")

		# cross-validated skill maps
		f.write("413\n")
		# save Pearson's Correlation
		f.write("1\n")
		file='../output/'+fprefix+'_'+mpref+'_Pearson_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)
		
		# cross-validated skill maps
		f.write("413\n")
		# save Spearmans Correlation
		f.write("2\n")
		file='../output/'+fprefix+'_'+mpref+'_Spearman_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save 2AFC score
		f.write("3\n")
		file='../output/'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocBelow score
		f.write("10\n")
		file='../output/'+fprefix+'_'+mpref+'_RocBelow_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		# cross-validated skill maps
		f.write("413\n")
		# save RocAbove score
		f.write("11\n")
		file='../output/'+fprefix+'_'+mpref+'_RocAbove_'+training_season+'_wk'+str(wk)+'\n'
		f.write(file)

		if MOS=='CCA':   #DO NOT USE CPT to compute probabilities if MOS='None' --use IRIDL for direct counting
			#######FORECAST(S)	!!!!!
			# Probabilistic (3 categories) maps
			f.write("455\n")
			# Output results
			f.write("111\n")
			# Forecast probabilities
			f.write("501\n")
			file='../output/'+fprefix+'_'+mpref+'FCST_P_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
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
			file='../output/'+fprefix+'_'+mpref+'FCST_V_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'\n'
			f.write(file)
			#502 # Forecast odds
			#Exit submenu
			f.write("0\n")	 

			# Stop saving  (not needed in newest version of CPT)

		# Exit
		f.write("0\n")

		f.close()