import numpy as np 
from scipy.signal import butter, lfilter 
import netCDF4 as nc 
import matplotlib.pyplot as plt

# 带通滤波
def butter_bandpass_filter(data, lowcut, highcut, sample_span, order=5):
    low = sample_span / lowcut * 2
    high = sample_span / highcut * 2
    b, a = butter(order, [low, high], btype='band')
    data_bbf = lfilter(b, a, data)
    return data_bbf 
  
if __name__ == '__main__':   
    start_day = 1
    end_day = 365
    filename = 'data/precip.2010.nc' 
    with nc.Dataset(filename, 'r') as f:
        lat = f.variables['lat'][:]
        lon = f.variables['lon'][:]
        precip = f.variables['precip'][start_day-1:end_day]
        ndays = precip.shape[0]
    days = np.arange(start_day, end_day+1)
    xlim = [start_day, end_day+1]
        
    #%%
    onepoint_precip = precip[:, :, np.logical_and(lon>115, lon<120)][:, np.logical_and(lat>30, lat<32), :]
    op_precip_mean = np.mean(np.mean(onepoint_precip, axis=-1), axis=-1)  
    op_precip_mean_a = np.ma.anom(op_precip_mean)   
    
    # # 带通滤波
    dt = 1
    lowcut = 50
    highcut = 30
    precip_bbf = butter_bandpass_filter(op_precip_mean_a, lowcut, highcut, dt, order=2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.bar(days, op_precip_mean_a, width=0.8, color='b', label='original')
    ax.plot(days, precip_bbf, lw=0.8, color='r', label='%d-%dd filter'%(highcut, lowcut))
    ax.set_xlim(xlim)
    ax.legend()      
    plt.show()  
    
    #%%
    from wavelet import wavelet
    from wave_signif import wave_signif
    
    timelist = op_precip_mean
    variance = np.std(timelist)**2
    mean = np.mean(timelist)
    timelist = (timelist - mean) / np.sqrt(variance)

    # 小波分析参数 
    pad = 1      # pad the time series with zeroes (recommended)
    dj = 0.25    # this will do 4 sub-octaves per octave
    s0 = 2*dt    # this says start at a scale of 6 months
    j1 = -1 # 7./dj    # this says do 7 powers-of-two with dj sub-octaves each    
    mother = 'Morlet'
    # 小波分析
    wave, period, scale, coi = wavelet(timelist, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave))**2
    # 置信检验
    sigtest = 0
    lag1 = 0.72  # lag-1 autocorrelation for red noise background
    siglvl = 0.95
    dof = -1
    signif, fft_theor = wave_signif(timelist, dt, scale, sigtest, lag1, siglvl, dof, mother)
    sig95 = np.dot(signif.reshape(len(signif), 1), np.ones((1, ndays))) # expand signif --> (J+1)x(N) array
    sig95 = power / sig95         # where ratio > 1, power is significa    
    # Global wavelet spectrum & significance levels
    global_ws = variance * power.sum(axis=1) / ndays   # time-average over all times
    dof = ndays - scale  # the -scale corrects for padding at edges
    global_signif,fft_theor = wave_signif(variance, dt, scale, 1, lag1, siglvl, dof, mother)
    
    # Scale-average between period of 30--50 days
    avg = (scale >= highcut) & (scale < lowcut)
    Cdelta = 0.776;   # this is for the MORLET wavelet
    scale_avg = np.dot(scale.reshape(len(scale), 1), np.ones((1, ndays))) # expand scale --> (J+1)x(N) array
    scale_avg = power / scale_avg   # [Eqn(24)]
    scale_avg = variance*dj * dt / Cdelta * sum(scale_avg[avg, :])   # [Eqn(24)]
    scaleavg_signif, fft_theor = wave_signif(variance, dt, scale, 2, lag1, siglvl, [2,7.9], mother)
    
    # Reconstuction
    from wavelet_inverse import wavelet_inverse
    iwave = wavelet_inverse(wave, scale, dt, dj, "Morlet")
    #%%
    fig=plt.figure(figsize=(10, 10))

    # subplot positions
    width = 0.65
    hight = 0.28;
    pos1a = [0.1, 0.75, width, 0.2]
    pos1b = [0.1, 0.37, width, hight]
    pos1c = [0.79, 0.37, 0.18, hight]
    pos1d = [0.1,  0.07, width, 0.2]
    
    ax = fig.add_axes(pos1a)
    ax.plot(days, timelist*np.sqrt(variance)+mean, "r-")
    #reconstruction
    ax.plot(days, iwave*np.sqrt(variance)+mean, "k--")
    ax.set_ylabel('Precipitation (mm)')
    plt.title('a) Precipitation (one year)')
    
    bx = fig.add_axes(pos1b, sharex=ax)
    # levels = [8, 16, 32, 64, 128, 256, 512, 1024] 
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16] 
    Yticks = 2**(np.arange(np.int(np.log2(np.min(period))), np.int(np.log2(np.max(period)))+1))
    cn = bx.contourf(days, np.log2(period), np.log2(power), np.log2(levels), cmap='Spectral_r', extend='both')
    bx.set_xlabel('Time (day)')
    bx.set_ylabel('Period (days)')
    # cb = fig.colorbar(cn, ax=bx, orientation='horizontal', shrink=0.8, aspect=25, pad=0.17)
    import matplotlib.ticker as ticker
    ymajorLocator = ticker.FixedLocator(np.log2(Yticks))
    bx.yaxis.set_major_locator(ymajorLocator)
    ticks = bx.yaxis.set_ticklabels(Yticks)
    bx.set_title('b) Wavelet Power Spectrum')
    
    cs = bx.contour(days, np.log2(period), sig95, [1], color='k', linewidth=1, linestyles='dashed')
    
    # cone-of-influence, anything "below" is dubious
    ts = days
    coi[coi <= np.min(period)] = np.min(period)
    coi_area = np.concatenate([[np.max(scale)], coi, [np.max(scale)], [np.max(scale)]])
    ts_area = np.concatenate([[ts[0]], ts, [ts[-1]] ,[ts[0]]]);
    L = bx.plot(ts_area, np.log2(coi_area), 'k', linewidth=3)
    F = bx.fill(ts_area, np.log2(coi_area), 'k', alpha=0.3, hatch="x")
    
    #--- Plot global wavelet spectrum
    cx = fig.add_axes(pos1c, sharey=bx)
    cx.plot(global_ws, np.log2(period), 'r-')
    cx.plot(global_signif, np.log2(period), 'k--')
    ylim = cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.invert_yaxis()
    cx.set_title('c) Global Wavelet Spectrum')
    xrangec = cx.set_xlim([0, 1.25*np.max(global_ws)])
    
    #--- Plot Scale-averaged spectrum -----------------
    dx = fig.add_axes(pos1d, sharex=bx)
    dx.plot(days, scale_avg, "r-")
    dx.plot([days[0], days[-1]], [scaleavg_signif, scaleavg_signif], "k--")
    xrange = dx.set_xlim(xlim)
    dx.set_ylabel('Avg variance (mm$^2$)')
    dx.set_title('d) Scale-average Time Series')
    plt.show()
