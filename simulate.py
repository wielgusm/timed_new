import numpy as np
import pandas as pd
import itertools
try:
    import ehtim as eh
except ModuleNotFoundError:
    sys.path.append('/Volumes/DATAPEN/Shared/EHT/EHTIM/eht-imaging_fork/eht-imaging/')
    import ehtim as eh

def simulate_lcamp(snrs,amps=np.ones(4),N=1000,debias=True,snr_calc_N=-1):
    snrs=np.asarray(snrs)
    amps=np.asarray(amps)
    sigmas = amps/snrs
    
    x0 = amps[0] + sigmas[0]*np.random.randn(N) + 1j*sigmas[0]*np.random.randn(N)
    x1 = amps[1] + sigmas[1]*np.random.randn(N) + 1j*sigmas[1]*np.random.randn(N)
    x2 = amps[2] + sigmas[2]*np.random.randn(N) + 1j*sigmas[2]*np.random.randn(N)
    x3 = amps[3] + sigmas[3]*np.random.randn(N) + 1j*sigmas[3]*np.random.randn(N)
    
    if snr_calc_N > 0 :
        x0s = amps[0] + sigmas[0]*np.random.randn(snr_calc_N) + 1j*sigmas[0]*np.random.randn(snr_calc_N)
        x1s = amps[1] + sigmas[1]*np.random.randn(snr_calc_N) + 1j*sigmas[1]*np.random.randn(snr_calc_N)
        x2s = amps[2] + sigmas[2]*np.random.randn(snr_calc_N) + 1j*sigmas[2]*np.random.randn(snr_calc_N)
        x3s = amps[3] + sigmas[3]*np.random.randn(snr_calc_N) + 1j*sigmas[3]*np.random.randn(snr_calc_N)
        snr0 = np.mean(x0s/sigmas[0])
        snr1 = np.mean(x1s/sigmas[0])
        snr2 = np.mean(x2s/sigmas[0])
        snr3 = np.mean(x3s/sigmas[0])
        snrs=np.asarray([snr0,snr1,snr2,snr3])
    
    A0 = np.abs(x0)
    A1 = np.abs(x1)
    A2 = np.abs(x2)
    A3 = np.abs(x3)
    
    if debias:
        A0 = deb_sample(A0,sigmas[0])
        A1 = deb_sample(A1,sigmas[0])
        A2 = deb_sample(A2,sigmas[0])
        A3 = deb_sample(A3,sigmas[0])
    
    mask = (A0>0)&(A1>0)&(A2>0)&(A3>0)
    
    lA0 = np.log(A0[mask])
    lA1 = np.log(A1[mask])
    lA2 = np.log(A2[mask])
    lA3 = np.log(A3[mask])
    
    lcamp = lA0 + lA1 - lA2 - lA3
    #sigma = np.std(lcamp)
    sigma = np.sqrt(np.sum(1./snrs**2))
    return lcamp, sigma


def simulate_camp(snrs,amps=np.ones(4),N=1000,debias=True,snr_calc_N=-1):
    snrs=np.asarray(snrs)
    amps=np.asarray(amps)
    sigmas = amps/snrs
    
    x0 = amps[0] + sigmas[0]*np.random.randn(N) + 1j*sigmas[0]*np.random.randn(N)
    x1 = amps[1] + sigmas[1]*np.random.randn(N) + 1j*sigmas[1]*np.random.randn(N)
    x2 = amps[2] + sigmas[2]*np.random.randn(N) + 1j*sigmas[2]*np.random.randn(N)
    x3 = amps[3] + sigmas[3]*np.random.randn(N) + 1j*sigmas[3]*np.random.randn(N)
    
    if snr_calc_N > 0 :
        x0s = amps[0] + sigmas[0]*np.random.randn(snr_calc_N) + 1j*sigmas[0]*np.random.randn(snr_calc_N)
        x1s = amps[1] + sigmas[1]*np.random.randn(snr_calc_N) + 1j*sigmas[1]*np.random.randn(snr_calc_N)
        x2s = amps[2] + sigmas[2]*np.random.randn(snr_calc_N) + 1j*sigmas[2]*np.random.randn(snr_calc_N)
        x3s = amps[3] + sigmas[3]*np.random.randn(snr_calc_N) + 1j*sigmas[3]*np.random.randn(snr_calc_N)
        snr0 = np.mean(x0s/sigmas[0])
        snr1 = np.mean(x1s/sigmas[0])
        snr2 = np.mean(x2s/sigmas[0])
        snr3 = np.mean(x3s/sigmas[0])
        snrs=np.asarray([snr0,snr1,snr2,snr3])
    
    A0 = np.abs(x0)
    A1 = np.abs(x1)
    A2 = np.abs(x2)
    A3 = np.abs(x3)
    
    if debias:
        A0 = deb_sample(A0,sigmas[0])
        A1 = deb_sample(A1,sigmas[0])
        A2 = deb_sample(A2,sigmas[0])
        A3 = deb_sample(A3,sigmas[0])
    
    mask = (A0>0)&(A1>0)&(A2>0)&(A3>0)
    
    
    camp = A0*A1/A2/A3
    #sigma = np.std(lcamp)
    sigma = camp*np.sqrt(np.sum(1./snrs**2))
    return camp, sigma


def simulate_cphase(snrs,amps=np.ones(3),N=1000,snr_calc_N=-1):
    snrs=np.asarray(snrs)
    amps=np.asarray(amps)
    sigmas = amps/snrs
    
    x0 = amps[0] + sigmas[0]*np.random.randn(N) + 1j*sigmas[0]*np.random.randn(N)
    x1 = amps[1] + sigmas[1]*np.random.randn(N) + 1j*sigmas[1]*np.random.randn(N)
    x2 = amps[2] + sigmas[2]*np.random.randn(N) + 1j*sigmas[2]*np.random.randn(N)
    
    if snr_calc_N > 0 :
        x0s = amps[0] + sigmas[0]*np.random.randn(snr_calc_N) + 1j*sigmas[0]*np.random.randn(snr_calc_N)
        x1s = amps[1] + sigmas[1]*np.random.randn(snr_calc_N) + 1j*sigmas[1]*np.random.randn(snr_calc_N)
        x2s = amps[2] + sigmas[2]*np.random.randn(snr_calc_N) + 1j*sigmas[2]*np.random.randn(snr_calc_N)
        snr0 = np.mean(x0s/sigmas[0])
        snr1 = np.mean(x1s/sigmas[0])
        snr2 = np.mean(x2s/sigmas[0])
        snrs=np.asarray([snr0,snr1,snr2])
    
    B=x0*x1*x2
    cp=np.angle(B)
      
    sigma = np.sqrt(np.sum(1./snrs**2))
    return cp, sigma

def deb_sample(amps,sigma):
    amps = np.asarray(amps)
    amps_deb = (amps**2 - sigma**2)
    amps_deb = np.sqrt(np.maximum(0,amps_deb))
    return amps_deb


def simulate_obs(amps,sigmas,cadence,duration,station_N=4):
    alphabet=['A','B','C','D',"E",'F','G','H','I','J','K','L','M','N']
    stationsL=alphabet[:station_N]
    baselinesL=[x[0]+x[1] for x in list(itertools.combinations(stationsL,2))]
    stations1 = [x[0] for x in baselinesL ]
    stations2 = [x[1] for x in baselinesL ]
    numB=len(baselinesL)
    try: 
        if len(amps)==1: amps=amps*numB
    except: amps=[amps]*numB
    try:
        N = len(sigmas)
        if N==1: sigmas=sigmas*numB
    except: sigmas=[sigmas]*numB

    N=int(duration/cadence)
    
    visL = [np.asarray(amps[c] + sigmas[c]*np.random.randn(N) + 1j*sigmas[c]*np.random.randn(N)) for c in range(numB)] 

    baselinesL=[x[0]+x[1] for x in list(itertools.combinations(stationsL,2))]
    stations1 = [x[0] for x in baselinesL ]
    stations2 = [x[1] for x in baselinesL ]
    df = pd.DataFrame({})
    for cou,vis in enumerate(visL):
        df_foo = pd.DataFrame({})
        df_foo['time'] = np.arange(0,N*cadence,cadence)
        #print(len(np.arange(0,N*cadence,cadence)))

        df_foo['vis'] = visL[cou]
        df_foo['sigma']=sigmas[cou]
        df_foo['amp'] = np.abs(df_foo['vis'])
        df_foo['phase'] = np.angle(df_foo['vis'])*180/np.pi
        df_foo['baseline'] = baselinesL[cou]
        df_foo['t1'] = stations1[cou]
        df_foo['t2'] = stations2[cou]
        df_foo['tint']=0
        df['tau1']=0
        df['tau2']=0
        df['u']=0
        df['v']=0
        df['qvis']=0
        df['uvis']=0
        df['vvis']=0
        df['qsigma']=0
        df['usigma']=0
        df['vsigma']=0

        df = pd.concat([df,df_foo],ignore_index=True,sort=False)
    
    tar=np.array([ ('A',  2225060.8136 , -5440059.59994, -2481681.15054,  0.,  0.,  0.+0.j,  0.+0.j,  0.,  0.,  0.),
       ('B',  2225039.5297 , -5441197.6292 , -2479303.3597 ,  0.,  0.,  0.+0.j,  0.+0.j,  0.,  0.,  0.),
       ('C', -1828796.2    , -5054406.8    ,  3427865.2    ,  0.,  0.,  0.+0.j,  0.+0.j,  0.,  0.,  0.),
       ('D', -5464584.676  , -2493001.17   ,  2150653.982  ,  0.,  0.,  0.+0.j,  0.+0.j,  0.,  0.,  0.)], 
      dtype=[('site', '<U32'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('sefdr', '<f8'), ('sefdl', '<f8'), ('dr', '<c16'), ('dl', '<c16'), ('fr_par', '<f8'), ('fr_elev', '<f8'), ('fr_off', '<f8')])
    obs = eh.obsdata.Obsdata(0,0,0,0,eh.statistics.dataframes.df_to_rec(df,'vis'),tar)
    return obs
    #newobs = Obsdata(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd,
    #                     ampcal=self.ampcal, phasecal=self.phasecal, opacitycal=self.opacitycal, dcal=self.dcal, frcal=self.frcal,
    #                     timetype=self.timetype, scantable=self.scans)