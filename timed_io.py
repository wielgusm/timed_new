import sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import qmetric
from timed_new import qmetric
from astropy.time import Time
import datetime as datetime
try:
    import ehtim as eh
except ModuleNotFoundError:
    sys.path.append('/Volumes/DATAPEN/Shared/EHT/EHTIM/eht-imaging_polrep/eht-imaging/')
import ehtim as eh

nam2lett = {'ALMA':'A','AA':'A','A':'A',
            'APEX':'X','AP':'X','X':'X',
            'LMT':'L','LM':'L','L':'L',
            'PICOVEL':'P','PICO':'P','PV':'P','P':'P','IRAM30':'P',
            'SMTO':'Z','SMT':'Z','AZ':'Z','Z':'Z',
            'SPT':'Y','SP':'Y','Y':'Y',
            'JCMT':'J','JC':'J','J':'J',
            'SMAP':'S','SMA':'S','SM':'S','S':'S',
            'SMAR':'R','R':'R','SR':'R',
            'B':'B','C':'C','D':'D'}
pol_dic={'LL':'ll','ll':'ll','L':'ll',
        'RR':'rr','rr':'rr','R':'rr',
        'RL':'rl','rl':'rl',
        'LR':'lr','lr':'lr'}

def load_uvfits(path_to_data,tcoh=-1,single_letter=True,polrep='circ',polar=None):
    if polar=='LL':polar='L'
    if polar=='RR':polar='R'
    try: obs = eh.obsdata.load_uvfits(path_to_data,polrep=polrep,force_singlepol=polar)
    except TypeError: obs = eh.obsdata.load_uvfits(path_to_data,force_singlepol=polar)
    #if full_polar: obs.df = make_df_full_cp(obs)
    #else: obs.df = eh.statistics.dataframes.make_df(obs)
    obs.df = eh.statistics.dataframes.make_df(obs)
    if (tcoh > 0):
        obs = obs.avg_coherent(inttime=tcoh)
    tobs=tobsdata(obs,single_letter=single_letter)
    return tobs

class tobsdata:
    def __init__(self,obs,single_letter=True):
        try: self.df=obs.df
        except AttributeError:
            obs.df = eh.statistics.dataframes.make_df(obs)
        if single_letter:
            if np.mean([len(x) for x in np.asarray(obs.df['baseline'])]) > 2.5:
                obs.df['baseline'] = [nam2lett[x.split('-')[0]]+nam2lett[x.split('-')[1]] for x in list(obs.df['baseline'])]

        self.source = obs.source
        self.df=obs.df
        self.ra=obs.ra
        self.dec=obs.dec
        self.data=obs.data
        self.mjd=obs.mjd

    def get_tseries(self,ident,product='',polar='none'):
        return tseries(self,ident,product=product,polar=polar) 
    
class tseries:
    def __init__(self,tobs,ident,product='',polar='none'):      
        if product=='':
            if len(ident)==2: product='amp'
            elif len(ident)==3: product='cphase'
            elif len(ident)==4: product='lcamp'
        self.type = product
        self.ident = ident
        self.polarization = polar
        self.source = tobs.source
        if product=='amp':
            foo = tobs.df[(tobs.df.baseline==ident) | (tobs.df.baseline==ident[1]+ident[0])]
            if polar != 'none':
                polamp=pol_dic[polar]+'amp'
                polsigma=pol_dic[polar]+'sigma'
            else: polamp='amp'; polsigma='sigma'
            foo=foo[foo[polamp]==foo[polamp]].copy()
            self.mjd = np.asarray(foo.mjd)
            self.time = np.asarray(foo.time)
            self.amp = np.asarray(foo[polamp])
            self.sigma = np.asarray(foo[polsigma])
            self.data = foo

        elif product=='cphase':
            foo = get_cphase(tobs,ident,polar=polar)
            foo=foo[foo.cphase==foo.cphase].copy()
            self.mjd = np.asarray(foo.mjd)
            self.time = np.asarray(foo.time)
            self.cphase = np.asarray(foo.cphase)
            self.sigmaCP = np.asarray(foo.sigmaCP)
            self.data = foo

        elif product=='lcamp':
            foo = get_lcamp(tobs,ident,polar=polar)
            #if polar!='none': foo = foo.dropna(subset=[polamp])
            foo=foo[foo.lcamp==foo.lcamp].copy()
            self.mjd = np.asarray(foo.mjd)
            self.time = np.asarray(foo.time)
            self.lcamp = np.asarray(foo.lcamp)
            self.sigmaLCA = np.asarray(foo.sigmaLCA)
            self.data = foo

    def plot(self,line=False,figsize='',errorscale=1.):
        if figsize=='':
            plt.figure(figsize=(10,5))
        else:
            plt.figure(figsize=figsize)
        if line: fmt='o-'
        else: fmt='o'
        plt.title(self.ident)
        if self.type=='cphase':
            plt.errorbar(self.time,self.cphase,errorscale*self.sigmaCP,fmt=fmt,capsize=5)
            plt.ylabel('cphase [deg]')
        elif self.type=='amp':
            plt.errorbar(self.time,self.amp,errorscale*self.sigma,fmt=fmt,capsize=5)
            plt.ylabel('amp')
        elif self.type=='lcamp':
            plt.errorbar(self.time,self.lcamp,errorscale*self.sigmaLCA,fmt=fmt,capsize=5)
            plt.ylabel('log camp')
        plt.grid()
        plt.xlabel('time [h]')
        plt.show()

    def hist(self,figsize='',perc=2.,show_normal=True):
        if figsize=='':
            plt.figure(figsize=(10,5))
        else:
            plt.figure(figsize=figsize)
        if self.type=='cphase':
            x=self.cphase
            err=self.sigmaCP
            rel_cl = self.cphase/self.sigmaCP
            plt.xlabel('(closure phase) / (estimated error)')
        elif self.type=='lcamp':
            x=self.lcamp
            err=self.sigmaLCA
            rel_cl = self.lcamp/self.sigmaLCA
            plt.xlabel('(log closure amp) / (estimated error)')
        elif self.type=='amp':
            x=(self.amp-np.mean(self.amp))
            err=self.sigma
            rel_cl = (self.amp-np.mean(self.amp))/self.sigma
            plt.xlabel('(amp - mean amp) / (estimated error)')
        
        binL = np.percentile(rel_cl,perc)
        binR = np.percentile(rel_cl,100.-perc)
        binDist = np.abs(binR-binL)
        binR = binR + 0.1*binDist
        binL = binL - 0.1*binDist
        bins = np.linspace(binL,binR,int(1.2*np.sqrt(len(rel_cl))))
        plt.hist(rel_cl,bins=bins,normed=True)

        if show_normal:
            plt.axvline(0,color='k',linestyle='--')
            xg = np.linspace(binL, binR,128)
            plt.plot(xg,1/np.sqrt(2.*np.pi)*np.exp(-xg**2/2.),'k--')

        plt.title(self.ident)
        plt.grid()
        plt.show()
        print('MAD0: ', 1.4826*np.median(np.abs(rel_cl)))
        print('MEDIAN ABSOLUTE: ',np.median(np.abs(x)))
        print('MEDIAN NORMALIZED: ', np.median(rel_cl))
        print('MEDIAN ABSOLUTE:',np.median(x))
        print('MEDIAN THERMAL ERROR: ', np.median(err))
        print('VARIATION: ',np.std(x) )
        
    def qmetric(self):

        if self.type=='amp':
            x = self.amp
            err_x = self.sigma
        if self.type=='cphase':
            x = self.cphase
            err_x = self.sigmaCP
        if self.type=='lcamp':
            x = self.lcamp
            err_x = self.sigmaLCA
        q,dq = qmetric.qmetric(self.time,x,err_x,product=self.type)
        return q,dq

    def save_csv(self,name_out,columns='default',sep=',',header=False):
        if columns=='default':
            if self.type=='amp':
                columns=['mjd','amp','sigma']
            elif self.type=='cphase':
                columns=['mjd','cphase','sigmaCP']
            elif self.type=='lcamp':
                columns=['mjd','lcamp','sigmaLCA']    
        self.data[columns].to_csv(name_out,index=False,header=header,sep=sep)


def get_cphase(tobs,triangle,polar='none'):
    if polar != 'none':
        polvis=pol_dic[polar]+'vis'
        polsnr=pol_dic[polar]+'snr'
    else: polvis='vis'; polsnr='snr'
    #    tobs.df=tobs.df[tobs.df.polarization==polar].copy()
    baseL=list(tobs.df.baseline.unique())
    #determine order stations
    b=[triangle[0]+triangle[1],triangle[1]+triangle[2],triangle[2]+triangle[0]]   
    sign=[0,0,0]
    baseT=b
    for cou in range(3):
        if (b[cou] in baseL)&(b[cou][::-1] not in baseL):
            sign[cou]=1
        elif (b[cou] not in baseL)&(b[cou][::-1] in baseL):
            sign[cou]=-1
            baseT[cou]= b[cou][::-1]
    #print(tobs.df.columns)
    #print(baseT)
    foo=tobs.df[list(map(lambda x: x in baseT, tobs.df.baseline))]
    #print('mjd',foo.columns)
    foo=foo.groupby('mjd').filter(lambda x: len(x)==3)
    fooB0=foo[foo.baseline==baseT[0]].sort_values('mjd').copy()
    fooB1=foo[foo.baseline==baseT[1]].sort_values('mjd').copy()
    fooB2=foo[foo.baseline==baseT[2]].sort_values('mjd').copy()
    foo_out=fooB0[['time','datetime','mjd']].copy()

    foo_out['u1'] = np.asarray(fooB0['u'])
    foo_out['v1'] = np.asarray(fooB0['v'])
    foo_out['vis1'] = np.asarray(fooB0[polvis])
    if sign[0]==-1:
        foo_out['vis1'] = np.asarray(foo_out['vis1']).conj()
    foo_out['snr1'] = np.asarray(fooB0[polsnr])

    foo_out['u2'] = np.asarray(fooB1['u'])
    foo_out['v2'] = np.asarray(fooB1['v'])
    foo_out['vis2'] = np.asarray(fooB1[polvis])
    if sign[1]==-1:
        foo_out['vis2'] = np.asarray(foo_out['vis2']).conj()
    foo_out['snr2'] = np.asarray(fooB1[polsnr])

    foo_out['u3'] = np.asarray(fooB2['u'])
    foo_out['v3'] = np.asarray(fooB2['v'])
    foo_out['vis3'] = np.asarray(fooB2[polvis])
    if sign[2]==-1:
        foo_out['vis3'] = np.asarray(foo_out['vis3']).conj()
    foo_out['snr3'] = np.asarray(fooB2[polsnr])

    foo_out['cphase'] = (180./np.pi)*np.angle( foo_out['vis1']* foo_out['vis2']*foo_out['vis3'])
    foo_out['sigmaCP'] = (180./np.pi)*np.sqrt(1./foo_out['snr1']**2 + 1./foo_out['snr2']**2 + 1./foo_out['snr3']**2)
    
    return foo_out


            
def get_lcamp(tobs,quadrangle,polar='none'):
    if polar != 'none':
        polvis=pol_dic[polar]+'vis'
        polsnr=pol_dic[polar]+'snr'
    else: polvis='vis'; polsnr='snr'
    baseL=list(tobs.df.baseline.unique())
    b=[quadrangle[0]+quadrangle[1],quadrangle[2]+quadrangle[3],quadrangle[0]+quadrangle[2],quadrangle[1]+quadrangle[3]]  
    baseQ=b
    for cou in range(4):
        if (b[cou] not in baseL)&(b[cou][::-1] in baseL):
            baseQ[cou]= b[cou][::-1]
    foo=tobs.df[list(map(lambda x: (x in baseQ), tobs.df.baseline))]
    foo=foo.groupby('mjd').filter(lambda x: len(x)==4)
    fooB0=foo[foo.baseline==baseQ[0]].sort_values('mjd').copy()
    fooB1=foo[foo.baseline==baseQ[1]].sort_values('mjd').copy()
    fooB2=foo[foo.baseline==baseQ[2]].sort_values('mjd').copy()
    fooB3=foo[foo.baseline==baseQ[3]].sort_values('mjd').copy()
    foo_out=fooB0[['time','datetime','mjd']].copy()
    foo_out['u1'] = np.asarray(fooB0['u'])
    foo_out['v1'] = np.asarray(fooB0['v'])
    foo_out['vis1'] = np.asarray(fooB0[polvis])
    foo_out['snr1'] = np.asarray(fooB0[polsnr])
    foo_out['u2'] = np.asarray(fooB1['u'])
    foo_out['v2'] = np.asarray(fooB1['v'])
    foo_out['vis2'] = np.asarray(fooB1[polvis])
    foo_out['snr2'] = np.asarray(fooB1[polsnr])
    foo_out['u3'] = np.asarray(fooB2['u'])
    foo_out['v3'] = np.asarray(fooB2['v'])
    foo_out['vis3'] = np.asarray(fooB2[polvis])
    foo_out['snr3'] = np.asarray(fooB2[polsnr])
    foo_out['u4'] = np.asarray(fooB3['u'])
    foo_out['v4'] = np.asarray(fooB3['v'])
    foo_out['vis4'] = np.asarray(fooB3[polvis])
    foo_out['snr4'] = np.asarray(fooB3[polsnr])
    foo_out['lcamp'] = np.log(np.abs(foo_out['vis1'])) + np.log(np.abs(foo_out['vis2'])) - np.log(np.abs(foo_out['vis3'])) - np.log(np.abs(foo_out['vis4']))
    foo_out['sigmaLCA'] = np.sqrt(1./foo_out['snr1']**2 + 1./foo_out['snr2']**2 + 1./foo_out['snr3']**2 + 1./foo_out['snr4']**2)
    
    return foo_out



def make_df_full_cp(obs,round_s=0.1):

    """converts visibilities from obs.data to DataFrame format

    Args:
        obs: ObsData object
        round_s: accuracy of datetime object in seconds
        polarization: just label for polarization
        save_polar: what to do about different polarizations, if

    Returns:
        df: observation visibility data in DataFrame format
    """
    sour=obs.source
    df = pd.DataFrame(data=obs.data)
    df['fmjd'] = df['time']/24.
    df['mjd'] = obs.mjd + df['fmjd']
    telescopes = list(zip(df['t1'],df['t2']))
    telescopes = [(x[0],x[1]) for x in telescopes]
    df['baseline'] = [x[0]+'-'+x[1] for x in telescopes]
    df['amp'] = list(map(np.abs,df['vis']))
    df['phase'] = list(map(lambda x: (180./np.pi)*np.angle(x),df['vis']))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: round_time(x,round_s=round_s),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    #df['snr'] = df['amp']/df['sigma']
    quantities=['llamp','rramp','rlamp','lramp','llsigma','rrsigma','rlsigma','lrsigma','rrphase','llphase','rlphase','lrphase']
    for quantity in quantities:
        df[quantity] = [x[0] for x in obs.unpack(quantity)]
    df['source'] = sour
    df['baselength'] = np.sqrt(np.asarray(df.u)**2+np.asarray(df.v)**2)
    basic_columns = list(set(df.columns)-set(quantities))
    dfrr=df[basic_columns+['rramp','rrphase','rrsigma']].copy()
    dfrr['amp']=dfrr['rramp']
    dfrr['phase']=dfrr['rrphase']
    dfrr['sigma']=dfrr['rrsigma']
    dfrr=dfrr[basic_columns]
    dfrr['polarization']='RR'
    dfll=df[basic_columns+['llamp','llphase','llsigma']].copy()
    dfll['amp']=dfll['llamp']
    dfll['phase']=dfll['llphase']
    dfll['sigma']=dfll['llsigma']
    dfll=dfll[basic_columns]
    dfll['polarization']='LL'
    dflr=df[basic_columns+['lramp','lrphase','lrsigma']].copy()
    dflr['amp']=dflr['lramp']
    dflr['phase']=dflr['lrphase']
    dflr['sigma']=dflr['lrsigma']
    dflr=dflr[basic_columns]
    dflr['polarization']='LR'
    dfrl=df[basic_columns+['rlamp','rlphase','rlsigma']].copy()
    dfrl['amp']=dfrl['rlamp']
    dfrl['phase']=dfrl['rlphase']
    dfrl['sigma']=dfrl['rlsigma']
    dfrl=dfrl[basic_columns]
    dfrl['polarization']='RL'

    df = pd.concat()
    return df

def round_time(t,round_s=0.1):

    """rounding time to given accuracy

    Args:
        t: time
        round_s: delta time to round to in seconds

    Returns:
        round_t: rounded time
    """
    t0 = datetime.datetime(t.year,1,1)
    foo = t - t0
    foo_s = foo.days*24*3600 + foo.seconds + foo.microseconds*(1e-6)
    foo_s = np.round(foo_s/round_s)*round_s
    days = np.floor(foo_s/24/3600)
    seconds = np.floor(foo_s - 24*3600*days)
    microseconds = int(1e6*(foo_s - days*3600*24 - seconds))
    round_t = t0+datetime.timedelta(days,seconds,microseconds)
    return round_t


def save_all_products(pathf,path_out,special_name,get_what=['AMP','CP','LCA'],get_pol=['LL','RR'],min_elem=100.,cadence=-1,polrep='stokes'):

    if get_pol==None: get_pol=[None]
    for pol in get_pol:
        tobs = load_uvfits(pathf,tcoh=cadence,polar=pol,polrep=polrep)
        if pol==None: pol=''
        stations = list(set(''.join(tobs.df.baseline)))
        stations = [x for x in stations if x!='R']

        if 'AMP' in get_what:
            print('Saving visibility amplitudes time series...')
            if not os.path.exists(path_out+'AMP'):
                os.makedirs(path_out+'AMP') 
            baseL=sorted([x[0]+x[1] for x in itertools.combinations(stations,2)])
            for base in baseL:
                tser = tseries(tobs,base)
                if len(tser.mjd)>min_elem:
                    tser.save_csv(path_out+'AMP/'+special_name+'_'+tser.source+'_'+base+'_'+pol+'.csv')

        if 'CP' in get_what:
            print('Saving closure phase time series...')
            if not os.path.exists(path_out+'CP'):
                os.makedirs(path_out+'CP') 
            triangleL=sorted([x[0]+x[1]+x[2] for x in itertools.combinations(stations,3)])
            for tri in triangleL:
                tser = tseries(tobs,tri)
                if len(tser.mjd)>min_elem:
                    tser.save_csv(path_out+'CP/'+special_name+'_'+tser.source+'_'+tri+'_'+pol+'.csv')

        if 'LCA' in get_what:
            print('Saving log closure amplitude time series...')
            if not os.path.exists(path_out+'LCA'):
                os.makedirs(path_out+'LCA') 
            quadrangleL1=sorted([x[0]+x[1]+x[2]+x[3] for x in itertools.combinations(stations,4)])
            quadrangleL2=sorted([x[0]+x[3]+x[1]+x[2] for x in itertools.combinations(stations,4)])
            quadrangleL=quadrangleL1+quadrangleL2
            for quad in quadrangleL:
                tser = tseries(tobs,quad)
                if len(tser.mjd)>min_elem:
                    tser.save_csv(path_out+'LCA/'+special_name+'_'+tser.source+'_'+quad+'_'+pol+'.csv')