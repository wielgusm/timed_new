import sys
import numpy as np
import matplotlib.pyplot as plt
import qmetric
try:
    import ehtim as eh
except ModuleNotFoundError:
    sys.path.append('/Volumes/DATAPEN/Shared/EHT/EHTIM/eht-imaging_fork/eht-imaging/')
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


def load_uvfits(path_to_data,tcoh=-1,single_letter=True,polar=None):
    if polar=='LL':polar='L'
    if polar=='RR':polar='R'
    obs = eh.obsdata.load_uvfits(path_to_data,force_singlepol=polar)
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
            if polar=='none':
                foo = tobs.df[(tobs.df.baseline==ident) | (tobs.df.baseline==ident[1]+ident[0])]
            else:
                foo = tobs.df[(tobs.df.polarization==polar)&((tobs.df.baseline==ident) | (tobs.df.baseline==ident[1]+ident[0]))]
            #print(foo)
            self.mjd = np.asarray(foo.mjd)
            self.time = np.asarray(foo.time)
            self.amp = np.asarray(foo.amp)
            self.sigma = np.asarray(foo.sigma)
            self.data = foo

        elif product=='cphase':
            foo = get_cphase(tobs,ident,polar=polar)
            self.mjd = np.asarray(foo.mjd)
            self.time = np.asarray(foo.time)
            self.cphase = np.asarray(foo.cphase)
            self.sigmaCP = np.asarray(foo.sigmaCP)
            self.data = foo

        elif product=='lcamp':
            foo = get_lcamp(tobs,ident,polar=polar)
            self.mjd = np.asarray(foo.mjd)
            self.time = np.asarray(foo.time)
            self.lcamp = np.asarray(foo.lcamp)
            self.sigmaLCA = np.asarray(foo.sigmaLCA)
            self.data = foo

    def plot(self,line=False,figsize=''):
        if figsize=='':
            plt.figure(figsize=(10,5))
        else:
            plt.figure(figsize=figsize)
        if line: fmt='o-'
        else: fmt='o'
        plt.title(self.ident)
        if self.type=='cphase':
            plt.errorbar(self.time,self.cphase,self.sigmaCP,fmt=fmt,capsize=5)
            plt.ylabel('cphase [deg]')
        elif self.type=='amp':
            plt.errorbar(self.time,self.amp,self.sigma,fmt=fmt,capsize=5)
            plt.ylabel('amp')
        elif self.type=='lcamp':
            plt.errorbar(self.time,self.lcamp,self.sigmaLCA,fmt=fmt,capsize=5)
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
            x = np.linspace(binL, binR,128)
            plt.plot(x,1/np.sqrt(2.*np.pi)*np.exp(-x**2/2.),'k--')

        plt.title(self.ident)
        plt.grid()
        plt.show()
        print('MAD0: ', 1.4826*np.median(np.abs(rel_cl)))
        print('MAD0 ABSOLUTE: ',np.median(np.abs(x)))
        print('MEDIAN NORMALIZED: ', np.median(rel_cl))
        print('MEDIAN ABSOLUTE:',np.median(x))
        print('MEDIAN THERMAL ERROR: ', np.median(err))
        
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
        tobs.df=tobs.df[tobs.df.polarization==polar].copy()
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
    foo_out['vis1'] = np.asarray(fooB0['vis'])
    if sign[0]==-1:
        foo_out['vis1'] = np.asarray(foo_out['vis1']).conj()
    foo_out['snr1'] = np.asarray(fooB0['snr'])
    foo_out['u2'] = np.asarray(fooB1['u'])
    foo_out['v2'] = np.asarray(fooB1['v'])
    foo_out['vis2'] = np.asarray(fooB1['vis'])
    if sign[1]==-1:
        foo_out['vis2'] = np.asarray(foo_out['vis2']).conj()
    foo_out['snr2'] = np.asarray(fooB1['snr'])
    foo_out['u3'] = np.asarray(fooB2['u'])
    foo_out['v3'] = np.asarray(fooB2['v'])
    foo_out['vis3'] = np.asarray(fooB2['vis'])
    if sign[2]==-1:
        foo_out['vis3'] = np.asarray(foo_out['vis3']).conj()
    foo_out['snr3'] = np.asarray(fooB2['snr'])
    foo_out['cphase'] = (180./np.pi)*np.angle( foo_out['vis1']* foo_out['vis2']*foo_out['vis3'])
    foo_out['sigmaCP'] = (180./np.pi)*np.sqrt(1./foo_out['snr1']**2 + 1./foo_out['snr2']**2 + 1./foo_out['snr3']**2)
    
    return foo_out


            
def get_lcamp(tobs,quadrangle,polar='none'):
    
    if polar != 'none':
        tobs.df=tobs.df[tobs.df.polarization==polar].copy()
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
    foo_out['vis1'] = np.asarray(fooB0['vis'])
    foo_out['snr1'] = np.asarray(fooB0['snr'])
    foo_out['u2'] = np.asarray(fooB1['u'])
    foo_out['v2'] = np.asarray(fooB1['v'])
    foo_out['vis2'] = np.asarray(fooB1['vis'])
    foo_out['snr2'] = np.asarray(fooB1['snr'])
    foo_out['u3'] = np.asarray(fooB2['u'])
    foo_out['v3'] = np.asarray(fooB2['v'])
    foo_out['vis3'] = np.asarray(fooB2['vis'])
    foo_out['snr3'] = np.asarray(fooB2['snr'])
    foo_out['u4'] = np.asarray(fooB3['u'])
    foo_out['v4'] = np.asarray(fooB3['v'])
    foo_out['vis4'] = np.asarray(fooB3['vis'])
    foo_out['snr4'] = np.asarray(fooB3['snr'])
    foo_out['lcamp'] = np.log(np.abs(foo_out['vis1'])) + np.log(np.abs(foo_out['vis2'])) - np.log(np.abs(foo_out['vis3'])) - np.log(np.abs(foo_out['vis4']))
    foo_out['sigmaLCA'] = np.sqrt(1./foo_out['snr1']**2 + 1./foo_out['snr2']**2 + 1./foo_out['snr3']**2 + 1./foo_out['snr4']**2)
    
    return foo_out


#def synthetic_L