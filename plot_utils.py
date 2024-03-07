import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mpllines
import h5py as h5
import corner
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Interped, Uniform, LogUniform
from scipy.stats import gaussian_kde
from pandas.core.frame import DataFrame
import pickle
import matplotlib
from tqdm import tqdm
from scipy.interpolate import interp1d

from model_libs_r2 import hyper_Double, PS_mass, Double_mass_pair_branch, Single_mass_pair_un, Event_probs, Double_priors,PLP_m

def spin_a(a1,mu_a,sigma_a,amin,amax):
    #sigma_a=sqr_sig_a**0.5
    return TG(mu_a,sigma_a,amin,amax).prob(a1)

def twopeak_a(a1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amin,amax):
    return TG(mu_a1,sigma_a1,amin,amax).prob(a1)*(1-rp)+TG(mu_a3,sigma_a3,amin,amax).prob(a1)*rp
####################################################################################
# calculate population-informed samples
####################################################################################

def pop_informed_samples(posteriors,post,size,filename):
    n_catalogs=size
    if 'log_prior' in post.keys():
        post.pop('log_prior')
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=n_catalogs)
    parameters={key:np.array(post[key])[indx] for key in post.keys()}
    parameters.pop('log_likelihood')
    
    resampled_m1 = np.zeros((len(posteriors),n_catalogs))
    resampled_a1 = np.zeros((len(posteriors),n_catalogs))
    resampled_ct1 = np.zeros((len(posteriors),n_catalogs))
    resampled_m2 = np.zeros((len(posteriors),n_catalogs))
    resampled_a2 = np.zeros((len(posteriors),n_catalogs))
    resampled_ct2 = np.zeros((len(posteriors),n_catalogs))
    
    # Loop across catalog realizations
    for i in tqdm(range(n_catalogs)):
        #print('No.{} of {} sets'.format(i,n_catalogs))
        # Select a random population sample
        para={key:parameters[key][i] for key in parameters.keys()}
        for ii in np.arange(len(posteriors)):
            samples = posteriors[ii]
            samples = {key:np.array(samples[key]) for key in samples.keys()}
            probs =hyper_Double(samples,**para)/samples['prior']
            probs /= np.sum(probs)
            chosenInd = np.random.choice(np.arange(len(probs)),p=probs)
            resampled_m1[ii,i] = samples['m1'][chosenInd]
            resampled_m2[ii,i] = samples['m2'][chosenInd]
            resampled_a1[ii,i] = samples['a1'][chosenInd]
            resampled_a2[ii,i] = samples['a2'][chosenInd]
            resampled_ct1[ii,i] = samples['cos_tilt_1'][chosenInd]
            resampled_ct2[ii,i] = samples['cos_tilt_2'][chosenInd]
            
    informed_samples=[]
    for ii in np.arange(len(posteriors)):
        informed_samples.append(DataFrame({'m1':resampled_m1[ii], 'm2':resampled_m2[ii], 'a1':resampled_a1[ii], 'a2':resampled_a2[ii], 'cos_tilt_1':resampled_ct1[ii], 'cos_tilt_2':resampled_ct1[ii]}, \
        columns=['m1', 'm2', 'a1', 'a2', 'cos_tilt_1', 'cos_tilt_2']))
    with open(filename, 'wb') as fp:
        pickle.dump((informed_samples), fp)

####################################################################################
# plot population-informed samples
####################################################################################

def plot_ma_sample(samples,size,filename):
    posteriors=samples
    n_catalogs=len(samples[0]['m1'])
    resampled_m1 = np.zeros((len(posteriors),n_catalogs))
    resampled_a1 = np.zeros((len(posteriors),n_catalogs))
    resampled_ct1 = np.zeros((len(posteriors),n_catalogs))
    resampled_m2 = np.zeros((len(posteriors),n_catalogs))
    resampled_a2 = np.zeros((len(posteriors),n_catalogs))
    resampled_ct2 = np.zeros((len(posteriors),n_catalogs))
    for ii in np.arange(len(posteriors)):
        resampled_m1[ii] = samples[ii]['m1']
        resampled_m2[ii] = samples[ii]['m2']
        resampled_a1[ii] = samples[ii]['a1']
        resampled_a2[ii] = samples[ii]['a2']
        resampled_ct1[ii] = samples[ii]['cos_tilt_1']
        resampled_ct2[ii] = samples[ii]['cos_tilt_2']

    fig,ax = plt.subplots(figsize=(10,5))
    ax.set_rasterization_zorder(1)
    # Grid over chi_eff and q
    m1_grid = np.linspace(2,100,1200)
    a1_grid = np.linspace(0,1,119)
    dm1 = m1_grid[1] - m1_grid[0]
    da1 = a1_grid[1] - a1_grid[0]
    m2_grid=m1_grid
    a2_grid=a1_grid
    dm2=dm1
    da2=da1

    X,Q = np.meshgrid(m1_grid,a1_grid)
    print('ploting')
    for i in tqdm(range(resampled_a1.shape[0])):
 
        # For each event, read out q and chi_eff samples
        m1s = resampled_m1[i,:]
        a1s = resampled_a1[i,:]

        # Set up KDE
        m1_a1_kde = gaussian_kde([m1s,a1s])

        # Evaluate KDE over our grid above
        # Note that we want symmetric boundary conditions at q=1
        heights = m1_a1_kde([X.reshape(-1),Q.reshape(-1)])
        heights = heights.reshape(X.shape)

        # Compute 1D cdf to find contours enclosing 90% of probability
        heights /= np.sum(heights)*dm1*da1
        heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
        cdf = np.cumsum(heights_large_to_small)*dm1*da1
        h90 = np.interp(0.9,cdf,heights_large_to_small)

        ax.scatter([np.median(m1s)],[np.median(a1s)],color='black',zorder=100,s=3)
        #ax.contour(m1_grid,a1_grid,heights,levels=(h90,np.inf),colors='black',linewidths=0.3,alpha=0.2,zorder=0)
        ax.contourf(m1_grid,a1_grid,heights,levels=(h90,np.inf),colors='#08519c',alpha=0.075,zorder=0)
        
    for i in tqdm(range(resampled_a2.shape[0])):

        # For each event, read out q and chi_eff samples
        m2s = resampled_m2[i,:]
        a2s = resampled_a2[i,:]
        m2_a2_kde = gaussian_kde([m2s,a2s])

        # Evaluate KDE over our grid above
        # Note that we want symmetric boundary conditions at q=1
        heights = m2_a2_kde([X.reshape(-1),Q.reshape(-1)])
        heights = heights.reshape(X.shape)

        # Compute 1D cdf to find contours enclosing 90% of probability
        heights /= np.sum(heights)*dm2*da2
        heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
        cdf = np.cumsum(heights_large_to_small)*dm2*da2
        h90 = np.interp(0.9,cdf,heights_large_to_small)

        ax.scatter([np.median(m2s)],[np.median(a2s)],color='orange',zorder=100,s=3)
        #ax.contour(m2_grid,a2_grid,heights,levels=(h90,np.inf),colors='red',linewidths=0.3,alpha=0.2,zorder=0)
        ax.contourf(m2_grid,a2_grid,heights,levels=(h90,np.inf),colors='#08519c',alpha=0.075,zorder=0)
    ax.set_xlabel(r'$m_\mathrm{1,2}[M_{\odot}]$',fontsize=18)
    ax.set_ylabel(r'$a_\mathrm{1,2}$',fontsize=18)
    ax.xaxis.grid(True,which='major',ls=':')
    ax.yaxis.grid(True,which='major',ls=':')
    ax.set_axisbelow(True)
    plt.savefig(filename,bbox_inches='tight',dpi=200)
    
####################################################################
#plot mass and spin distributions
####################################################################
def DF_ct(ct,sigma_t,zeta,zmin=-1,mu_t=1):
    return Uniform(-1,1).prob(ct)*(1-zeta)+TG(mu_t,sigma_t,zmin,1).prob(ct)*zeta
    
def calculated_fneg(post,iidct=False):
    parameters={}
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(post[key])})
    pct=[]
    ct_sam=np.linspace(-1,0,500)
    for i in tqdm(np.arange(len(parameters['sigma_t']))):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pneg=np.sum(pct,axis=-1)/500.
    post['f_neg']=pneg
    if iidct:
        parameters={}
        keys=['sigma_t2','zeta2','zmin2','mu_t2']
        for key in keys:
            parameters.update({key:np.array(post[key])})
        pct=[]
        ct_sam=np.linspace(-1,0,500)
        for i in tqdm(np.arange(len(parameters['sigma_t2']))):
            para_ct=[parameters[key][i] for key in keys]
            pct.append(DF_ct(ct_sam,*para_ct))
        pct=np.array(pct)
        pneg=np.sum(pct,axis=-1)/500.
        post['f_neg2']=pneg
    return post
    
import h5py

def plot_iidct(ax2,post,colors,indx,size):
    ax2.cla()
    parameters={}
    keys1=['sigma_t','zeta','zmin','mu_t']
    keys2=['sigma_t2','zeta2','zmin2','mu_t2']
    for key in keys1+keys2:
        parameters.update({key:np.array(post[key])[indx]})
    pct1=[]
    pct2=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(size)):
        para_ct1=[parameters[key][i] for key in keys1]
        pct1.append(DF_ct(ct_sam,*para_ct1))
        para_ct2=[parameters[key][i] for key in keys2]
        pct2.append(DF_ct(ct_sam,*para_ct2))
    pct1=np.array(pct1)
    pct1_pup=np.percentile(pct1,95,axis=0)
    pct1_plow=np.percentile(pct1,5,axis=0)
    pct1_pmid=np.percentile(pct1,50,axis=0)
    pct1_pmean=np.mean(pct1,axis=0)
    pct2=np.array(pct2)
    pct2_pup=np.percentile(pct2,95,axis=0)
    pct2_plow=np.percentile(pct2,5,axis=0)
    pct2_pmid=np.percentile(pct2,50,axis=0)
    pct2_pmean=np.mean(pct2,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct1_plow,pct1_pup,color=colors[0],alpha=0.2,label='low spin')
    ax2.plot(ct_sam,pct1_pmid,color=colors[0],alpha=0.8)
    ax2.fill_between(ct_sam,pct2_plow,pct2_pup,color=colors[1],alpha=0.2,label='high spin')
    ax2.plot(ct_sam,pct2_pmid,color=colors[1],alpha=0.8)
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel('PDF')
    ax2.set_xlim(-1,1)
    return ax2
    
def plot_linear_a(ax3,post,colors,indx,size):
    ax3.cla()
    parameters={}
    keys=['mu_al1','mu_ar1','sigma_a1','mu_al2','mu_ar2','sigma_a2','amin2','amin1','amax1','amax2']
    keys1a=['mu_al1','sigma_a1','amin1','amax1']
    keys2a=['mu_al2','sigma_a2','amin2','amax2']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in tqdm(np.arange(size)):
        para1a=[parameters[key][i] for key in keys1a]
        para2a=[parameters[key][i] for key in keys2a]
        a_1G.append(spin_a(a_sam,*para1a))
        a_2G.append(spin_a(a_sam,*para2a))
    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,95,axis=0)
    a_1G_plow=np.percentile(a_1G,5,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,95,axis=0)
    a_2G_plow=np.percentile(a_2G,5,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)

    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.2,label='LS left')
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.2,label='HS left')
    
    keys1a=['mu_ar1','sigma_a1','amin1','amax1']
    keys2a=['mu_ar2','sigma_a2','amin2','amax2']
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in tqdm(np.arange(size)):
        para1a=[parameters[key][i] for key in keys1a]
        para2a=[parameters[key][i] for key in keys2a]
        a_1G.append(spin_a(a_sam,*para1a))
        a_2G.append(spin_a(a_sam,*para2a))
    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,95,axis=0)
    a_1G_plow=np.percentile(a_1G,5,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,95,axis=0)
    a_2G_plow=np.percentile(a_2G,5,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)
        

    ax3.plot(a_sam,a_1G_plow,color=colors[0],alpha=1,lw=1,label='LS right')
    ax3.plot(a_sam,a_1G_pup,color=colors[0],alpha=1,lw=1)
    ax3.plot(a_sam,a_2G_plow,color=colors[1],alpha=1,lw=1,label='HS right')
    ax3.plot(a_sam,a_2G_pup,color=colors[1],alpha=1,lw=1)
    return ax3

import matplotlib.gridspec as gridspec
def plot_ma_dist(post,colors,size,filename,plot_LIGO=None,iidct=False,linear_a=False,twopeak=False,plot_inj=False):

    fig=plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(5, 2)
    ax1 = fig.add_subplot(gs[:3,:])
    ax3 = fig.add_subplot(gs[3:,0])
    ax2 = fig.add_subplot(gs[3:,1])

    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2']
    keys1=['alpha1','mmin1','mmax1','delta1']
    keys2=['alpha2','mmin2','mmax2','delta2']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))
    for i in np.arange(12):
        keys.append('o'+str(i+1))
        keys2.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m_1G=[]
    m_2G=[]
    x=np.linspace(2,100,5000)
    m1_sam=x
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        para1=[parameters[key][i] for key in keys1]
        para2=[parameters[key][i] for key in keys2]
        m_1G.append(PS_mass(x,*para1)*(1-para['r2']))
        m_2G.append(PS_mass(x,*para2)*para['r2'])
    m_1G=np.array(m_1G)
    m_1G_pup=np.percentile(m_1G,95,axis=0)
    m_1G_plow=np.percentile(m_1G,5,axis=0)
    m_1G_pmid=np.percentile(m_1G,50,axis=0)
    m_1G_pmean=np.mean(m_1G,axis=0)
    m_2G=np.array(m_2G)
    m_2G_pup=np.percentile(m_2G,95,axis=0)
    m_2G_plow=np.percentile(m_2G,5,axis=0)
    m_2G_pmid=np.percentile(m_2G,50,axis=0)
    m_2G_pmean=np.mean(m_2G,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    #plt.fill_between(m1_sam,plow,pup,color=colors[0],alpha=0.4,label='total')
    #plt.plot(m1_sam,pmean,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_1G_plow,m_1G_pup,color=colors[0],alpha=0.2,label='low spin')
    ax1.plot(m1_sam,m_1G_plow,color=colors[0],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_1G_pup,color=colors[0],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_1G_pmean,color=colors[0],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_2G_plow,m_2G_pup,color=colors[1],alpha=0.2,label='high spin')
    ax1.plot(m1_sam,m_2G_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_2G_pup,color=colors[1],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_2G_pmean,color=colors[1],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_2G_pmid,color=colors[1],alpha=0.9)

    ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_ylim(5e-5,5e-1)
    ax1.set_xlim(2,100)
    ax1.set_xlabel(r'$m/M_{\odot}$')
    ax1.set_ylabel(r'$p(m)$')
    ax1.legend(loc=0)

    parameters={}
    keys=['mu_a1','sigma_a1','mu_a2','sigma_a2','amin2','amin1','amax1','amax2']
    keys1a=['mu_a1','sigma_a1','amin1','amax1']
    if twopeak:
        keys=['mu_a1','sigma_a1','mu_a3','sigma_a3','rp','mu_a2','sigma_a2','amin2','amin1','amax1','amax2']
        keys1a=['mu_a1','sigma_a1','mu_a3','sigma_a3','rp','amin1','amax1']
    keys2a=['mu_a2','sigma_a2','amin2','amax2']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in tqdm(np.arange(size)):
        para1a=[parameters[key][i] for key in keys1a]
        para2a=[parameters[key][i] for key in keys2a]
        if twopeak:
            a_1G.append(twopeak_a(a_sam,*para1a))
        else:
            a_1G.append(spin_a(a_sam,*para1a))
        a_2G.append(spin_a(a_sam,*para2a))
    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,95,axis=0)
    a_1G_plow=np.percentile(a_1G,5,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,95,axis=0)
    a_2G_plow=np.percentile(a_2G,5,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)

    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.2,label='low spin')
    #ax3.plot(a_sam,a_1G_pmean,color=colors[0],alpha=0.8)
    ax3.plot(a_sam,a_1G_pmid,color=colors[0],alpha=0.8)
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.2,label='high spin')
    #ax3.plot(a_sam,a_2G_pmean,color=colors[1],alpha=0.8)
    ax3.plot(a_sam,a_2G_pmid,color=colors[1],alpha=0.8)
    if linear_a:
        ax3=plot_linear_a(ax3,post,colors,indx,size)
    ax3.set_xlabel(r'$a$')
    ax3.set_ylabel(r'$p(a)$')
    ax3.set_xlim(0,1)
    #ax3.set_ylim(0,5)
    ax3.legend(loc=0)
    
    parameters={}
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    pct=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.2)
    #ax2.plot(ct_sam,pct_pmean,color=colors[2],alpha=0.8)
    ax2.plot(ct_sam,pct_pmid,color=colors[2],alpha=0.8)
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel(r'$p(\cos\theta)$')
    ax2.set_xlim(-1,1)
    #ax2.set_ylim(0,1.8)
    if plot_LIGO:
        print('plotting LIGO results')
        ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.2,label='This work')
        ax2=plot_LIGO_ct(ax2)
        ax2.legend(loc=0)
    if iidct:
        ax2=plot_iidct(ax2,post,colors,indx,size)
        ax2.legend(loc=0)
    if plot_inj:
        ax2.plot(ct_sam,DF_ct(ct_sam,sigma_t=0.8,zeta=0.7),color='black',label='injection')
        ax3.plot(a_sam,spin_a(a_sam,mu_a=0.15,sigma_a=0.1,amin=0,amax=1),color='black',label='injection')
        ax3.plot(a_sam,spin_a(a_sam,mu_a=0.7,sigma_a=0.1,amin=0,amax=1),color='black')
        ax1.plot(m1_sam,PLP_m(m1_sam,alpha=2.15,mmin=4,mmax=85,delta=6,mu=30,sigma=5,r_peak=0.05),color='black',label='injection')
    
    plt.tight_layout()
    plt.savefig(filename)

def plot_m1_m2_dist(post,colors,size,filename):

    fig=plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2','beta','lgR0']
    keys1=['alpha1','mmin1','mmax1','delta1','alpha2','r2','mmin2','mmax2','delta2','beta']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))
    for i in np.arange(12):
        keys.append('o'+str(i+1))
        keys1.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m1_sam=np.linspace(2,100,500)
    m2_sam=np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    pm1_1G=[]
    pm1_2G=[]
    pm2_1G=[]
    pm2_2G=[]
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        R0=10**parameters['lgR0'][i]
        para1={key:parameters[key][i] for key in keys1}
        p11,p12,p21,p22 = Double_mass_pair_branch(x,y,**para1)
        pm1_1G.append(np.sum((p11+p12)*dy,axis=0)*R0)
        pm1_2G.append(np.sum((p21+p22)*dy,axis=0)*R0)
        pm2_1G.append(np.sum((p11+p21)*dx,axis=1)*R0)
        pm2_2G.append(np.sum((p12+p22)*dx,axis=1)*R0)
    pm1_1G=np.array(pm1_1G)
    pm1_1G_pup=np.percentile(pm1_1G,95,axis=0)
    pm1_1G_plow=np.percentile(pm1_1G,5,axis=0)
    pm1_1G_pmid=np.percentile(pm1_1G,50,axis=0)
    pm1_1G_pmean=np.mean(pm1_1G,axis=0)
    pm1_2G=np.array(pm1_2G)
    pm1_2G_pup=np.percentile(pm1_2G,95,axis=0)
    pm1_2G_plow=np.percentile(pm1_2G,5,axis=0)
    pm1_2G_pmid=np.percentile(pm1_2G,50,axis=0)
    pm1_2G_pmean=np.mean(pm1_2G,axis=0)
    pm1=pm1_1G+pm1_2G
    pm1_pup=np.percentile(pm1,95,axis=0)
    pm1_plow=np.percentile(pm1,5,axis=0)
    pm1_pmid=np.percentile(pm1,50,axis=0)
    pm1_pmean=np.mean(pm1,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    ax1.fill_between(m1_sam,pm1_1G_plow,pm1_1G_pup,color=colors[0],alpha=0.15,label='low spin')
    #ax1.plot(m1_sam,pm1_1G_plow,color=colors[0],alpha=0.8,lw=0.3)
    #ax1.plot(m1_sam,pm1_1G_pup,color=colors[0],alpha=0.8,lw=0.3)
    ax1.plot(m1_sam,pm1_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,pm1_2G_plow,pm1_2G_pup,color=colors[1],alpha=0.15,label='high spin')
    #ax1.plot(m1_sam,pm1_2G_plow,color=colors[1],alpha=0.8,lw=0.3)
    #ax1.plot(m1_sam,pm1_2G_pup,color=colors[1],alpha=0.8,lw=0.3)
    ax1.plot(m1_sam,pm1_2G_pmid,color=colors[1],alpha=0.9)
        
    #ax1.fill_between(m1_sam,pm1_plow,pm1_pup,color=colors[2],alpha=0.15,label='HM')
    ax1.plot(m1_sam,pm1_plow,color=colors[2],alpha=0.8,lw=1.5,label='Over all')
    ax1.plot(m1_sam,pm1_pup,color=colors[2],alpha=0.8,lw=1.5)
    ax1.plot(m1_sam,pm1_pmid,color=colors[2],alpha=0.9,ls=':')

    ax1.set_yscale('log')
    ax1.set_ylim(1e-4,3e1)
    ax1.set_xlim(0,100)
    ax1.set_xlabel(r'$m_{1}/M_{\odot}$')
    ax1.set_ylabel(r'$\frac{{\rm d}\mathcal{R}(z=0)}{{\rm d}m_{1}}~[{\rm Gpc}^{-3}~{\rm yr}^{-1}~M_{\odot}^{-1}]$')
    ax1.legend(loc='upper right')
    
    
    pm2_1G=np.array(pm2_1G)
    pm2_1G_pup=np.percentile(pm2_1G,95,axis=0)
    pm2_1G_plow=np.percentile(pm2_1G,5,axis=0)
    pm2_1G_pmid=np.percentile(pm2_1G,50,axis=0)
    pm2_1G_pmean=np.mean(pm2_1G,axis=0)
    pm2_2G=np.array(pm2_2G)
    pm2_2G_pup=np.percentile(pm2_2G,95,axis=0)
    pm2_2G_plow=np.percentile(pm2_2G,5,axis=0)
    pm2_2G_pmid=np.percentile(pm2_2G,50,axis=0)
    pm2_2G_pmean=np.mean(pm2_2G,axis=0)
    
    pm2=pm2_1G+pm2_2G
    pm2_pup=np.percentile(pm2,95,axis=0)
    pm2_plow=np.percentile(pm2,5,axis=0)
    pm2_pmid=np.percentile(pm2,50,axis=0)
    pm2_pmean=np.mean(pm2,axis=0)

    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')
    ax2.fill_between(m2_sam,pm2_1G_plow,pm2_1G_pup,color=colors[0],alpha=0.2,label='low spin')
    #ax2.plot(m2_sam,pm2_1G_plow,color=colors[0],alpha=0.8,lw=0.3)
    #ax2.plot(m2_sam,pm2_1G_pup,color=colors[0],alpha=0.8,lw=0.3)
    ax2.plot(m2_sam,pm2_1G_pmid,color=colors[0],alpha=0.9)
    ax2.fill_between(m2_sam,pm2_2G_plow,pm2_2G_pup,color=colors[1],alpha=0.2,label='high spin')
    #ax2.plot(m2_sam,pm2_2G_plow,color=colors[1],alpha=0.8,lw=0.3)
    #ax2.plot(m2_sam,pm2_2G_pup,color=colors[1],alpha=0.8,lw=0.3)
    ax2.plot(m2_sam,pm2_2G_pmid,color=colors[1],alpha=0.9)
            
    #ax2.fill_between(m2_sam,pm2_plow,pm2_pup,color=colors[2],alpha=0.15,label='HM')
    ax2.plot(m2_sam,pm2_plow,color=colors[2],alpha=0.8,lw=1.5,label='Over all')
    ax2.plot(m2_sam,pm2_pup,color=colors[2],alpha=0.8,lw=1.5)
    ax2.plot(m2_sam,pm2_pmid,color=colors[2],alpha=0.9,ls=':')


    ax2.set_yscale('log')
    ax2.set_ylim(1e-4,3e1)
    ax2.set_xlim(0,100)
    ax2.set_xlabel(r'$m_{2}/M_{\odot}$')
    ax2.set_ylabel(r'$\frac{{\rm d}\mathcal{R}(z=0)}{{\rm d}m_{2}}~[{\rm Gpc}^{-3}~{\rm yr}^{-1}~M_{\odot}^{-1}]$')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(filename)

####################################################################
#plot mmax and kick velocity
####################################################################
def plot_mmax(post,filename):
    from scipy.stats import gaussian_kde
    kde=gaussian_kde(post['mmax1'])
    hs=np.linspace(30,100,1000)
    pdfs=kde(hs)
    midindx=np.argmax(pdfs)
    m=hs[midindx]
    cdf=0
    i=1
    j=1
    while cdf<0.683:
        if pdfs[midindx+i]>pdfs[midindx-j]:
            cdf+=pdfs[midindx+i]*70/1000.
            i+=1
        else:
            cdf+=pdfs[midindx-j]*70/1000.
            j+=1
    low68=hs[midindx-j]
    high68=hs[midindx+i]
    while cdf<0.954:
        if pdfs[midindx+i]>pdfs[midindx-j]:
            cdf+=pdfs[midindx+i]*70/1000.
            i+=1
        else:
            cdf+=pdfs[midindx-j]*70/1000.
            j+=1
    low95=hs[midindx-j]
    high95=hs[midindx+i]

    fig,ax = plt.subplots(figsize=(4,3))
    #ax.xaxis.grid(True,which='major',ls=':',color='grey')
    #ax.yaxis.grid(True,which='major',ls=':',color='grey')
    ax.hist(post['mmax1'],bins=20,density=1,color='#88CCEE',linewidth=1.0,alpha=0.2)
    ax.hist(post['mmax1'],bins=20,density=1,color='#88CCEE',histtype='step',linewidth=1.0,alpha=0.8)
    #ax.axvline([np.percentile(post['mmax1'],90)],ls='dashed',color='black')
    ax.plot(hs,pdfs,linewidth=1.0,alpha=0.8)
    ax.axvline(low68, ls='dotted',color='#88CCEE',alpha=1,linewidth=1.5)
    ax.axvline(high68, ls='dotted',color='#88CCEE',alpha=1,linewidth=1.5)
    #ax.axvline(low95, ls='dashed',color='#88CCEE',alpha=0.8,linewidth=1.5)
    #ax.axvline(high95, ls='dashed',color='#88CCEE',alpha=0.8,linewidth=1.5)
    ax.set_xlabel(r'$m_{\rm max,1}[M_{\odot}]$',labelpad=5,fontsize=13)
    ax.set_ylabel('PDF',fontsize=13)
    print(r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(m,m-low95,high95-m))
    print(r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(m,m-low68,high68-m))
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')
    
####################################################################
#plot corners
####################################################################

def plot_corner(post,params,show_keys,color,filename,smooth=1.,bins=25):
    print('ploting')
    data2=np.array([np.array(post[key]) for key in params])
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
    c1=color
    ranges=[[min(data2[i]),max(data2[i])] for i in np.arange(len(data2))]
    percentiles=[[np.percentile(data2[i],5),np.percentile(data2[i],50),np.percentile(data2[i],95)] for i in np.arange(len(data2))]
    titles=[r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(percentiles[i][1],percentiles[i][1]-percentiles[i][0], percentiles[i][2]-percentiles[i][1]) for i in np.arange(len(data2)) ]
    kwargs = dict(title_kwargs=dict(fontsize=15), labels=show_keys, smooth=smooth, bins=bins,  quantiles=[0.05,0.5,0.95], range=ranges,\
    levels=levels, show_titles=True, titles=None, plot_density=False, plot_datapoints=True, fill_contours=True, title_qs=[0.05,0.95],\
    label_kwargs=dict(fontsize=20), max_n_ticks=1, alpha=0.5, hist_kwargs=dict(color=c1))
    groupdata=[data2]
    plt.cla()
    fig = corner.corner(groupdata[0].T, color=c1, **kwargs)
    lines = [mpllines.Line2D([0], [0], color=c1)]
    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    #axes[ndim - 1].legend(lines, labels, fontsize=14)
    plt.savefig(filename)

####################################################################
#plot mu sigma for linearly changed spin distribution
####################################################################
def plot_mu_sigma(post,keys,labels,color,filename,lims=[0,1]):

    beta_1=np.array(post[keys[0]])
    beta_2=np.array(post[keys[1]])

    # Plotting bounds
    x_min = lims[0]
    x_max = lims[1]
    y_min = lims[0]
    y_max = lims[1]

    # Create a linear colormap between white and the "Broken PL" model color
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",color])

    # Plot 2D alpha1 vs. alpha2 posterior
    fig,ax = plt.subplots(figsize=(4,3))
    ax.hexbin(beta_1,beta_2,cmap=cmap,gridsize=25,mincnt=1,extent=(x_min,x_max,y_min,y_max),linewidths=(0.,))

    # The next chunk of code creates contours
    # First construct a KDE and evaluate it on a grid across alpha1 vs. alpha2 space
    kde = gaussian_kde([beta_1,beta_2])
    x_gridpoints = np.linspace(x_min,x_max,60)
    y_gridpoints = np.linspace(y_min,y_max,59)
    x_grid,y_grid = np.meshgrid(x_gridpoints,y_gridpoints)
    z_grid = kde([x_grid.reshape(-1),y_grid.reshape(-1)]).reshape(y_gridpoints.size,x_gridpoints.size)

    # Sort the resulting z-values to get estimates of where to place 50% and 90% contours
    sortedVals = np.sort(z_grid.flatten())[::-1]
    cdfVals = np.cumsum(sortedVals)/np.sum(sortedVals)
    i50 = np.argmin(np.abs(cdfVals - 0.50))
    i90 = np.argmin(np.abs(cdfVals - 0.90))
    val50 = sortedVals[i50]
    val90 = sortedVals[i90]

    # Draw contours at these locations
    CS = ax.contour(x_gridpoints,y_gridpoints,z_grid,levels=(val90,val50),linestyles=('dashed','solid'),colors='k',linewidths=1)

    # Draw a diagonal line for illustration purposes
    ax.plot(np.arange(-1,2),np.arange(-1,2),lw=1,ls='--',color='black',alpha=0.75)

    # Misc formatting
    ax.grid(True,dashes=(1,3))
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel(labels[0],fontsize=14)
    ax.set_ylabel(labels[1],fontsize=14)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')

####################################################################
#plot likelihood variance distritbution 
####################################################################

def plot_vars(data2,filename,show_keys,c1):
    print('ploting')
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
    ranges=[[min(data2[i]),max(data2[i])] for i in np.arange(len(data2))]
    percentiles=[[np.percentile(data2[i],5),np.percentile(data2[i],50),np.percentile(data2[i],95)] for i in np.arange(len(data2))]
    titles=[r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(percentiles[i][1],percentiles[i][1]-percentiles[i][0], percentiles[i][2]-percentiles[i][1]) for i in np.arange(len(data2)) ]
    kwargs = dict(title_kwargs=dict(fontsize=15), labels=show_keys, smooth=1., bins=25,  quantiles=[0.05,0.5,0.95], range=ranges,\
    levels=levels, show_titles=True, plot_density=False, plot_datapoints=True, fill_contours=True, title_qs=[0.05,0.95],\
    label_kwargs=dict(fontsize=20), max_n_ticks=1, alpha=0.5, hist_kwargs=dict(color=c1))
    groupdata=[data2]
    plt.cla()
    fig = corner.corner(groupdata[0].T, color=c1, **kwargs)
    lines = [mpllines.Line2D([0], [0], color=c1)]
    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    #axes[ndim - 1].legend(lines, labels, fontsize=20)
    plt.savefig(filename)


####################################################################################
# plot population-informed chi_eff v.s. q figure
####################################################################################
GWTC3=['GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158', 'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043', 'GW170818_022509', 'GW170823_131358', 'GW190408_181802', 'GW190412_053044', 'GW190413_134308', 'GW190421_213856', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428', 'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190521_074359', 'GW190527_092055', 'GW190602_175927', 'GW190620_030421', 'GW190630_185205', 'GW190701_203306', 'GW190706_222641', 'GW190707_093326', 'GW190708_232457', 'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190803_022701', 'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190925_232845', 'GW190929_012149', 'GW190930_133541', 'GW190413_052954', 'GW190719_215514', 'GW190725_174728', 'GW190731_140936', 'GW190805_211137', 'GW191105_143521', 'GW191109_010717', 'GW191127_050227', 'GW191129_134029', 'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW191230_180458', 'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 'GW200208_130117', 'GW200209_085452', 'GW200219_094415', 'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200311_115853', 'GW200316_215756', 'GW191103_012549', 'GW200216_220804']

G2_events=['GW170729_185629','GW190517_055101','GW190519_153544','GW190521_030229',
'GW190602_175927','GW190620_030421','GW190701_203306','GW190706_222641',
'GW190929_012149','GW190805_211137','GW191109_010717']

def plot_qchieff_sample(samples,size,filename,event_list=GWTC3,G2_list=G2_events):
    posteriors=samples
    n_catalogs=len(samples[0]['m1'])
    resampled_m1 = np.zeros((len(posteriors),n_catalogs))
    resampled_a1 = np.zeros((len(posteriors),n_catalogs))
    resampled_ct1 = np.zeros((len(posteriors),n_catalogs))
    resampled_m2 = np.zeros((len(posteriors),n_catalogs))
    resampled_a2 = np.zeros((len(posteriors),n_catalogs))
    resampled_ct2 = np.zeros((len(posteriors),n_catalogs))
    resampled_q = np.zeros((len(posteriors),n_catalogs))
    resampled_chieff = np.zeros((len(posteriors),n_catalogs))
    for ii in np.arange(len(posteriors)):
        resampled_m1[ii] = samples[ii]['m1']
        resampled_m2[ii] = samples[ii]['m2']
        resampled_a1[ii] = samples[ii]['a1']
        resampled_a2[ii] = samples[ii]['a2']
        resampled_ct1[ii] = samples[ii]['cos_tilt_1']
        resampled_ct2[ii] = samples[ii]['cos_tilt_2']
        resampled_q[ii] = samples[ii]['m2']/samples[ii]['m1']
        resampled_chieff[ii] = (samples[ii]['m2']*samples[ii]['a2']*samples[ii]['cos_tilt_2']+samples[ii]['m1']*\
            samples[ii]['a1']*samples[ii]['cos_tilt_1'])/(samples[ii]['m2']+samples[ii]['m1'])

    fig,ax = plt.subplots(figsize=(6,5))
    ax.set_rasterization_zorder(1)
    # Grid over chi_eff and q
    chieff_grid = np.linspace(-1,1,1200)
    q_grid = np.linspace(0,1,119)
    dchieff = chieff_grid[1] - chieff_grid[0]
    dq = q_grid[1] - q_grid[0]

    X,Q = np.meshgrid(chieff_grid,q_grid)
    print('ploting')
    for i in tqdm(range(resampled_q.shape[0])):
        if event_list[i] in G2_list:
            color1='orange'
            color2='green'
        else:
            color1='black'
            color2='#08519c'

        # For each event, read out q and chi_eff samples
        chieffs = resampled_chieff[i,:]
        qs = resampled_q[i,:]

        # Set up KDE
        chieff_q_kde = gaussian_kde([chieffs,qs])

        # Evaluate KDE over our grid above
        # Note that we want symmetric boundary conditions at q=1
        heights = chieff_q_kde([X.reshape(-1),Q.reshape(-1)])
        heights = heights.reshape(X.shape)

        # Compute 1D cdf to find contours enclosing 90% of probability
        heights /= np.sum(heights)*dchieff*dq
        heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
        cdf = np.cumsum(heights_large_to_small)*dchieff*dq
        h90 = np.interp(0.9,cdf,heights_large_to_small)

        ax.scatter([np.median(chieffs)],[np.median(qs)],color=color1,zorder=100,s=3)
        #ax.contour(m1_grid,a1_grid,heights,levels=(h90,np.inf),colors='black',linewidths=0.3,alpha=0.2,zorder=0)
        ax.contourf(chieff_grid,q_grid,heights,levels=(h90,np.inf),colors=color2,alpha=0.075,zorder=0)
        
    ax.set_xlabel(r'$\chi_\mathrm{eff}$',fontsize=18)
    ax.set_ylabel(r'$q$',fontsize=18)
    ax.xaxis.grid(True,which='major',ls=':')
    ax.yaxis.grid(True,which='major',ls=':')
    ax.set_axisbelow(True)
    plt.savefig(filename,bbox_inches='tight',dpi=200)


####################################################################################
# plot quantiles
####################################################################################


def cal_quantiles(post,values,Nsample=None):
    quants=[]
    keys=['alpha1','mmin1','mmax1','delta1']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
    ms=np.linspace(2,100,10000)
    if Nsample==None:
        Nsample=len(post['mmin1'])
    for i in tqdm(np.arange(Nsample)):
        para=[post[key][i] for key in keys]
        pdf=PS_mass(ms,*para)
        cdf=np.cumsum(pdf)/np.sum(pdf)
        f=interp1d(cdf,ms)
        quants.append(f(values))
    quants=np.array(quants).T
    return quants

def cal_BBH_quantiles(post,values,Nsample=None):
    quants=[]
    keys=['alpha1','mmin1','mmax1','delta1']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
    keys.append('beta')
    m1_sam=np.linspace(2,100,1000)
    m2_sam=np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    if Nsample==None:
        Nsample=len(post['mmin1'])
    for i in tqdm(np.arange(Nsample)):
        para=[post[key][i] for key in keys]
        pdf=np.sum(Single_mass_pair_un(x,y,*para),axis=0)
        cdf=np.cumsum(pdf)/np.sum(pdf)
        f=interp1d(cdf,m1_sam)
        quants.append(f(values))
    quants=np.array(quants).T
    return quants

def cal_BBH_rate(post,values,Nsample=None):
    quants=[]
    keys=['alpha1','mmin1','mmax1','delta1']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
    keys.append('beta')
    m1_sam=np.linspace(2,100,1000)
    m2_sam=np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    if Nsample==None:
        Nsample=len(post['mmin1'])
    for i in tqdm(np.arange(Nsample)):
        para=[post[key][i] for key in keys]
        pdf=np.sum(Single_mass_pair_un(x,y,*para),axis=0)
        cdf=np.cumsum(pdf)/np.sum(pdf)
        f=interp1d(m1_sam,cdf)
        quants.append((1-f(values))*100.)
    quants=np.array(quants).T
    return quants


####################################################################################
# plot knots
####################################################################################

from scipy.interpolate import CubicSpline
def fm(x,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15):
    xi=np.exp(np.linspace(np.log(5),np.log(100),15))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15])
    cs = CubicSpline(xi,yi,bc_type='natural')
    return cs(x)
   
def plot_knots(filename,post,mmax=30,size=500):
    xx=np.linspace(5,100,1000)
    mmax=np.array(post['mmax1'])
    idx=np.where(mmax>81)
    parameters={}
    keys=[]
    for i in np.arange(15):
        keys.append('n'+str(i+1))
    for key in keys:
        parameters.update({key:np.array(post[key])[idx]})
        
    fig,ax = plt.subplots(figsize=(10,3))
    ax.vlines(np.exp(np.linspace(np.log(5),np.log(100),15)),np.percentile(np.array([parameters[key] for key in keys]),5,axis=-1),np.percentile(np.array([parameters[key] for key in keys]),95,axis=-1),color='black')
    
    Nsample=min(len(parameters['n1']),size)
    for i in tqdm(np.arange(Nsample)):
        para=[parameters[key][i] for key in keys]
        yy=fm(xx,*para)
        ax.plot(xx,yy,alpha=0.01,color='grey')
        
    ax.set_ylabel(r'$f(m)$',fontsize=18)
    ax.set_xlabel(r'$m$',fontsize=18)
    ax.xaxis.grid(True,which='major',ls=':')
    ax.yaxis.grid(True,which='major',ls=':')
    ax.set_axisbelow(True)
    ax.set_xscale('log')
    plt.savefig(filename,bbox_inches='tight',dpi=200)

    
from model_libs_r2 import m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,detection_selector
def plot_inj_spin(filename):
    xx=np.linspace(0,1,10)
    yy=np.ones(10)
    fig=plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    #ax.xaxis.grid(True,which='major',ls=':',color='grey')
    #ax.yaxis.grid(True,which='major',ls=':',color='grey')
    idx_low=np.where(((m1_inj<45) & (m1_inj>5) & detection_selector))
    ax1.hist(a1_inj[idx_low],bins=20,density=1,color='orange',linewidth=1.0,alpha=0.2,label=r'Observable BHs for $5<m_1/M_{\odot}<45$')
    ax1.plot(xx,yy,color='black',label='Injection')
    ax1.legend(loc='upper right')
    ax1.set_xlabel(r'$a_1$',labelpad=5,fontsize=13)
    ax1.set_ylabel('PDF',fontsize=13)
    ax1.set_ylim(0,2)
        
    ax2.hist(a2_inj[idx_low],bins=20,density=1,color='orange',linewidth=1.0,alpha=0.2,label=r'Observable BHs for $5<m_{2}/M_{\odot}<45$')
    ax2.plot(xx,yy,color='black',label='Injection')
    ax2.legend(loc='upper right')
    ax2.set_xlabel(r'$a_2$',labelpad=5,fontsize=13)
    ax2.set_ylabel('PDF',fontsize=13)
    ax2.set_ylim(0,2)

    
    idx_high=np.where(((m2_inj>25)&(m1_inj<80) & detection_selector))
    ax3.hist(a1_inj[idx_high],bins=20,density=1,color='orange',linewidth=1.0,alpha=0.2,label=r'Observable BHs with $25<m_{1}/M_{\odot}<80$')
    ax3.plot(xx,yy,color='black',label='Injection')
    ax3.legend(loc='upper right')
    ax3.set_xlabel(r'$a_1$',labelpad=5,fontsize=13)
    ax3.set_ylabel('PDF',fontsize=13)
    ax3.set_ylim(0,2)
        
    ax4.hist(a2_inj[idx_high],bins=20,density=1,color='orange',linewidth=1.0,alpha=0.2,label=r'Observable BHs with $25<m_{2}/M_{\odot}<80$')
    ax4.plot(xx,yy,color='black',label='Injection')
    ax4.legend(loc='upper right')
    ax4.set_xlabel(r'$a_2$',labelpad=5,fontsize=13)
    ax4.set_ylabel('PDF',fontsize=13)
    ax4.set_ylim(0,2)
        
    plt.savefig(filename,bbox_inches='tight',dpi=200)


def plot_inj_ct(filename):
    xx=np.linspace(-1,1,10)
    yy=np.ones(10)*0.5
    fig=plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot()
        
    idx1_low=np.where(((a1_inj<0.3) & detection_selector))
    idx2_low=np.where(((a2_inj<0.3) & detection_selector))
    idx1_high=np.where(((a1_inj>0.4) & (a1_inj<0.9) & detection_selector))
    idx2_high=np.where(((a2_inj>0.4) & (a2_inj<0.9)& detection_selector))
    ct_all=np.hstack((ct1_inj[detection_selector],ct2_inj[detection_selector]))
    ct_low=np.hstack((ct1_inj[idx1_low],ct2_inj[idx2_low]))
    ct_high=np.hstack((ct1_inj[idx1_high],ct2_inj[idx2_high]))
    ax1.hist(ct_all,bins=40,density=1,color='olive',histtype='step',linewidth=1.0,alpha=0.8,label=r'all Observable BHs')
    ax1.hist(ct_low,bins=40,density=1,color='orange',histtype='step',linewidth=1.0,alpha=0.8,label=r'Observable BHs with $a<0.3$')
    ax1.hist(ct_high,bins=40,density=1,color='blue',histtype='step',linewidth=1.0,alpha=0.8,label=r'Observable BHs with $0.4<a<0.9$')
    ax1.plot(xx,yy,color='black',label='Injection')
    ax1.legend(loc='upper right')
    ax1.set_xlabel(r'$\cos\theta_{1,2}$',labelpad=5,fontsize=13)
    ax1.set_ylabel('PDF',fontsize=13)
    ax1.set_ylim(0,1)
        
    plt.savefig(filename,bbox_inches='tight',dpi=200)

####################################################################################
# plot branch ratios
####################################################################################

def cal_branch_ratios(post,filename,size=None):
    
    parameters={}
    if size is None:
        size=len(post['log_likelihood'])
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2','beta']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m1_sam=np.linspace(2,100,500)
    m2_sam=np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    pll=[]
    plh=[]
    phl=[]
    phh=[]
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        p11,p12,p21,p22 = Double_mass_pair_branch(x,y,**para)
        pll.append(np.sum((p11)*dy*dx))
        plh.append(np.sum((p12)*dy*dx))
        phl.append(np.sum((p21)*dy*dx))
        phh.append(np.sum((p22)*dy*dx))
    pll=np.array(pll)
    plh=np.array(plh)
    phl=np.array(phl)
    phh=np.array(phh)
    ptot=pll+plh+phl+phh
    pll=pll/ptot
    plh=plh/ptot
    phl=phl/ptot
    phh=phh/ptot
    parameters.update({'pll':pll,'plh':plh,'phl':phl,'phh':phh,'lgR0':np.array(post['lgR0'])[indx]})
    with open(filename, 'wb') as fp:
        pickle.dump((parameters), fp)


####################################################################################
# plot events prob
####################################################################################


with open('./data/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples, evidences = pickle.load(fp)

keys=['m1','m2','a1','a2','cos_tilt_1','cos_tilt_2','z','prior']
data={}
for key in keys:
    data[key]=np.array([samples[i][key] for i in np.arange(len(samples))])

def cal_evets_probs(post,data=data,size=1000):
    size
    if 'log_prior' in post.keys():
        post.pop('log_prior')
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    parameters={key:np.array(post[key])[indx] for key in post.keys()}
    parameters.pop('log_likelihood')
    parameters.pop('lgR0')
    Z=[]
    for i in tqdm(np.arange(size)):
        # Select a random population sample
        para={key:parameters[key][i] for key in parameters.keys()}
        B=np.average(Event_probs(data,**para),axis=2)
        Bnorm=np.sum(B,axis=0)
        Z.append(np.array([B[0]/Bnorm,B[1]/Bnorm,B[2]/Bnorm,B[3]/Bnorm]))
        
    Z=np.array(Z).T
    Z05=np.percentile(Z,5,axis=2)
    Z50=np.percentile(Z,50,axis=2)
    Z95=np.percentile(Z,95,axis=2)
    Zmean=np.average(Z,axis=2)
    zeroindex=np.where(Z95>1e-2)

    events=['GW150914_095045', 'GW151012_095443', 'GW151226_033853',
            'GW170104_101158', 'GW170608_020116', 'GW170729_185629',
            'GW170809_082821', 'GW170814_103043', 'GW170818_022509',
            'GW170823_131358', 'GW190408_181802', 'GW190412_053044',
            'GW190413_134308', 'GW190421_213856', 'GW190503_185404',
            'GW190512_180714', 'GW190513_205428', 'GW190517_055101',
            'GW190519_153544', 'GW190521_030229', 'GW190521_074359',
            'GW190527_092055', 'GW190602_175927', 'GW190620_030421',
            'GW190630_185205', 'GW190701_203306', 'GW190706_222641',
            'GW190707_093326', 'GW190708_232457', 'GW190720_000836',
            'GW190727_060333', 'GW190728_064510', 'GW190803_022701',
            'GW190828_063405', 'GW190828_065509', 'GW190910_112807',
            'GW190915_235702', 'GW190924_021846', 'GW190925_232845',
            'GW190929_012149', 'GW190930_133541', 'GW190413_052954',
            'GW190719_215514', 'GW190725_174728', 'GW190731_140936',
            'GW190805_211137', 'GW191105_143521', 'GW191109_010717',
            'GW191127_050227', 'GW191129_134029', 'GW191204_171526',
            'GW191215_223052', 'GW191216_213338', 'GW191222_033537',
            'GW191230_180458', 'GW200112_155838', 'GW200128_022011',
            'GW200129_065458', 'GW200202_154313', 'GW200208_130117',
            'GW200209_085452', 'GW200219_094415', 'GW200224_222234',
            'GW200225_060421', 'GW200302_015811', 'GW200311_115853',
            'GW200316_215756', 'GW191103_012549', 'GW200216_220804']

    #print('1+1','1+2','2+1','2+2')
    ####
    #latex

    for n in np.arange(len(events)):
        i=events[n]
        if (round(Zmean[n][0],3))<0.5:
            print('{}\\{}&{}&{}&{}&{} \\\\'.format(i[:8],i[8:],round(Zmean[n][0],3),round(Zmean[n][1],3),round(Zmean[n][2],3),round(Zmean[n][3],3)))
    print('all events:')
    for n in np.arange(len(events)):
        i=events[n]
        print('{}\\{}&{}&{}&{}&{} \\\\'.format(i[:8],i[8:],round(Zmean[n][0],3),round(Zmean[n][1],3),round(Zmean[n][2],3),round(Zmean[n][3],3)))



####################################################################################
# plot prior
####################################################################################

priors=Double_priors()
keys=['delta1', 'mmin1', 'mmax1', 'alpha1', 'mu_a1', 'sigma_a1', 'amax1', 'delta2', 'mmin2', 'mmax2', 'alpha2', 'amin2', 'mu_a2', 'sigma_a2', 'amax2', 'r2', 'beta', 'sigma_t', 'zeta', 'lgR0', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11']
prior_sample={}
size=1000000
for key in keys:
    prior_sample[key]=priors[key].sample(size)
    
constraint=((prior_sample['mmax2']-prior_sample['mmin2']>20) & (prior_sample['mmax1']-prior_sample['mmin1']>20) & (prior_sample['amax2']-prior_sample['amin2']>0.2) &\
    (prior_sample['amax2']-prior_sample['mu_a2']>0.) & (prior_sample['amax1']-prior_sample['mu_a1']>0.) & (prior_sample['mu_a2']-prior_sample['mu_a1']>0.) & (prior_sample['mu_a2']-prior_sample['amin2']>0.))
idx=np.where(constraint)
for key in keys:
    prior_sample[key]=prior_sample[key][idx]
sel_size=len(prior_sample['mmax1'])
    
prior_sample.update({'n1':np.ones(sel_size)*0.,'n12':np.ones(sel_size)*0.,'o1':np.ones(sel_size)*0.,'o12':np.ones(sel_size)*0.,'gamma':np.ones(sel_size)*2.7, 'mu_t':np.ones(sel_size)*1.,'zmin':np.ones(sel_size)*-1.,'amin1':np.ones(sel_size)*0})

def plot_prior(colors):
    fig=plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(5, 2)
    ax1 = fig.add_subplot(gs[:3,:])
    ax3 = fig.add_subplot(gs[3:,0])
    ax2 = fig.add_subplot(gs[3:,1])

    parameters={}
    indx=np.random.choice(np.arange(len(prior_sample['mmax1'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2']
    keys1=['alpha1','mmin1','mmax1','delta1']
    keys2=['alpha2','mmin2','mmax2','delta2']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))
    for i in np.arange(12):
        keys.append('o'+str(i+1))
        keys2.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(prior_sample[key])[indx]})

    m_1G=[]
    m_2G=[]
    x=np.linspace(2,100,5000)
    m1_sam=x
    for i in tqdm(np.arange(5000)):
        para={key:parameters[key][i] for key in parameters.keys()}
        para1=[parameters[key][i] for key in keys1]
        para2=[parameters[key][i] for key in keys2]
        m_1G.append(PS_mass(x,*para1)*(1-para['r2']))
        m_2G.append(PS_mass(x,*para2)*para['r2'])
    m_1G=np.array(m_1G)
    m_1G_pup=np.percentile(m_1G,99.9,axis=0)
    m_1G_plow=np.percentile(m_1G,0,axis=0)
    m_1G_pmid=np.percentile(m_1G,50,axis=0)
    m_1G_pmean=np.mean(m_1G,axis=0)
    m_2G=np.array(m_2G)
    m_2G_pup=np.percentile(m_2G,99.9,axis=0)
    m_2G_plow=np.percentile(m_2G,0,axis=0)
    m_2G_pmid=np.percentile(m_2G,50,axis=0)
    m_2G_pmean=np.mean(m_2G,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    #plt.fill_between(m1_sam,plow,pup,color=colors[0],alpha=0.4,label='total')
    #plt.plot(m1_sam,pmean,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_1G_plow,m_1G_pup,color=colors[0],alpha=0.1,label='LS prior')
    #ax1.plot(m1_sam,m_1G.T,color=colors[0],alpha=0.01,lw=0.01,label='low spin prior')
    #ax1.plot(m1_sam,m_1G_pup,color=colors[0],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_1G_pmean,color=colors[0],alpha=0.9,ls=':')
    #ax1.plot(m1_sam,m_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_2G_plow,m_2G_pup,color=colors[1],alpha=0.1,label='HS prior')
    #ax1.plot(m1_sam,m_2G.T,color=colors[1],alpha=0.01,lw=0.01,label='high spin prior')
    #ax1.plot(m1_sam,m_2G_pup,color=colors[1],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_2G_pmean,color=colors[1],alpha=0.9,ls=':')
    #ax1.plot(m1_sam,m_2G_pmid,color=colors[1],alpha=0.9)

    ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_ylim(5e-5,5e-1)
    ax1.set_xlim(2,100)
    ax1.set_xlabel(r'$m_{1,2}/M_{\odot}$')
    ax1.set_ylabel('PDF')
    ax1.legend(loc=0)

    parameters={}
    keys=['mu_a1','sigma_a1','mu_a2','sigma_a2','amin2','amin1','amax1','amax2']
    keys1a=['mu_a1','sigma_a1','amin1','amax1']
    keys2a=['mu_a2','sigma_a2','amin2','amax2']
    for key in keys:
        parameters.update({key:np.array(prior_sample[key])[indx]})
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in tqdm(np.arange(50000)):
        para1a=[parameters[key][i] for key in keys1a]
        para2a=[parameters[key][i] for key in keys2a]
        a_1G.append(spin_a(a_sam,*para1a))
        a_2G.append(spin_a(a_sam,*para2a))
    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,99.9,axis=0)
    a_1G_plow=np.percentile(a_1G,0,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,99.9,axis=0)
    a_2G_plow=np.percentile(a_2G,0,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)

    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.1,label='LS prior')
    #ax3.plot(a_sam,a_1G_pmean,color=colors[0],alpha=0.8)
    #ax3.plot(a_sam,a_1G_pmid,color=colors[0],alpha=0.8)
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.1,label='HS prior')
    #ax3.plot(a_sam,a_2G_pmean,color=colors[1],alpha=0.8)
    #ax3.plot(a_sam,a_2G_pmid,color=colors[1],alpha=0.8)
    ax3.set_xlabel(r'$a_{1,2}$')
    ax3.set_ylabel('PDF')
    ax3.set_xlim(0,1)
    #ax3.set_ylim(0,5)
    ax3.legend(loc=0)
    
    parameters={}
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(prior_sample[key])[indx]})
    pct=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(5000)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,99.9,axis=0)
    pct_plow=np.percentile(pct,0,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.1,label='prior')
    #ax2.plot(ct_sam,pct_pmean,color=colors[2],alpha=0.8)
    #ax2.plot(ct_sam,pct_pmid,color=colors[2],alpha=0.8)
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel('PDF')
    ax2.set_xlim(-1,1)
    #ax2.set_ylim(0,1.8)
    ax2.legend(loc=0)
    
    plt.tight_layout()
    return fig, ax1, ax2, ax3

def plot_ma_with_prior(fig, ax1, ax2, ax3, post,colors,size,filename):
    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2']
    keys1=['alpha1','mmin1','mmax1','delta1']
    keys2=['alpha2','mmin2','mmax2','delta2']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))
    for i in np.arange(12):
        keys.append('o'+str(i+1))
        keys2.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m_1G=[]
    m_2G=[]
    x=np.linspace(2,100,5000)
    m1_sam=x
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        para1=[parameters[key][i] for key in keys1]
        para2=[parameters[key][i] for key in keys2]
        m_1G.append(PS_mass(x,*para1)*(1-para['r2']))
        m_2G.append(PS_mass(x,*para2)*para['r2'])
    m_1G=np.array(m_1G)
    m_1G_pup=np.percentile(m_1G,95,axis=0)
    m_1G_plow=np.percentile(m_1G,5,axis=0)
    m_1G_pmid=np.percentile(m_1G,50,axis=0)
    m_1G_pmean=np.mean(m_1G,axis=0)
    m_2G=np.array(m_2G)
    m_2G_pup=np.percentile(m_2G,95,axis=0)
    m_2G_plow=np.percentile(m_2G,5,axis=0)
    m_2G_pmid=np.percentile(m_2G,50,axis=0)
    m_2G_pmean=np.mean(m_2G,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    #plt.fill_between(m1_sam,plow,pup,color=colors[0],alpha=0.4,label='total')
    #plt.plot(m1_sam,pmean,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_1G_plow,m_1G_pup,color=colors[0],alpha=0.2,label='LS posterior')
    ax1.plot(m1_sam,m_1G_plow,color=colors[0],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_1G_pup,color=colors[0],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_1G_pmean,color=colors[0],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_2G_plow,m_2G_pup,color=colors[1],alpha=0.2,label='HS posterior')
    ax1.plot(m1_sam,m_2G_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_2G_pup,color=colors[1],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_2G_pmean,color=colors[1],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_2G_pmid,color=colors[1],alpha=0.9)

    ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_ylim(5e-5,5e-1)
    ax1.set_xlim(2,100)
    ax1.set_xlabel(r'$m/M_{\odot}$')
    ax1.set_ylabel('PDF')
    ax1.legend(loc=0)

    parameters={}
    keys=['mu_a1','sigma_a1','mu_a2','sigma_a2','amin2','amin1','amax1','amax2']
    keys1a=['mu_a1','sigma_a1','amin1','amax1']
    keys2a=['mu_a2','sigma_a2','amin2','amax2']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in tqdm(np.arange(size)):
        para1a=[parameters[key][i] for key in keys1a]
        para2a=[parameters[key][i] for key in keys2a]
        a_1G.append(spin_a(a_sam,*para1a))
        a_2G.append(spin_a(a_sam,*para2a))
    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,95,axis=0)
    a_1G_plow=np.percentile(a_1G,5,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,95,axis=0)
    a_2G_plow=np.percentile(a_2G,5,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)

    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.2,label='LS posterior')
    #ax3.plot(a_sam,a_1G_pmean,color=colors[0],alpha=0.8)
    ax3.plot(a_sam,a_1G_pmid,color=colors[0],alpha=0.8)
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.2,label='HS posterior')
    #ax3.plot(a_sam,a_2G_pmean,color=colors[1],alpha=0.8)
    ax3.plot(a_sam,a_2G_pmid,color=colors[1],alpha=0.8)
    ax3.set_xlabel(r'$a$')
    ax3.set_ylabel('PDF')
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,5)
    ax3.legend(loc=0)
    
    parameters={}
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    pct=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.2,label='posterior')
    #ax2.plot(ct_sam,pct_pmean,color=colors[2],alpha=0.8)
    ax2.plot(ct_sam,pct_pmid,color=colors[2],alpha=0.8)
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel('PDF')
    ax2.set_xlim(-1,1)
    ax2.set_ylim(0,1.8)
    ax2.legend(loc=0)
    
    plt.tight_layout()
    plt.savefig(filename)


########


def plot_q_dist(post,colors,size,filename):

    fig=plt.figure(figsize=(6,4))
    ax1 = fig.add_subplot()
    
    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2','beta','lgR0']
    keys1=['alpha1','mmin1','mmax1','delta1','alpha2','r2','mmin2','mmax2','delta2','beta']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))
    for i in np.arange(12):
        keys.append('o'+str(i+1))
        keys1.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m1_sam=np.linspace(2,100,500)
    q_sam=np.linspace(0.01,1.01,499)
    
    j,k=np.meshgrid(m1_sam,q_sam)
    dj = m1_sam[1]-m1_sam[0]
    q_1G=[]
    q_2G=[]
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        R0=10**parameters['lgR0'][i]
        para1={key:parameters[key][i] for key in keys1}
        p11,p12,p21,p22 = Double_mass_pair_branch(j,k*j,**para1)
        q_1G.append(np.sum(p11*j*dj,axis=1)*R0)
        q_2G.append(np.sum((p12+p21+p22)*j*dj,axis=1)*R0)
    q_1G=np.array(q_1G)
    q_1G_pup=np.percentile(q_1G,95,axis=0)
    q_1G_plow=np.percentile(q_1G,5,axis=0)
    q_1G_pmid=np.percentile(q_1G,50,axis=0)
    q_1G_pmean=np.mean(q_1G,axis=0)
    q_2G=np.array(q_2G)
    q_2G_pup=np.percentile(q_2G,95,axis=0)
    q_2G_plow=np.percentile(q_2G,5,axis=0)
    q_2G_pmid=np.percentile(q_2G,50,axis=0)
    q_2G_pmean=np.mean(q_2G,axis=0)
    q=q_1G+q_2G
    q_pup=np.percentile(q,95,axis=0)
    q_plow=np.percentile(q,5,axis=0)
    q_pmid=np.percentile(q,50,axis=0)
    q_pmean=np.mean(q,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    ax1.fill_between(q_sam,q_1G_plow,q_1G_pup,color=colors[0],alpha=0.15,label='First-generation mergers')
    #ax1.plot(q_sam,q_1G_plow,color=colors[0],alpha=0.8,lw=0.3)
    #ax1.plot(q_sam,q_1G_pup,color=colors[0],alpha=0.8,lw=0.3)
    ax1.plot(q_sam,q_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(q_sam,q_2G_plow,q_2G_pup,color=colors[1],alpha=0.15,label='Hierarchical mergers')
    #ax1.plot(q_sam,q_2G_plow,color=colors[1],alpha=0.8,lw=0.3)
    #ax1.plot(q_sam,q_2G_pup,color=colors[1],alpha=0.8,lw=0.3)
    ax1.plot(q_sam,q_2G_pmid,color=colors[1],alpha=0.9)
        
    #ax1.fill_between(q_sam,q_plow,q_pup,color=colors[2],alpha=0.15,label='HM')
    ax1.plot(q_sam,q_plow,color=colors[2],alpha=0.8,lw=1.5,label='Over all')
    ax1.plot(q_sam,q_pup,color=colors[2],alpha=0.8,lw=1.5)
    ax1.plot(q_sam,q_pmid,color=colors[2],alpha=0.9,ls=':')

    ax1.set_yscale('log')
    ax1.set_ylim(1e-2,1e3)
    ax1.set_xlim(0,1)
    ax1.set_xlabel(r'$q$')
    ax1.set_ylabel(r'$\frac{{\rm d}\mathcal{R}(z=0)}{{\rm d}q}~[{\rm Gpc}^{-3}~{\rm yr}^{-1}]$')
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)


def plot_step_ma_dist(post,colors,size,filename):

    fig=plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(5, 2)
    ax1 = fig.add_subplot(gs[:3,:])
    ax3 = fig.add_subplot(gs[3:,0])
    ax2 = fig.add_subplot(gs[3:,1])

    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','md']
    keys1=['alpha1','mmin1','mmax1','delta1']
    for i in np.arange(12):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m_1G=[]
    m_2G=[]
    x=np.linspace(2,100,5000)
    m1_sam=x
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        para1=[parameters[key][i] for key in keys1]
        m_1G.append(PS_mass(x,*para1)*(x<para['md']))
        m_2G.append(PS_mass(x,*para1)*(x>para['md']))
    m_1G=np.array(m_1G)
    m_1G_pup=np.percentile(m_1G,95,axis=0)
    m_1G_plow=np.percentile(m_1G,5,axis=0)
    m_1G_pmid=np.percentile(m_1G,50,axis=0)
    m_1G_pmean=np.mean(m_1G,axis=0)
    m_2G=np.array(m_2G)
    m_2G_pup=np.percentile(m_2G,95,axis=0)
    m_2G_plow=np.percentile(m_2G,5,axis=0)
    m_2G_pmid=np.percentile(m_2G,50,axis=0)
    m_2G_pmean=np.mean(m_2G,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    #plt.fill_between(m1_sam,plow,pup,color=colors[0],alpha=0.4,label='total')
    #plt.plot(m1_sam,pmean,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_1G_plow,m_1G_pup,color=colors[0],alpha=0.2,label=r'$m<m_{\rm d}$')
    ax1.plot(m1_sam,m_1G_plow,color=colors[0],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_1G_pup,color=colors[0],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_1G_pmean,color=colors[0],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_2G_plow,m_2G_pup,color=colors[1],alpha=0.2,label=r'$m>m_{\rm d}$')
    ax1.plot(m1_sam,m_2G_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_2G_pup,color=colors[1],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_2G_pmean,color=colors[1],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_2G_pmid,color=colors[1],alpha=0.9)

    ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_ylim(5e-5,5e-1)
    ax1.set_xlim(2,100)
    ax1.set_xlabel(r'$m/M_{\odot}$')
    ax1.set_ylabel(r'$p(m)$')
    ax1.legend(loc=0)

    parameters={}
    keys=['mu_a1','sigma_a1','mu_a2','amin1','amax1']
    keys1a=['mu_a1','sigma_a1','amin1','amax1']
    keys2a=['mu_a2','sigma_a1','amin1','amax1']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in tqdm(np.arange(size)):
        para1a=[parameters[key][i] for key in keys1a]
        para2a=[parameters[key][i] for key in keys2a]
        a_1G.append(spin_a(a_sam,*para1a))
        a_2G.append(spin_a(a_sam,*para2a))
    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,95,axis=0)
    a_1G_plow=np.percentile(a_1G,5,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,95,axis=0)
    a_2G_plow=np.percentile(a_2G,5,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)

    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.2,label=r'$m<m_{\rm d}$')
    #ax3.plot(a_sam,a_1G_pmean,color=colors[0],alpha=0.8)
    ax3.plot(a_sam,a_1G_pmid,color=colors[0],alpha=0.8)
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.2,label=r'$m>m_{\rm d}$')
    #ax3.plot(a_sam,a_2G_pmean,color=colors[1],alpha=0.8)
    ax3.plot(a_sam,a_2G_pmid,color=colors[1],alpha=0.8)
    ax3.set_xlabel(r'$a$')
    ax3.set_ylabel(r'$p(a)$')
    ax3.set_xlim(0,1)
    #ax3.set_ylim(0,5)
    ax3.legend(loc=0)
    
    parameters={}
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    pct=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.2)
    #ax2.plot(ct_sam,pct_pmean,color=colors[2],alpha=0.8)
    ax2.plot(ct_sam,pct_pmid,color=colors[2],alpha=0.8)
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel(r'$p(\cos\theta)$')
    ax2.set_xlim(-1,1)
    #ax2.set_ylim(0,1.8)
    ax2.legend(loc=0)

    plt.tight_layout()
    plt.savefig(filename)


####################################################################################
# cal_pert
####################################################################################

def f_m(m1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12):
    xi=np.exp(np.linspace(np.log(6),np.log(80),12))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12])
    cs = CubicSpline(xi,yi,bc_type='natural')
    pm1 = cs(m1)*(m1>6)*(m1<80)
    return pm1


def cal_f_m(post,values,labels):
    keys=[]
    for i in np.arange(12):
        keys.append('n'+str(i+1))
    Nsample=len(post['mmin1'])
    quants=[]
    for i in tqdm(np.arange(Nsample)):
        para=[post[key][i] for key in keys]
        quants.append(f_m(values,*para))
    quants=np.array(quants).T
    post.update({labels[i]:quants[i] for i in np.arange(len(values))})
    return post


def plot_tilt_dist(post,post2,colors,size,filename):

    fig=plt.figure(figsize=(5,3))
    gs = gridspec.GridSpec(5, 2)
    ax2 = fig.add_subplot()
    
    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    pct=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.2,label='Two-component')
    #ax2.plot(ct_sam,pct_pmean,color=colors[2],alpha=0.8)
    ax2.plot(ct_sam,pct_pmid,color=colors[2],alpha=0.8)
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel(r'$p(\cos\theta)$')
    ax2.set_xlim(-1,1)
    
    parameters={}
    indx=np.random.choice(np.arange(len(post2['log_likelihood'])),size=size)
    keys=['sigma_t','zeta']
    for key in keys:
        parameters.update({key:np.array(post2[key])[indx]})
    pct=[]
    ct_sam=np.linspace(-1,1,500)
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[3],alpha=0.2,label='PP&Default')
    ax2.plot(ct_sam,pct_pmid,color=colors[3],alpha=0.8)


    plt.tight_layout()
    plt.savefig(filename)

def plot_reweighed_ct(post,filename,colors,size):
    xx=np.linspace(-1,1,20)
    ct_all=np.hstack((ct1_inj[detection_selector],ct2_inj[detection_selector]))
    ns=np.histogram(ct_all,bins=20)
    fx=interp1d(xx,ns[0]/np.sum(ns[0]))
    
    ct_sam=np.linspace(-1,1,500)
    fs=fx(ct_sam)

    fig=plt.figure(figsize=(5,3))
    gs = gridspec.GridSpec(5, 2)
    ax2 = fig.add_subplot()
    
    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['sigma_t','zeta','mu_t','zmin']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    pct=[]
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct_un = DF_ct(ct_sam,**para_ct)/fs
        norm=np.sum(pct_un)*2./500.
        pct.append(pct_un/norm)
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')

    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[3],alpha=0.2,label='Two-component (re-weighted)')
    ax2.plot(ct_sam,pct_pmid,color=colors[3],alpha=0.8)
    
    pct=[]
    for i in tqdm(np.arange(size)):
        para_ct={key:parameters[key][i] for key in keys}
        pct.append(DF_ct(ct_sam,**para_ct))
    pct=np.array(pct)
    pct_pup=np.percentile(pct,95,axis=0)
    pct_plow=np.percentile(pct,5,axis=0)
    pct_pmid=np.percentile(pct,50,axis=0)
    pct_pmean=np.mean(pct,axis=0)
    
    ax2.fill_between(ct_sam,pct_plow,pct_pup,color=colors[2],alpha=0.2,label='Two-component')
    ax2.plot(ct_sam,pct_pmid,color=colors[2],alpha=0.8)
 
    print('plotting LIGO results')
    ax2=plot_LIGO_ct(ax2)
    ax2.legend(loc=0)
    
    ax2.set_xlabel(r'$\cos\theta_{1,2}$')
    ax2.set_ylabel(r'$p(\cos\theta)$')
    ax2.set_xlim(-1,1)

        
    plt.savefig(filename,bbox_inches='tight',dpi=200)
