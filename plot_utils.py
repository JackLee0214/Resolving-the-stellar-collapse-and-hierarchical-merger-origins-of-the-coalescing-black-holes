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


####################################################################################
# calculate population-informed samples
####################################################################################
from model_libs import hyper_Double

def pop_informed_samples(posteriors,post,size,filename):
    n_catalogs=size
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
    for i in range(n_catalogs):
        print('No.{} of {} sets'.format(i,n_catalogs))
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
    for i in range(resampled_a1.shape[0]):
        print(i)

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
        
    for i in range(resampled_a2.shape[0]):
        print(i)

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
from model_libs import PS_mass, spin_a

def plot_ma_dist(post,colors,size,filename,afinal=None):

    fig=plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot()
    ax3 = fig.add_axes([0.55,0.55,0.35,0.35])

    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=['mmin1','mmax1','alpha1','delta1','alpha2','r2','mmin2','mmax2','delta2']
    keys1=['alpha1','mmin1','mmax1','delta1']
    keys2=['alpha2','mmin2','mmax2','delta2']
    for i in np.arange(15):
        keys.append('n'+str(i+1))
        keys1.append('n'+str(i+1))
    for i in np.arange(15):
        keys.append('o'+str(i+1))
        keys2.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    m_1G=[]
    m_2G=[]
    x=np.linspace(2,100,5000)
    m1_sam=x
    for i in np.arange(size):
        if i%50==0:
            print(i)
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
    ax1.set_ylim(1e-4,1)
    ax1.set_xlim(0,100)
    ax1.set_xlabel(r'$m_{1,2}/M_{\odot}$')
    ax1.set_ylabel('PDF')
    ax1.legend(loc=2)

    parameters={}
    keys=['mu_a1','sigma_a1','sigma_t1','mu_a2','sigma_a2','sigma_t2','amin2','zeta1','amin1','amax1','zmin1','zmin2','amax2','zeta2']
    keys1a=['mu_a1','sigma_a1','amin1','amax1']
    keys2a=['mu_a2','sigma_a2','amin2','amax2']
    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    for i in np.arange(size):
        if i%50==0:
            print(i)
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

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.2,label='low spin')
    #ax3.plot(a_sam,a_1G_pmean,color=colors[0],alpha=0.8)
    ax3.plot(a_sam,a_1G_pmid,color=colors[0],alpha=0.8)
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.2,label='high spin')
    #ax3.plot(a_sam,a_2G_pmean,color=colors[1],alpha=0.8)
    ax3.plot(a_sam,a_2G_pmid,color=colors[1],alpha=0.8)
    if afinal is not None:
        af_sam=np.linspace(0,1,len(afinal[0]))
        for i in np.arange(len(afinal)):
            ax3.plot(af_sam,afinal[i],color='grey',alpha=0.1)
    ax3.set_xlabel(r'$a_{1,2}$')
    ax3.set_ylabel('PDF')
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,5.5)
    plt.tight_layout()
    plt.savefig(filename)

####################################################################
#plot mmax and kick velocity
####################################################################
from model_libs import Single_mass_pair_un, spin_ct
import precession

#generate kick velocity from a specified BBH distribution

def generate_kick_velocity(post,N_sample=10000):
    parameters=['alpha1','mmin1','mmax1','delta1','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14','n15','amin1','mu_a1','sigma_a1','amax1','sigma_t1','zmin1','zeta1','beta']
    paras = {key:np.mean(post[key]) for key in parameters}
    size=1000000
    m1_sam=Uniform(2,100).sample(size=size)
    m2_sam=Uniform(2,100).sample(size=size)
    ct_sam=Uniform(-1,1).sample(size=size)
    chosen_size=N_sample

    probs=Single_mass_pair_un(m1_sam,m2_sam, **paras)
    probs /= np.sum(probs)
    chosenInd = np.random.choice(np.arange(len(probs)),p=probs,size=chosen_size)
    m1=m1_sam[chosenInd]
    m2=m2_sam[chosenInd]
    a1=TG(paras['mu_a1'],paras['sigma_a1'],paras['amin1'],paras['amax1']).sample(size=chosen_size)
    a2=TG(paras['mu_a1'],paras['sigma_a1'],paras['amin1'],paras['amax1']).sample(size=chosen_size)

    ct_probs=spin_ct(ct_sam,paras['sigma_t1'],paras['zeta1'],paras['zmin1'])
    ct_probs /= np.sum(ct_probs)
    ct_chosenInd = np.random.choice(np.arange(len(ct_probs)),p=ct_probs,size=chosen_size*2)
    ct1=ct_sam[ct_chosenInd][:chosen_size]
    ct2=ct_sam[ct_chosenInd][chosen_size:]
    tilt1=np.arccos(ct1)
    tilt2=np.arccos(ct2)
    q=m2/m1
    spin_phase=Uniform(0,2*np.pi).sample(size=chosen_size)
    v_final=[]
    for j in np.arange(chosen_size):
        _, _, _, S1, S2 = precession.get_fixed(q[j], a1[j], a2[j])
        vj=precession.finalkick(tilt1[j],tilt2[j],spin_phase[j],q[j],S1,S2,maxkick=False, kms=True)
        if vj<5000:
            v_final.append(vj)
    v_final=np.array(v_final)
    return v_final

##plot mmax1 distribution and kick velocity distribution
def plot_mmax_vkick(post,filename):
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

    vk=generate_kick_velocity(post)

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot('121')
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


    ax = fig.add_subplot('122')
    bins=30
    #np.exp(np.linspace(np.log(1),np.log(1200),30))
    #ax.xaxis.grid(True,which='major',ls=':',color='grey')
    #ax.yaxis.grid(True,which='major',ls=':',color='grey')
    ax.hist(vk,bins=bins,density=1,color='green',linewidth=1.0,alpha=0.2,log=False)
    ax.hist(vk,bins=bins,density=1,color='green',histtype='step',linewidth=1.0,alpha=0.8,log=False)
    #ax.set_xscale('log')
    #ax.set_xlim(1,1200)
    ax.set_xlabel(r'$v_{\rm kick}[{\rm km~s^{-1}}]$',labelpad=5,fontsize=13)
    ax.set_ylabel('PDF',fontsize=13)
    ax.axvline([np.percentile(vk,5)],ls='dashed',color='black')
    print('vk:5\%',r"${:.2f}$".format(np.percentile(vk,5)))
    v=np.percentile(vk,50)
    v5=np.percentile(vk,5)
    v95=np.percentile(vk,95)
    print(r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(v,v-v5,v95-v))
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')

####################################################################
#plot corners
####################################################################

def plot_corner(post,params,show_keys,color,filename,smooth=1.):
    print('ploting')
    data2=np.array([np.array(post[key]) for key in params])
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
    c1=color
    ranges=[[min(data2[i]),max(data2[i])] for i in np.arange(len(data2))]
    percentiles=[[np.percentile(data2[i],5),np.percentile(data2[i],50),np.percentile(data2[i],95)] for i in np.arange(len(data2))]
    titles=[r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(percentiles[i][1],percentiles[i][1]-percentiles[i][0], percentiles[i][2]-percentiles[i][1]) for i in np.arange(len(data2)) ]
    kwargs = dict(title_kwargs=dict(fontsize=15), labels=show_keys, smooth=smooth, bins=25,  quantiles=[0.05,0.5,0.95], range=ranges,\
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

