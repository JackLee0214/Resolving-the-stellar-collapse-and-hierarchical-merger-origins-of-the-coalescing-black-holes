import bilby
import pickle
import json
import numpy as np
from plot_utils import pop_informed_samples, plot_ma_dist, plot_ma_sample
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pickle

outdir='results'
label='Double_spin'
fig_dir='figures'

##################################################
#read data
##################################################
"""
with open('./{}/Double_spin_result.json'.format(outdir)) as a:
    data=json.load(a)
post=data['posterior']['content']

with open('./{}/Double_spin_post.pickle'.format(outdir), 'wb') as fp:
    pickle.dump((post), fp)
"""

with open('./{}/Double_spin_post.pickle'.format(outdir), 'rb') as fp:
    post = pickle.load(fp)

##################################################
#plot Figure 1
##################################################
print('ploting Figure 1')

colors=["#08306b","#4d9221","#e31a1c"]
size=20000
filename='./{}/Double_spin_mass_dist.pdf'.format(fig_dir)
plot_ma_dist(post,colors,size,filename)

##################################################
#plot Figure 2
##################################################
print('ploting Figure 2')

#read pop-infromed samples
filename='./{}/m_vs_a_reweighted_events.pdf'.format(fig_dir)
with open('./{}/Double_spin_informed.pickle'.format(outdir), 'rb') as fp:
    informed_samples = pickle.load(fp)
    
#plot pop-infromed events
plot_ma_sample(samples=informed_samples,size=1000,filename=filename)

##################################################
#plot Figure 3
##################################################
print('ploting Figure 3')

from plot_utils import plot_corner
plot_corner(post,color="#4d9221",params=['zeta2','zmin2','sigma_t2'],show_keys=[r'$\zeta_2$',r'$z_{\rm min,2}$',r'$\sigma_{\rm t,2}$'],filename='./{}/zeta2.pdf'.format(fig_dir))
plot_corner(post,color="#CC6677",params=['zeta1','zmin1','sigma_t1'],show_keys=[r'$\zeta_1$',r'$z_{\rm min,1}$',r'$\sigma_{\rm t,1}$'],filename='./{}/zeta1.pdf'.format(fig_dir))

##################################################
#plot Figure 4
##################################################
print('ploting Figure 4')

from plot_utils import plot_mmax_vkick
plot_mmax_vkick(post=post,filename='./{}/mmax_vkick.pdf'.format(fig_dir))

##################################################
#plot Supplementary Figure 1
##################################################

total_vars_para=np.loadtxt('./{}/likelihood_vars.txt'.format(outdir))
print('The average variance of total log likelihood is: {}'.format(np.mean(total_vars_para.T[0])))
print('The average variance of log likelihood from events is: {}'.format(np.mean(total_vars_para.T[1])))
print('The average variance of log likelihood from selection effect is: {}'.format(np.mean(total_vars_para.T[3])))


print('ploting Supplementary Figure 1')
from plot_utils import plot_vars
vars_para=total_vars_para.T[(np.array([0,1,3,5,6,7,8,10,11,12,14]),)]
show_keys = np.array([r'$\sigma_{\rm \ln\mathcal{L}}^2$', r'$\sigma_{\rm obs}^2$',r'$\sigma_{\rm sel}^2$',\
    r'$M_{\rm max,1}$', r'$r_2$',r'$\mu_{\rm a,1}$',r'$\sigma_{\rm a,1}$',r'$a_{\rm max,1}$',r'$\mu_{\rm a,2}$',r'$\sigma_{\rm a,2}$',r'$a_{\rm min,2}$'])
plot_vars(data2=vars_para,filename='./{}/likelihood_var_corner.pdf'.format(fig_dir),show_keys=show_keys,c1='salmon')

##################################################
#plot Supplementary Figure 2
##################################################
print('ploting Supplementary Figure 2')

from plot_utils import plot_corner
params=['mmax1','alpha1','mmax2','n11','n12','n13','n14']
show_keys = np.array([r'$m_{\rm max,1}[M_{\odot}]$', r'$\alpha_1$', r'$m_{\rm max,2}[M_{\odot}]$', r'$f_1^{11}$',r'$f_1^{12}$',r'$f_1^{13}$',r'$f_1^{14}$'])
color='purple'
filename='./{}/mmax1_corner.pdf'.format(fig_dir)
plot_corner(post,params,show_keys,color,filename,smooth=0.5)

##################################################
#plot Supplementary Figure 3
##################################################
print('ploting Supplementary Figure 3')

with open('./{}/PS_linear_result.json'.format(outdir)) as a:
    data=json.load(a)
post_PS_linear=data['posterior']['content']

from plot_utils import plot_mu_sigma
params=['mu_al','mu_ar']
show_keys = np.array([r'$\mu_{\rm a,left}$', r'$\mu_{\rm a,right}$'])
color='orange'
filename='./{}/mu_a_linear.pdf'.format(fig_dir)

plot_mu_sigma(post_PS_linear,params,show_keys,color,filename,lims=[0,1])

color = "#4d9221"
params=['sigma_al','sigma_ar']
show_keys = np.array([r'$\sigma_{\rm a,left}$', r'$\sigma_{\rm a,right}$'])
filename='./{}/sigma_a_linear.pdf'.format(fig_dir)

plot_mu_sigma(post_PS_linear,params,show_keys,color,filename,lims=[0,0.5])

from plot_utils import plot_corner
##################################################
#plot Supplementary Figure 4
##################################################
print('ploting Supplementary Figure 4')

filename='./{}/mass_corner.pdf'.format(fig_dir)
params=['beta','mmin1','mmax1','alpha1','r2','mmin2','mmax2','alpha2']
show_keys=[r'$\beta$',r'$m_{\rm min,1}$', r'$m_{\rm max,1}$', r'$\alpha_1$', r'$r_2$',r'$m_{\rm min,2}$', r'$m_{\rm max,2}$', r'$\alpha_2$']
color='green'
plot_corner(post,params,show_keys,color,filename,smooth=0.5)

##################################################
#plot Supplementary Figure 5
##################################################
print('ploting Supplementary Figure 5')

filename='./{}/spin_corner.pdf'.format(fig_dir)
params=['mu_a1','sigma_a1','mu_a2','sigma_a2','sigma_t1','sigma_t2','zeta1','zeta2','zmin1','zmin2','amin2','amax1','amax2']
show_keys=[r'$\mu_{\rm a,1}$', r'$\sigma_{\rm a,1}$',r'$\mu_{\rm a,2}$', r'$\sigma_{\rm a,2}$', r'$\sigma_{\rm t,1}$', r'$\sigma_{\rm t,2}$',r'$\zeta_1$',r'$\zeta_2$',r'$z_{\rm min,1}$',r'$z_{\rm min,2}$',r'$a_{\rm min,2}$',r'$a_{\rm max,1}$',r'$a_{\rm max,2}$']
color='orange'
plot_corner(post,params,show_keys,color,filename,smooth=0.5)

##################################################
#plot Supplementary Figure 6
##################################################
print('ploting Supplementary Figure 6')

Rrs=np.loadtxt('./{}/branch_ratios.txt'.format(outdir))
print('ploting')
params=['ll','lh','hl','hh']
post={params[i]:Rrs[i] for i in np.arange(4)}
show_keys = np.array([r'$r_{\rm low,low}$', r'$r_{\rm low,high}$',r'$r_{\rm high,low}$',r'$r_{\rm high,high}$'])
filename='./{}/branch_ratios_corner.pdf'.format(fig_dir)
color='blue'
from plot_utils import plot_corner
plot_corner(post,params,show_keys,color,filename,smooth=0.5)


