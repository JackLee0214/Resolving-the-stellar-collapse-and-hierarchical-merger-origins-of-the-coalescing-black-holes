import bilby
import pickle
import json
import numpy as np
from plot_utils import pop_informed_samples, plot_ma_dist, plot_ma_sample, plot_qchieff_sample, plot_corner, cal_quantiles, plot_m1_m2_dist, cal_BBH_quantiles, plot_knots, cal_BBH_rate,plot_q_dist, calculated_fneg,cal_branch_ratios,cal_evets_probs,plot_prior,plot_ma_with_prior,plot_step_ma_dist,cal_f_m
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pickle

outdir='results'
label='Double_spin'
add_label=''
fig_dir='figures'

##################################################
#read data
##################################################
'''
with open('./{}/Double_spin_result.json'.format(outdir)) as a:
    data=json.load(a)
post=data['posterior']['content']

with open('./Double_spin_post.pickle', 'wb') as fp:
    pickle.dump((post), fp)
'''

with open('./{}/Double_spin_post.pickle'.format(outdir), 'rb') as fp:
    post = pickle.load(fp)

##################################################
#plot Figure 1
##################################################
print('ploting Figure 1')

colors=["#08306b","#4d9221","#e31a1c"]
size=5000
filename='./{}/{}{}_spin_mass_dist.pdf'.format(fig_dir,label,add_label)
plot_ma_dist(post,colors,size,filename,plot_LIGO=False,iidct=False,linear_a=False,twopeak=False,plot_inj=False)

##################################################
#plot spin corner
##################################################
print('ploting spin corner')

filename='./{}/{}{}_spin_corner.pdf'.format(fig_dir,label,add_label)
params=['mu_a1','sigma_a1','mu_a2','sigma_a2','amin2','amax1','amax2','sigma_t','zeta']
show_keys=[r'$\mu_{\rm a,1}$', r'$\sigma_{\rm a,1}$',r'$\mu_{\rm a,2}$', r'$\sigma_{\rm a,2}$',r'$a_{\rm min,2}$',r'$a_{\rm max,1}$',r'$a_{\rm max,2}$', r'$\sigma_{\rm t}$', r'$\zeta$']
color='orange'
plot_corner(post,params,show_keys,color,filename,smooth=1.)

print('ploting sigma_t, zeta, f_neg corner')
post=calculated_fneg(post)
plot_corner(post,params=['sigma_t','zeta','f_neg'],show_keys=[r'$\sigma_{{\rm t}}$', r'$\zeta$', r'$f_{\rm neg}$'],color='goldenrod',filename='./{}/{}{}_sigmat_corner.pdf'.format(fig_dir,label,add_label))

##################################################
#plot re-weighted samples
##################################################

"""
#generate pop-informed samples
print('ploting re-weighted samples')
with open('./data/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples, evidences = pickle.load(fp)
post.pop('lgR0')
pop_informed_samples(posteriors=samples,post=post,size=1000,filename='./{}/{}{}_informed.pickle'.format(outdir,label,add_label))
"""
#read pop-infromed samples
filename='./{}/{}{}_m_vs_a_reweighted_events.pdf'.format(fig_dir,label,add_label)
with open('./{}/{}{}_informed.pickle'.format(outdir,label,add_label), 'rb') as fp:
    informed_samples = pickle.load(fp)
#plot pop-infromed events
plot_ma_sample(samples=informed_samples,size=1000,filename=filename)

##################################################
#plot Branch ratios of hierarchical mergers
##################################################
print('ploting Supplementary Figure 6')

#calculate the branch ratios
filename='./{}/{}{}_branch_ratio.pickle'.format(outdir,label,add_label)

cal_branch_ratios(post,filename,size=None)

with open(filename, 'rb') as fp:
    post_ratio = pickle.load(fp)
R_H=10**post_ratio['lgR0']*(post_ratio['plh']+post_ratio['phl']+post_ratio['phh'])
    
R_H=np.array(R_H)
R_H_pup=np.percentile(R_H,95,axis=0)
R_H_plow=np.percentile(R_H,5,axis=0)
R_H_pmid=np.percentile(R_H,50,axis=0)
print(R_H_pmid,R_H_pup-R_H_pmid,R_H_plow-R_H_pmid)

print('ploting')
params=['pll','plh','phl','phh']
for key in params:
    post_ratio[key]=post_ratio[key]*100
show_keys = np.array([r'$r_{\rm low,low}[\%]$', r'$r_{\rm low,high}[\%]$',r'$r_{\rm high,low}[\%]$',r'$r_{\rm high,high}[\%]$'])
filename='./{}/{}{}_branch_ratios_corner.pdf'.format(fig_dir,label,add_label)
color='skyblue'
plot_corner(post_ratio,params,show_keys,color,filename,smooth=0.5)

##################################################
#plot mass posterior corner
##################################################
print('ploting mass posterior corner')

filename='./{}/{}_mass_corner.pdf'.format(fig_dir,label,add_label)
params=['beta','mmin1','mmax1','alpha1','r2','mmin2','mmax2','alpha2']
show_keys=[r'$\beta$',r'$m_{\rm min,1}$', r'$m_{\rm max,1}$', r'$\alpha_1$', r'$r_2$',r'$m_{\rm min,2}$', r'$m_{\rm max,2}$', r'$\alpha_2$']
color='green'
plot_corner(post,params,show_keys,color,filename,smooth=0.5)

##################################################
#plot mmax HPD CI
##################################################
print('ploting mmax HPD CI')

from plot_utils import plot_mmax
plot_mmax(post=post,filename='./{}/mmax{}.pdf'.format(fig_dir,add_label))

##################################################
#plot Figure mass1 mass2
##################################################
print('ploting Figure mass1 mass2 distribution')

colors=["#08306b","#4d9221",'orange',"#e31a1c"]
size=2000
filename='./{}/{}{}_m1_m2_dist.pdf'.format(fig_dir,label,add_label)
plot_m1_m2_dist(post,colors,size,filename)

##################################################
#plot Figure mass ratio
##################################################
print('ploting Figure mass ratio distribution')

colors=["#08306b","#4d9221",'orange',"#e31a1c"]
size=2000
filename='./{}/{}{}_q_dist.pdf'.format(fig_dir,label,add_label)
plot_q_dist(post,colors,size,filename)

##################################################
#plot chieff v.s. q
##################################################
#read pop-infromed samples
filename='./{}/G2_chieff_vs_q_reweighted{}_events.pdf'.format(fig_dir,add_label)
with open('./{}/Double_spin_informed.pickle'.format(outdir), 'rb') as fp:
    informed_samples = pickle.load(fp)
    
#plot pop-infromed events
plot_qchieff_sample(samples=informed_samples,size=1000,filename=filename)

##################################################
#plot Figures for One-component Step
##################################################
"""
with open('./{}/Single_step_result.json'.format(outdir)) as a:
    data=json.load(a)
post=data['posterior']['content']
with open('./{}/Single_step_post.pickle'.format(outdir), 'wb') as fp:
    pickle.dump((post), fp)
"""
with open('./{}/Single_step_post.pickle'.format(outdir), 'rb') as fp:
    post = pickle.load(fp)

colors=["#08306b","#4d9221","#e31a1c",'orange']
size=5000
filename='./{}/Single_step_dist.pdf'.format(fig_dir)
plot_step_ma_dist(post,colors,size,filename)

filename='./{}/Single_step_spin_corner.pdf'.format(fig_dir)
params=['mu_a1','sigma_a1','mu_a2','sigma_t','zeta','md']
show_keys=[r'$\mu_{\rm a,1}$', r'$\sigma_{\rm a,1}$',r'$\mu_{\rm a,2}$', r'$\sigma_{\rm t}$', r'$\zeta$',r'$m_{\rm d}[M_{\odot}]$']
color='orange'
plot_corner(post,params,show_keys,color,filename,smooth=1.5)


