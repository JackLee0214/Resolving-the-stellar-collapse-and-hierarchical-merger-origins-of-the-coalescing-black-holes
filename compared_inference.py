import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import pickle
import h5py
from astropy.cosmology import Planck15
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.result import read_in_result as rr
from bilby.core.prior import Uniform, LogUniform, PowerLaw
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Beta as Bt
from scipy.interpolate import interp1d
import astropy.units as u
import sys
from scipy.special._ufuncs import xlogy, erf

outdir='results'

##########################################################
# choose a model
##########################################################

label='PS_default'
#label='PP_default'
#label='PS_bimodal'
#label='PS_linear'

################################################################
#read data
################################################################
with open('./data/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples, evidences = pickle.load(fp)
ln_evidences=np.log(evidences)

###################################################
#mass model
###################################################

from model_libs import PS_mass, PL, smooth

def PP_m1m2_un(m1,m2,alpha,mmin,mmax,delta,mu,sigma,r_peak,beta):
    pm1=(PowerLaw(-alpha,mmin,mmax).prob(m1)*(1-r_peak)+TG(mu,sigma,mmin,mmax).prob(m1)*r_peak)*smooth(m1,mmin,delta)
    pm2=PL(m2,mmin,m1,-beta,delta)
    pdf = pm1*pm2
    return np.where((m2<m1), pdf , 1e-1000)
    
def PP_m1m2(m1,m2,alpha,mmin,mmax,delta,mu,sigma,r_peak,beta,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    m1_sam = np.linspace(mmin,mmax,500)
    m2_sam = np.linspace(mmin,mmax,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = PP_m1m2_un(x,y,alpha,mmin,mmax,delta,mu,sigma,r_peak,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = PP_m1m2_un(m1,m2,alpha,mmin,mmax,delta,mu,sigma,r_peak,beta)/AMP1
    return np.where((m2<m1), pdf , 1e-1000)

def PS_m1m2_un(m1,m2,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta):
    pm1=PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)
    pm2=PL(m2,mmin,m1,-beta,delta)
    pdf = pm1*pm2
    return np.where((m2<m1), pdf , 1e-1000)
    
def PS_m1m2(m1,m2,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta,mu_a=None,sigma_a=None,mu_al=None,sigma_al=None,mu_ar=None,sigma_ar=None,sigma_t=None,zeta=None,mu_a2=None,sigma_a2=None,sigma_t2=None,r=None,zmin=None):
    m1_sam = np.linspace(mmin,mmax,500)
    m2_sam = np.linspace(mmin,mmax,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = PS_m1m2_un(x,y,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = PS_m1m2_un(m1,m2,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta)/AMP1
    return np.where((m2<m1), pdf , 1e-1000)
 

########################################################
# spin model
########################################################

def default_spin(a1,a2,ct1,ct2,mu_a,sigma_a,sigma_t,zeta):
    t=mu_a*(1-mu_a)/sigma_a**2-1
    alpha_a=mu_a**2*(1-mu_a)/sigma_a**2-mu_a
    beta_a=t-alpha_a
    pa=(Bt(alpha_a,beta_a).prob(a1.reshape(-1))*Bt(alpha_a,beta_a).prob(a2.reshape(-1))).reshape(a1.shape)
    p1ct=TG(1,sigma_t,-1,1).prob(ct1)*TG(1,sigma_t,-1,1).prob(ct2)
    p2ct=Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)
    pct = p1ct*zeta+(1-zeta)*p2ct
    return pct*pa
                   
def default_spin_constraint(params):
    if label=='PS_bimodal':
        params['constraint']=np.sign(params['mu_a2']-params['mu_a'])-1
    elif label=='PS_linear':
        params['constraint']= 0
    else:
        mu_a=params['mu_a']
        sigma_a=params['sigma_a']
        t=mu_a*(1-mu_a)/sigma_a**2-1
        alpha_a=mu_a**2*(1-mu_a)/sigma_a**2-mu_a
        beta_a=t-alpha_a
        params['constraint']=np.sign(alpha_a)+np.sign(beta_a)-2
    return params

def Double_spin(a1,a2,ct1,ct2,mu_a,sigma_a,sigma_t,zeta,mu_a2,sigma_a2,sigma_t2,r):
    p1=TG(mu_a,sigma_a,0,1).prob(a1)*(zeta*TG(1,sigma_t,-1,1).prob(ct1)+(1-zeta)*Uniform(-1,1).prob(ct1))*r+\
        TG(mu_a2,sigma_a2,0,1).prob(a1)*TG(1,sigma_t2,-1,1).prob(ct1)*(1-r)
    p2=TG(mu_a,sigma_a,0,1).prob(a2)*(zeta*TG(1,sigma_t,-1,1).prob(ct2)+(1-zeta)*Uniform(-1,1).prob(ct2))*r+\
        TG(mu_a2,sigma_a2,0,1).prob(a2)*TG(1,sigma_t2,-1,1).prob(ct2)*(1-r)
    return p1*p2
        
###################################
#liner spin
###################################
def line_model(x,x1,x2,y1,y2):
    y=((x-x1)*y2+(x2-x)*y1)/(x2-x1)*(x>x1)*(x2>x)
    y=y*(x>x1)*(x<x2)+y1*(x<x1)+y2*(x>x2)
    return y
def T_gaussian_a(a,mu,sigma):
    normalisation = (erf((1 - mu) / 2 ** 0.5 / sigma) - erf(
            (0 - mu) / 2 ** 0.5 / sigma)) / 2
    pa = np.where((0 < a) & (a<1), np.exp(-(mu - a) ** 2 / (2 * sigma ** 2)) / (2 * np.pi) ** 0.5 / sigma / normalisation, 1e-100)
    return pa

def mdependent_a(a,m,mmin,mmax,mu_al,mu_ar,sigma_al,sigma_ar):
    mu_a= line_model(m,mmin ,mmax ,mu_al ,mu_ar)
    sigma_a= line_model(m,mmin ,mmax ,sigma_al ,sigma_ar)
    return T_gaussian_a(a,mu_a,sigma_a)

def default_ct(ct1,sigma_t1,zeta1,zmin1):
    return TG(1,sigma_t1,zmin1,1).prob(ct1)*zeta1+Uniform(-1,1).prob(ct1)*(1-zeta1)

def line_spin(a1,a2,ct1,ct2,m1,m2,mmin,mmax,mu_al,mu_ar,sigma_al,sigma_ar,sigma_t,zeta,zmin):
    p1=mdependent_a(a1,m1,mmin,mmax,mu_al,mu_ar,sigma_al,sigma_ar)*default_ct(ct1,sigma_t,zeta,zmin)
    p2=mdependent_a(a2,m2,mmin,mmax,mu_al,mu_ar,sigma_al,sigma_ar)*default_ct(ct2,sigma_t,zeta,zmin)
    return p1*p2

###################################################
#redshift
###################################################
from model_libs import llh_z, p_z, log_N

###################################################
#hyper prior
###################################################
def hyper_PS_default(dataset, alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta, gamma,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    z = dataset['z']
    m1,m2 = dataset['m1'],dataset['m2']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    hp = PS_m1m2(m1,m2,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta)*llh_z(z,gamma)
    hp = hp*default_spin(a1,a2,ct1,ct2,mu_a,sigma_a,sigma_t,zeta)
    return hp

def hyper_PS_2s_uncorr(dataset, alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta, gamma,mu_a=None,sigma_a=None,sigma_t=None,zeta=None,mu_a2=None,sigma_a2=None,sigma_t2=None,r=None):
    z = dataset['z']
    m1,m2 = dataset['m1'],dataset['m2']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    hp = PS_m1m2(m1,m2,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta)*llh_z(z,gamma)
    hp = hp*Double_spin(a1,a2,ct1,ct2,mu_a,sigma_a,sigma_t,zeta,mu_a2,sigma_a2,sigma_t2,r)
    return hp
    
def hyper_PP_default(dataset, alpha,mmin,mmax,delta,mu,sigma,r_peak,beta, gamma,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    z = dataset['z']
    m1,m2 = dataset['m1'],dataset['m2']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    hp = PP_m1m2(m1,m2,alpha,mmin,mmax,delta,mu,sigma,r_peak,beta)*llh_z(z,gamma)
    hp = hp*default_spin(a1,a2,ct1,ct2,mu_a,sigma_a,sigma_t,zeta)
    return hp
    
def hyper_PS_linear(dataset, alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta, gamma,mu_al=None,sigma_al=None,mu_ar=None,sigma_ar=None,sigma_t=None,zeta=None,zmin=None):
    z = dataset['z']
    m1,m2 = dataset['m1'],dataset['m2']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    hp = PS_m1m2(m1,m2,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,beta)*llh_z(z,gamma)
    hp = hp*line_spin(a1,a2,ct1,ct2,m1,m2,mmin,mmax,mu_al,mu_ar,sigma_al,sigma_ar,sigma_t,zeta,zmin)
    return hp

#priors
priors=bilby.prior.PriorDict(conversion_function=default_spin_constraint)
priors.update(dict(mu_a = Uniform(0., 1., 'mu_a', '$mu_{\\rm a}$'),
                    sigma_a = Uniform(0.05, 0.5, 'sigma_a', '$\\sigma_{\\rm a}$'),
                    zeta = Uniform(0., 1., 'zeta', '$\\zeta$'),
                    sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                    constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
                     ))
priors.update(dict(lgR0 = Uniform(0,3),
                    gamma=2.7,
                    beta = Uniform(0,8.),
                    mmin = Uniform(2., 10., 'mmin', '$m_{\\rm min}$'),
                    mmax = Uniform(50., 100, 'mmax', '$m_{\\rm max}$'),
                    alpha = Uniform(-4., 8., 'alpha', '$\\alpha$'),
                    delta = Uniform(0., 10., 'delta', '$\\delta_{\\rm m}$')
                        ))

    
###############################################################
#model selection
###############################################################
if label=='PP_default':
    mass_model = PP_m1m2
    hyper_prior = hyper_PP_default
    priors.update(dict(mu=Uniform(20,50,'mu','$\\mu$'),
                       sigma=Uniform(1,10,'sigma','$\\sigma$'),
                       r_peak=Uniform(0,1,'r_peak','$r_{\\rm peak}$')
                       ))
else:
    mass_model = PS_m1m2
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    if label=='PS_default':
        hyper_prior = hyper_PS_default
    if label=='PS_bimodal':
        hyper_prior = hyper_PS_2s_uncorr
        priors.update(dict(mu_a2 = Uniform(0., 1., 'mu_a2', '$mu_{\\rm a2}$'),
            sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a2}$'),
            r = Uniform(0., 1., 'r', '$r_{\\rm low}$'),
            sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t2}$')))
    if label=='PS_linear':
        hyper_prior = hyper_PS_linear
        priors.pop('mu_a')
        priors.pop('sigma_a')
        priors.update(dict(mu_al = Uniform(0., 1., 'mu_al', '$mu_{\\rm a,l}$'),
                        sigma_al = Uniform(0.05, 0.5, 'sigma_al', '$\\sigma_{\\rm a,l}$'),
                        mu_ar = Uniform(0., 1., 'mu_ar', '$mu_{\\rm a,r}$'),
                        sigma_ar = Uniform(0.05, 0.5, 'sigma_ar', '$\\sigma_{\\rm a,r}$'),
                        zmin = -1))

####################################################################################
#selection effects
####################################################################################
from model_libs import Rate_selection_function_with_uncertainty

class Hyper_selection_with_var(HyperparameterLikelihood):
    
    def likelihood_obs_var(self):
        weights = self.hyper_prior.prob(self.data) / self.data['prior']
        expectations = np.mean(weights, axis=-1)
        square_expectations = np.mean(weights**2, axis=-1)
        variances = (square_expectations - expectations**2) / (
            self.samples_per_posterior * expectations**2
        )
        variance = np.sum(variances)
        Neffs = expectations**2/square_expectations*self.samples_per_posterior
        Neffmin = np.min(Neffs)
        return variance, Neffmin
        
    def log_likelihood(self):
        selection, sel_vars, sel_Neff = Rate_selection_function_with_uncertainty(self.n_posteriors, mass_model, **self.parameters)
        obs_vars, obs_Neff = self.likelihood_obs_var()
        if ((sel_Neff>4*self.n_posteriors) & (obs_Neff>10)):
            return self.noise_log_likelihood() + self.log_likelihood_ratio()+ selection
        else:
            return -1e+100
            
hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)

bilby.core.utils.setup_logger(outdir=outdir, label=label)
result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler='pymultinest', nlive=1000, use_ratio=False, outdir=outdir, label=label)
