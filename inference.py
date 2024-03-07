import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import pickle
from pandas.core.frame import DataFrame
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import Uniform

from model_libs_r2 import Double_mact_pair, hyper_Double, Double_priors, Rate_selection_function_with_uncertainty
from model_libs_r2 import Double_two_peak_mact_pair_nospin, hyper_Double_two_peak, Double_two_peak_priors
from model_libs_r2 import Double_mact_line_pair, hyper_Double_line, Double_line_priors
from model_libs_r2 import Single_mact_pair, hyper_Single, Single_priors
from model_libs_r2 import Double_mact_unpair, hyper_Double_unpair, Single_mact_unpair, hyper_Single_unpair
from model_libs_r2 import hyper_Double_iidct, Double_iidct_priors, Double_ma_iidct_pair_nospin
from model_libs_r2 import Single_mact_pair_nospin, Double_mact_pair_nospin, Double_mact_unpair_nospin, Single_mact_unpair_nospin, Double_mact_line_pair_nospin
from model_libs_r2 import hyper_Single_step, Single_step_priors
from model_libs_r2 import hyper_Double_nosel

with open('./data/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples, evidences = pickle.load(fp)
"""
with open('./data/GWTC3_BBH_Mixed_10000.pickle', 'rb') as fp:
    samples, evidences = pickle.load(fp)
"""
##########################################################
# configurations for likelihood uncertainty
##########################################################
#Threshold for Neff of per-event sample
Neff_obs_thr=10
#Threshold for Neff of injection sample
Neff_sel_thr=len(samples)*4

outdir='results'
sampler='pymultinest'
npool=1
add_label=''

##########################################################
# choose a model 'Double_spin' or 'Single_spin'
##########################################################
import sys
label=['Single_spin','Double_spin','Double_linear','Double_two_peak','Double_spin_unpair','Single_step'][int(sys.argv[1])]

#'Single_spin' for One-component
#'Double_spin' for Two-component
#'Double_linear' for Two-component&LinearCorrelation
#'Double_two_peak' for Two-component&DoubleSpin
#'Double_spin_unpair' for Two-component without pairing
#'Single_step'  for One-component&Step
##########################################################
#read data
##########################################################
    
ln_evidences=np.log(evidences)
Nobs=len(samples)

##########################################################
#select models
##########################################################
#add_label=='_iso' for isotropic spin-orientation distribution
#add_label=='_mut' for variable mu_t

if label=='Single_spin':
    hyper_prior=hyper_Single
    priors=Single_priors()
    mass_spin_model=Single_mact_pair_nospin
elif label=='Single_step':
    hyper_prior=hyper_Single_step
    priors=Single_step_priors()
    mass_spin_model=Single_mact_pair_nospin
elif label=='Double_spin':
    hyper_prior=hyper_Double
    priors=Double_priors()
    mass_spin_model=Double_mact_pair_nospin
    if add_label=='_iso':
        priors['zeta']=0
        priors['sigma_t']=100
        priors['zmin']=-1
    elif add_label=='_mut':
        priors['mu_t']=Uniform(-1,1)
elif label=='Double_linear':
    hyper_prior=hyper_Double_line
    priors=Double_line_priors()
    mass_spin_model=Double_mact_line_pair_nospin
elif label=='Double_two_peak':
    hyper_prior=hyper_Double_two_peak
    priors=Double_two_peak_priors()
    mass_spin_model=Double_two_peak_mact_pair_nospin
elif label=='Double_spin_unpair':
    hyper_prior=hyper_Double_unpair
    priors=Double_priors()
    mass_spin_model=Double_mact_unpair_nospin
    priors.pop('beta')

##########################################################
#likelihood
##########################################################

class Hyper_selection_with_var(HyperparameterLikelihood):

    def likelihood_ratio_obs_var(self):
        self.hyper_prior.parameters.update(self.parameters)
        weights = np.nan_to_num(self.hyper_prior.prob(self.data) / self.data['prior'])
        expectations = np.nan_to_num(np.mean(weights, axis=-1))
        square_expectations = np.mean(weights**2, axis=-1)
        variances = (square_expectations - expectations**2) / (
            self.samples_per_posterior * expectations**2
        )
        variance = np.sum(variances)
        Neffs = expectations**2/square_expectations*self.samples_per_posterior
        Neffmin = np.min(Neffs)
        return np.nan_to_num(variance), np.nan_to_num(Neffmin), np.nan_to_num(np.sum(np.log(expectations)))
    
    def log_likelihood(self):
        obs_vars, obs_Neff, llhr= self.likelihood_ratio_obs_var()
        if (obs_Neff>Neff_obs_thr):
            #print('obs_Neff:',obs_Neff)
            selection, sel_vars, sel_Neff = Rate_selection_function_with_uncertainty(self.n_posteriors, mass_spin_model, **self.parameters)
            if ((sel_Neff>Neff_sel_thr)):
                #print('sel_Neff:',sel_Neff)
                return self.noise_log_likelihood() + llhr + selection
            else:
                return -1e100
        else:
            return -1e100
            
    def get_log_likelihood_vars(self, mass_spin_model):
        selection, sel_vars, sel_Neff = Rate_selection_function_with_uncertainty(self.n_posteriors, mass_spin_model, **self.parameters)
        obs_vars, obs_Neff, llhr = self.likelihood_ratio_obs_var()
        total_vars=sel_vars+obs_vars
        return total_vars, obs_vars, sel_vars

hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)

##########################################################
#sampling
##########################################################
bilby.core.utils.setup_logger(outdir=outdir, label=label+add_label)
result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler=sampler, nlive=1000,npool=npool,
                use_ratio=False, outdir=outdir, label=label+add_label)

plot_paras=[key for key in result.search_parameter_keys if key not in ['n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9' , 'q10', 'q11','o2','o3','o4','o5','o6','o7','o8','o9', 'o10', 'o11']]
result.plot_corner(quantiles=[0.05, 0.95],parameters=plot_paras,filename='./{}/{}_corner.pdf'.format(outdir,label+add_label),smooth=1.5,color='green')
