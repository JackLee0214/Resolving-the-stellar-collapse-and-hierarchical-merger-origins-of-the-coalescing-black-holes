import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import pickle
from pandas.core.frame import DataFrame
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import Uniform

from model_libs import Double_mass_pair, hyper_Double, Double_priors, Rate_selection_function_with_uncertainty
from model_libs import Single_mass_pair, hyper_Single, Single_priors

outdir='results'

##########################################################
# choose a model 'Double_spin' or 'Single_spin'
##########################################################

#label='Double_spin'
label='Single_spin'
add_label=''

if label=='Single_spin':
    hyper_prior=hyper_Single
    priors=Single_priors()
    mass_model=Single_mass_pair
elif label=='Double_spin':
    hyper_prior=hyper_Double
    priors=Double_priors()
    mass_model=Double_mass_pair

##########################################################
#read data 
##########################################################
with open('./data/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples, evidences = pickle.load(fp)
ln_evidences=np.log(evidences)
Nobs=len(samples)

##########################################################
#likelihood
##########################################################

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
            llh=self.noise_log_likelihood() + self.log_likelihood_ratio()+ selection
            return llh
        else:
            return -1e+100
            
    def get_log_likelihood_vars(self, mass_model):
        selection, sel_vars, sel_Neff = Rate_selection_function_with_uncertainty(self.n_posteriors, mass_model, **self.parameters)
        obs_vars, obs_Neff = self.likelihood_obs_var()
        total_vars=sel_vars+obs_vars
        return total_vars, obs_vars, sel_vars

hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)

##########################################################
#sampling
##########################################################
bilby.core.utils.setup_logger(outdir=outdir, label=label+add_label)
result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler='pymultinest', nlive=2000,
                use_ratio=False, outdir=outdir, label=label+add_label)
