import numpy as np
from scipy.interpolate import CubicSpline
import bilby
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Uniform
from bilby.core.sampler import run_sampler
from bilby.hyper.likelihood import HyperparameterLikelihood
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
import astropy.units as u
import h5py
from scipy.special import erf

###########################################################################################################
#
#mass and spin
#
###########################################################################################################

def TG_analy(x,mu,sig,min,max):
    norm = (erf((max - mu) / 2 ** 0.5 / sig) - erf(
            (min - mu) / 2 ** 0.5 / sig)) / 2
    pdf = np.exp(-(mu - x) ** 2 / (2 * sig ** 2)) / (2 * np.pi) ** 0.5 \
            / sig / norm
    return np.where((min<x) & (x<max), pdf , 1e-10000)

############
#mass
############
def smooth(m,mmin,delta):
    A = (m-mmin == 0.)*1e-10 + (m-mmin)
    B = (m-mmin-delta == 0.)*1e-10 + abs(m-mmin-delta)
    f_m_delta = delta/A - delta/B
    return (np.exp(f_m_delta) + 1.)**(-1.)*(m<=(mmin+delta))+1.*(m>(mmin+delta))

def PL(m1,mmin,mmax,alpha,delta):
    norm=(mmax**(1-alpha)-mmin**(1-alpha))/(1-alpha)
    pdf = m1**(-alpha)/norm*smooth(m1,mmin,delta)
    return np.where((mmin<m1) & (m1<mmax), pdf , 1e-10000)

def PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12):
    xi=np.exp(np.linspace(np.log(6),np.log(80),12))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(2,100,1000)
    yy=np.exp(cs(xx)*(xx>6)*(xx<80))*PL(xx,mmin,mmax,alpha,delta)
    norm=np.sum(yy)*98./1000.
    pm1 = np.exp(cs(m1)*(m1>6)*(m1<80))*PL(m1,mmin,mmax,alpha,delta)/norm
    return pm1
        
####################################
#spin
####################################

#magnitude
def spin_a(a1,mu_a,sigma_a,amin,amax):
    return TG(mu_a,sigma_a,amin,amax).prob(a1)
#Double_peak
def two_peak_a(a1,mu_a,sigma_a,mu_a3,sigma_a3,rp,amin,amax):
    return TG(mu_a,sigma_a,amin,amax).prob(a1)*(1-rp)+TG(mu_a3,sigma_a3,amin,amax).prob(a1)*rp
#step mu
def step_a(a1,m,mu_a,sigma_a,mu_a2,md,amin,amax):
    return TG(mu_a,sigma_a,amin,amax).prob(a1)*(m<md)+TG(mu_a2,sigma_a,amin,amax).prob(a1)*(m>md)
#cosine tilt angle
def Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin):
    return TG(mu_t,sigma_t,zmin,1).prob(ct1)*TG(mu_t,sigma_t,zmin,1).prob(ct2)*zeta+\
        Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)*(1-zeta)
        
def spin_ct(ct1,mu_t,sigma_t,zeta,zmin):
    return TG(mu_t,sigma_t,zmin,1).prob(ct1)*zeta+Uniform(-1,1).prob(ct1)*(1-zeta)

####################################
#only mass model
####################################
#Single
def Single_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta):
    pm1 = PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pm2 = PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-10000)
    
def Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Single_mass_pair_un(x,y, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Single_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)/AMP1
    return pdf

#Double
def Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2):
    p1=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)
    p2=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2
    return p1+p2

def Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta):
    pm1=Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pm2=Double_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-10000)
        
def Double_mass_pair_branch(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    p11=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)*\
        PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)*(m2/m1)**beta/AMP1
    p12=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)*\
        PS_mass(m2,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2*(m2/m1)**beta/AMP1
    p21=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2*\
        PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)*(m2/m1)**beta/AMP1
    p22=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2*\
        PS_mass(m2,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2*(m2/m1)**beta/AMP1
    return np.where((m2<m1),p11,1e-10000), np.where((m2<m1),p12,1e-10000),np.where((m2<m1),p21,1e-10000),np.where((m2<m1),p22,1e-10000)
    
########################
#mass vs spin model
########################

#Double
def PS_ma(m1,a1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a,sigma_a,amin,amax):
    pdf=PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*spin_a(a1,mu_a,sigma_a,amin,amax)
    return pdf

def Double_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2
    return p1+p2

def Double_ma_iidct(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)*spin_ct(ct1,mu_t1,sigma_t1,zeta1,zmin1)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2*spin_ct(ct1,mu_t2,sigma_t2,zeta2,zmin2)
    return p1+p2

def Double_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    pma1=Double_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pma2=Double_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_ma_iidct_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta):
    pmact1=Double_ma_iidct(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2)
    pmact2=Double_ma_iidct(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2)
    pdf = pmact1*pmact2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)
    
def Double_mact_unpair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,mu_t,sigma_t,zmin,zeta):
    pma1=Double_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pma2=Double_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct
    return np.where((m2<m1), pdf*2 , 1e-100)

def Double_mact_unpair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,mu_t,sigma_t,zmin,zeta):
    pm1=Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pm2=Double_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pdf = pm1*pm2*1/4.
    return np.where((m2<m1), pdf*2 , 1e-100)

def Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)/AMP1
    return pdf

def Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)/AMP1
    pdf_spin = 1/4.
    return pdf*pdf_spin

def Double_ma_iidct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_ma_iidct_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta)/AMP1
    return pdf

def Double_ma_iidct_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)/AMP1
    pdf_spin = 1/4.
    return pdf*pdf_spin

#Single
def Single_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,mu_t,sigma_t,zmin,zeta):
    pm=Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    pact=spin_a(a1,mu_a1,sigma_a1,amin1,amax1)*spin_a(a2,mu_a1,sigma_a1,amin1,amax1)*Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pm*pact
    return pdf
    
def Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,mu_t,sigma_t,zmin,zeta,mu_a2=None,md=None):
    pdf = Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)*1/4.
    return pdf

def Single_mact_unpair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zmin,zeta):
    pm1 = PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pm2 = PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pdf = pm1*pm2*spin_a(a1,mu_a1,sigma_a1,amin1,amax1)*spin_a(a2,mu_a1,sigma_a1,amin1,amax1)*Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    return np.where((m2<m1), pdf*2 , 1e-100)
    
def Single_mact_unpair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zmin,zeta):
    pm1 = PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pm2 = PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pdf = pm1*pm2*1/4.
    return np.where((m2<m1), pdf*2 , 1e-100)
    
#Single step
def Single_step_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a2,md,amax1,beta,mu_t,sigma_t,zmin,zeta):
    pm=Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    pact=step_a(a1,m1,mu_a1,sigma_a1,mu_a2,md,amin1,amax1)*step_a(a2,m2,mu_a1,sigma_a1,mu_a2,md,amin1,amax1)*Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pm*pact
    return pdf

#Double peaks
def PS_two_peak(m1,a1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a,sigma_a,mu_a3,sigma_a3,rp,amin,amax):
    pdf=PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*two_peak_a(a1,mu_a,sigma_a,mu_a3,sigma_a3,rp,amin,amax)
    return pdf

def Double_two_peak_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2):
    p1=PS_two_peak(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amin1,amax1)*(1-r2)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2
    return p1+p2

def Double_two_peak_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    pma1=Double_two_peak_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pma2=Double_two_peak_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_two_peak_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_two_peak_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)/AMP1
    return pdf

def Double_two_peak_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)/AMP1
    pdf_spin = 1/4.
    return pdf*pdf_spin
    
###################################
#liner spin
###################################
def line_model(x,x1,x2,y1,y2):
    y=((x-x1)*y2+(x2-x)*y1)/(x2-x1)*(x>x1)*(x2>x)
    y=y*(x>x1)*(x<x2)+y1*(x<x1)+y2*(x>x2)
    return y

def mdependent_a(a,m,mmin,mmax,amin,mu_al,mu_ar,sigma_a,amax):
    mu_a= line_model(m,mmin ,mmax ,mu_al ,mu_ar)
    return TG_analy(a,mu_a,sigma_a,min=amin,max=amax)

def PS_ma_line(m1,a1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin,mu_al,mu_ar,sigma_a,amax):
    pdf=PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*mdependent_a(a1,m1,mmin,mmax,amin,mu_al,mu_ar,sigma_a,amax)
    return pdf

def Double_ma_line(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                         alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2):
    p1=PS_ma_line(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1)*(1-r2)
    p2=PS_ma_line(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2)*r2
    return p1+p2

def Double_mact_line_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                                                 alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    pma1=Double_ma_line(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                         alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2)
    pma2=Double_ma_line(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                         alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_mact_line_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                                              alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mact_line_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                                                 alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)/AMP1
    return pdf
    
def Double_mact_line_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                                              alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)/AMP1
    return pdf*1/4.
#########################################################################################################
#
#conversion function
#
#########################################################################################################
#Let the second subpopulations spin faster than the first, if exist.
#Each subpopulation has a mass range wider than 10 Msun.

def Double_constraint(params):
    params['constraint1']=np.sign(params['amax1']-params['mu_a1'])-1
    params['constraint2']=np.sign(params['mu_a2']-params['mu_a1'])-1
    params['constraint3']=np.sign(params['mu_a2']-params['amin2'])-1
    params['constraint4']=np.sign(params['amax2']-params['mu_a2'])-1
    params['constraint5']=np.sign(params['amax2']-params['amin2']-0.2)-1
    params['constraint6']=np.sign(params['mmax2']-params['mmin2']-20)-1
    params['constraint7']=np.sign(params['mmax1']-params['mmin1']-20)-1
    
    return params

#priors
def Double_priors():
    priors=bilby.prior.PriorDict(conversion_function=Double_constraint)
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$'),
                    
                    delta2=Uniform(1,10),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 100, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
                    mu_a2 = Uniform(0, 1., 'mu_a2', '$mu_{\\rm a,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
                    amax2 = Uniform(0.2,1, 'amax2', '$a_{\\rm max,2}$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    beta = Uniform(0,6,'beta','$\\beta$'),
                    mu_t = 1,
                    sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                    zmin = -1,
                    zeta = Uniform(0,1,'zeta','$\\zeta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
    priors.update({'o1':0,'o'+str(12): 0})
    priors.update({'constraint'+str(i+1):bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for i in np.arange(7)})
    return priors

#Double peak
def Double_two_peak_constraint(params):
    params=Double_constraint(params)
    params['constraint8']=np.sign(params['mu_a3']-params['mu_a1'])-1
    params['constraint9']=np.sign(params['amax1']-params['mu_a3'])-1
    return params

def Double_two_peak_priors():
    priors=bilby.prior.PriorDict(conversion_function=Double_two_peak_constraint)
    priors.update(Double_priors())
    priors.update(dict(
                    mu_a3 = Uniform(0, 1., 'mu_a3', '$mu_{\\rm a,3}$'),
                    sigma_a3 = Uniform(0.05, 0.5, 'sigma_a3', '$\\sigma_{\\rm a,3}$'),
                    rp = Uniform(0,1, 'rp', '$r_p$')))
    priors.update({'constraint'+str(i+8):bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for i in np.arange(2)})
    return priors

#Double iid ct
def Double_iidct_priors():
    priors=Double_priors()
    for key in ['mu_t','sigma_t','zmin','zeta']:
        priors.pop(key)
    priors.update(dict(mu_t1 = 1,
                    sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t,1}$'),
                    zmin1 = -1,
                    zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),
                    mu_t2 = 1,
                    sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t,2}$'),
                    zmin2 = -1,
                    zeta2 = Uniform(0,1,'zeta2','$\\zeta_2$')))
    return priors
    
#Double_linear

def Double_line_constraint(params):
    params['constraint1']=np.sign(params['mu_ar2']-params['mu_al1'])-1
    params['constraint2']=np.sign(params['mmax2']-params['mmin2']-20)-1
    params['constraint3']=np.sign(params['mmax1']-params['mmin1']-20)-1
    
    params['constraint4']=np.sign(params['amax1']-params['mu_ar1'])-1
    params['constraint5']=np.sign(params['amax1']-params['mu_al1'])-1
    params['constraint6']=np.sign(params['mu_ar2']-params['amin2'])-1
    params['constraint7']=np.sign(params['mu_al2']-params['amin2'])-1
    params['constraint8']=np.sign(params['amax2']-params['mu_ar2'])-1
    params['constraint9']=np.sign(params['amax2']-params['mu_al2'])-1
    params['constraint10']=np.sign(params['amax2']-params['amin2']-0.2)-1
    return params

def Double_line_priors():
    priors=bilby.prior.PriorDict(conversion_function=Double_line_constraint)
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    
                    delta2=Uniform(1,10),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 100, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    beta = Uniform(0,6,'beta','$\\beta$'),
                    mu_t = 1,
                    sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                    zmin = -1,
                    zeta = Uniform(0,1,'zeta','$\\zeta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7,
                    amin1=0,
                    mu_al1 = Uniform(0., 1., 'mu_al1', '$\\mu_{\\rm al,1}$'),
                    mu_ar1 = Uniform(0., 1., 'mu_ar1', '$\\mu_{\\rm ar,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$'),
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
                    mu_al2 = Uniform(0., 1., 'mu_al2', '$mu_{\\rm al,2}$'),
                    mu_ar2 = Uniform(0., 1., 'mu_ar2', '$mu_{\\rm ar,2}$'),
                    amax2 = Uniform(0.2,1, 'amax2', '$a_{\\rm max,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$')
                    ))
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
    priors.update({'o1':0,'o'+str(12): 0})
    priors.update({'constraint'+str(i+1):bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for i in np.arange(10)})
    return priors
            
    
#Single

def Single_constraint(params):
    params['constraint']=0
    params['constraint']+=np.sign(params['amax1']-params['mu_a1'])-1
    
    return params

def Single_priors():
    priors=bilby.prior.PriorDict(conversion_function=Single_constraint)
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 10., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(50., 100, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$'),
                    mu_t = 1,
                    sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                    zmin = -1,
                    zeta = Uniform(0,1,'zeta','$\\zeta$'),

                    beta = Uniform(0,6,'beta','$\\beta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7,
                    
                    constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    return priors
#Single step

def Single_step_constraint(params):
    params['constraint']=0
    params['constraint']+=np.sign(params['amax1']-params['mu_a2'])-1
    params['constraint']+=np.sign(params['amax1']-params['mu_a1'])-1
    return params
def Single_step_priors():
    priors=bilby.prior.PriorDict(conversion_function=Single_step_constraint)
    single_priors=Single_priors()
    priors.update(single_priors)
    priors.update(dict(
                    mu_a2 = Uniform(0., 1., 'mu_a2', '$\\mu_{\\rm a,2}$'),
                    md = Uniform(2., 100., 'md', '$m_{\\rm d}$')))
    return priors
    
#############################################################################################################################
#
#redshift
#
#############################################################################################################################
fdVcdz=interp1d(np.linspace(0,5,10000),4*np.pi*Planck15.differential_comoving_volume(np.linspace(0,5,10000)).to(u.Gpc**3/u.sr).value)
zs=np.linspace(0,1.9,1000)
dVdzs=fdVcdz(zs)
logdVdzs=np.log(dVdzs)

#The log likelihood for redshift distribution, which is (log_hyper_prior - log_default_prior), note than each prior is normalized 
def llh_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*1.9/1000.
    norm0=np.sum((1+zs)**(-1)*dVdzs)*1.9/1000.
    return np.where((z>0) & (z<1.9), (1+z)**gamma/norm*norm0 , 1e-100)

# The normalized redshift distribution: log_hyper_prior
def p_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*1.9/1000.
    p = (1+z)**(gamma-1)*fdVcdz(z)/norm
    return np.where((z>0) & (z<1.9), p , 1e-100)

# The expected number of mergers in the surveyed VT 
def log_N(T,lgR0,gamma):
    return np.log(T) + lgR0/np.log10(np.e) + np.logaddexp.reduce((gamma-1)*np.log(zs+1) + logdVdzs) + np.log(1.9/1000)

#############################################################################################################################
#
#hyper prior
#
#############################################################################################################################

#Double
def hyper_Double(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp

#Double no_sel
def hyper_Double_nosel(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta,gamma):
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)
    return hp

#Single
def hyper_Single(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Single_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp
    
def hyper_Single_step(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a2,md,amax1,beta,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Single_step_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a2,md,amax1,beta,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp

#Double_linear
def hyper_Double_line(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                                              alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_mact_line_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_al1,mu_ar1,sigma_a1,amax1,\
                                              alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_al2,mu_ar2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp
    
#Double iid costilt angle
def hyper_Double_iidct(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_ma_iidct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta)*llh_z(z,gamma)
    return hp
#Double two peak
def hyper_Double_two_peak(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_two_peak_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,mu_a3,sigma_a3,rp,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp
###########################################################################
#unpaired model
#####################################################################

def hyper_Double_unpair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_mact_unpair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp
    
def hyper_Single_unpair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Single_mact_unpair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp

#####################################################################
#selection effects with injection campaign
#####################################################################

#Transfrom spinx, spiny, spinz to spin magnitude
def a_from_xyz(x,y,z):
    return (x**2+y**2+z**2)**0.5

#Transfrom spinx, spiny, spinz to cosine tilt angle
def ct_from_xyz(x,y,z):
    return z/(x**2+y**2+z**2)**0.5

#The injection pdf of spins described by spin magnitude and cosine tilt angle
def act_Uniform(a1,a2,ct1,ct2):
    pa=Uniform(0,1).prob(a1)*Uniform(0,1).prob(a1)
    pct=Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)
    return pa*pct

inject_dir='./data/'
with h5py.File(inject_dir+'o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5', 'r') as f:
    Tobs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    Ndraw = f.attrs['total_generated']
    
    m1_inj = np.array(f['injections/mass1_source'])
    m2_inj = np.array(f['injections/mass2_source'])
    z_inj = np.array(f['injections/redshift'])
    log1pz_inj = np.log1p(z_inj)
    logdVdz_inj = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z_inj).to(u.Gpc**3/u.sr).value)
   
    s1z_inj = np.array(f['injections/spin1z'])
    s2z_inj = np.array(f['injections/spin2z'])
    s1x_inj = np.array(f['injections/spin1x'])
    s2x_inj = np.array(f['injections/spin2x'])
    s1y_inj = np.array(f['injections/spin1y'])
    s2y_inj = np.array(f['injections/spin2y'])
    a1_inj = a_from_xyz(s1x_inj,s1y_inj,s1z_inj)
    ct1_inj = ct_from_xyz(s1x_inj,s1y_inj,s1z_inj)
    a2_inj = a_from_xyz(s2x_inj,s2y_inj,s2z_inj)
    ct2_inj = ct_from_xyz(s2x_inj,s2y_inj,s2z_inj)
    
    p_draw = np.array(f['injections/sampling_pdf'])
    logpdraw = np.log(p_draw)

    gstlal_ifar = np.array(f['injections/ifar_gstlal'])
    pycbc_ifar = np.array(f['injections/ifar_pycbc_hyperbank'])
    pycbc_bbh_ifar = np.array(f['injections/ifar_pycbc_bbh'])
    opt_snr = np.array(f['injections/optimal_snr_net'])
    name = np.array(f['injections/name'])

snr_thr = 10.
ifar_thr = 1.
detection_selector_O3 = (gstlal_ifar > ifar_thr) | (pycbc_ifar > ifar_thr) | (pycbc_bbh_ifar > ifar_thr)
detection_selector_O12 = (opt_snr > snr_thr)
detection_selector = np.where(name == b'o3', detection_selector_O3,detection_selector_O12)

#####################################################################
#spin pdf on xyz, which is injected
def log_SpinF():
    r1=s1x_inj**2+s1y_inj**2+s1z_inj**2
    r2=s2x_inj**2+s2y_inj**2+s2z_inj**2
    return -2*np.log(4*np.pi)-np.log(r1)-np.log(r2)
logpspin_inj=log_SpinF()
#####################################################################
#spin pdf on magnitude and cosine tilt angle, whcih is needed
logpact_inj=np.log(act_Uniform(a1_inj,a2_inj,ct1_inj,ct2_inj))
#reweight the draw pdf
logpdraw=logpdraw-logpspin_inj+logpact_inj

#This selection effect accounts for spin distribution
def Rate_selection_function_with_uncertainty(Nobs,mass_spin_model,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdmds = np.log(mass_spin_model(m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,**kwargs))
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdmds, np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds - logpdraw) - np.log(Ndraw)
    term1 = Nobs*log_N(Tobs,lgR0,gamma)
    term2 = -np.exp(log_Nexp)
    selection=term1 + term2
    logmu=log_Nexp-log_N(Tobs,lgR0,gamma)
    varsel= np.sum(np.exp(2*(np.log(Tobs)+log_dNdzdmds - logpdraw-log_N(Tobs,lgR0,gamma)- np.log(Ndraw))))-np.exp(2*logmu)/Ndraw
    total_vars=Nobs**2 * varsel / np.exp(2*logmu)
    Neff=np.exp(2*logmu)/varsel
    return selection, total_vars, Neff


#####################################
def Event_probs(dataset,gamma,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    prior=dataset['prior']
    pdf11 = (1-r2)*(1-r2)*PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*\
                        PS_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(m2/m1)**beta
    pdf12 = r2*(1-r2)*PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*\
                        PS_ma(m2,a2,alpha2,mmin2,mmax2,delta2,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a2,sigma_a2,amin2,amax2)*(m2/m1)**beta
    pdf21 = r2*(1-r2)*PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a2,sigma_a2,amin2,amax2)*\
                        PS_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(m2/m1)**beta
    pdf22 = r2*r2*PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a2,sigma_a2,amin2,amax2)*\
                    PS_ma(m2,a2,alpha2,mmin2,mmax2,delta2,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a2,sigma_a2,amin2,amax2)*(m2/m1)**beta
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdfz = llh_z(z,gamma)/prior
    return np.array([pdf11*pct*pdfz, pdf12*pct*pdfz, pdf21*pct*pdfz, pdf22*pct*pdfz])




#######mock injection
from bilby.core.prior import PowerLaw
def PLP_m_un(m,alpha,mmin,mmax,delta,mu,sigma,r_peak):
    return (PowerLaw(-alpha,mmin,mmax).prob(m)*(1-r_peak)+TG(mu,sigma,mmin,mmax).prob(m)*r_peak)*smooth(m,mmin,delta)

def PLP_m(m,alpha,mmin,mmax,delta,mu,sigma,r_peak):
    xx=np.linspace(2,100,1000)
    yy=PLP_m_un(xx,alpha,mmin,mmax,delta,mu,sigma,r_peak)
    norm=np.sum(yy)*98./1000.
    pm1 = PLP_m_un(m,alpha,mmin,mmax,delta,mu,sigma,r_peak)/norm
    return pm1
        

