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

use_ssr=1

###########################################################################################################
#
#mass and spin
#
###########################################################################################################

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

def PS_mass(m1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15):
    xi=np.exp(np.linspace(np.log(5),np.log(100),15))
    yi=np.array([N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(2,100,1000)
    yy=np.exp(cs(xx)*(xx>5))*PL(xx,mmin,mmax,alpha,delta)
    norm=np.sum(yy)*98./1000.
    pm1 = np.exp(cs(m1)*(m1>5)*(m1<100))*PL(m1,mmin,mmax,alpha,delta)/norm
    return pm1
    
############
#spin
############

def spin_a(a1,mu_a,sigma_a,amin,amax):
    return TG(mu_a,sigma_a,amin,amax).prob(a1)
   
def spin_ct(ct1,sigma_t,zeta,zmin):
    return TG(1,sigma_t,zmin,1).prob(ct1)*zeta+Uniform(-1,1).prob(ct1)*(1-zeta)

def spin_act(a1,ct1,mu_a,sigma_a,sigma_t,amin,amax,zmin,zeta):
    pa=spin_a(a1,mu_a,sigma_a,amin,amax)
    pct=spin_ct(ct1,sigma_t,zeta,zmin)
    return pct*pa+ 1e-10000

########################
#only mass model
########################

#Single
def Single_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta):
    pm1 = PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)
    pm2 = PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-10000)
    
def Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Single_mass_pair_un(x,y, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Single_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta)/AMP1
    return pdf

#Double
def Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2):
    p1=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)*(1-r2)
    p2=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15)*r2
    return p1+p2

def Double_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta):
    pm1=Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2)
    pm2=Double_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-1000)
    
def Double_mass_pair(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta)/AMP1
    return pdf


#Triple

def Triple_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3):
    p1=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15)*(1-r2-r3)
    p2=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15)*r2
    p3=PS_mass(m1,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15)*r3
    return p1+p2+p3

def Triple_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta):
    pm1=Triple_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3)
    pm2=Triple_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-1000)
        
def Triple_mass_pair(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Triple_mass_pair_un(x,y, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Triple_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta)/AMP1
    return pdf

########################
#mass vs spin model
########################

#Double
def PS_mact(m1,a1,ct1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,mu_a,sigma_a,sigma_t,amin,amax,zmin,zeta):
    pdf=PS_mass(m1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15)*spin_act(a1,ct1,mu_a,sigma_a,sigma_t,amin,amax,zmin,zeta)
    return pdf

def Double_mact(m1,a1,ct1,  alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2):
    p1=PS_mact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,mu_a1,sigma_a1,sigma_t1,amin1,amax1,zmin1,zeta1)*(1-r2)
    p2=PS_mact(m1,a1,ct1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,mu_a2,sigma_a2,sigma_t2,amin2,amax2,zmin2,zeta2)*r2
    return p1+p2

def Double_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta):
    pmact1=Double_mact(m1,a1,ct1,  alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2)
    pmact2=Double_mact(m2,a2,ct2,  alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2)
    pdf = pmact1*pmact2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta)/AMP1
    return pdf
 
#Single
def Single_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta):
    pm=Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta)
    pact=spin_act(a1,ct1,mu_a1,sigma_a1,sigma_t1,amin1,amax1,zmin1,zeta1)*spin_act(a2,ct2,mu_a1,sigma_a1,sigma_t1,amin1,amax1,zmin1,zeta1)
    pdf = pm*pact
    return pdf

#Triple

def Triple_mact(m1,a1,ct1,  alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3):
    p1=PS_mact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,mu_a1,sigma_a1,sigma_t1,amin1,amax1,zmin1,zeta1)*(1-r2-r3)
    p2=PS_mact(m1,a1,ct1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,mu_a2,sigma_a2,sigma_t2,amin2,amax2,zmin2,zeta2)*r2
    p3=PS_mact(m1,a1,ct1,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,mu_a3,sigma_a3,sigma_t3,amin3,amax3,zmin3,zeta3)*r3
    return p1+p2+p3

def Triple_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta):
    pmact1=Triple_mact(m1,a1,ct1,  alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3)
    pmact2=Triple_mact(m2,a2,ct2,  alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3)
    pdf = pmact1*pmact2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Triple_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                                        alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta):
    m1_sam = np.linspace(2,100,500)
    m2_sam = np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Triple_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Triple_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                                                alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta)/AMP1
    return pdf

#########################################################################################################
#
#conversion function
#
#########################################################################################################
#Let the second subpopulations spin faster than the first, if exist.
#Each subpopulation has a mass range wider than 10 Msun.

def Double_constraint(params):
    params['constraint']=0
    params['constraint']+=np.sign(params['amax1']-params['mu_a1'])-1
    params['constraint']+=np.sign(params['mu_a2']-params['mu_a1'])-1
    
    params['constraint']+=np.sign(params['mu_a2']-params['amin2'])-1
    params['constraint']+=np.sign(params['amax2']-params['mu_a2'])-1
    params['constraint']+=np.sign(params['amax2']-params['amin2']-0.1)-1
    
    params['constraint']+=np.sign(params['mmax2']-params['mmin2']-10)-1
    params['constraint']+=np.sign(params['mmax1']-params['mmin1']-10)-1
    
    return params

#priors
def Double_priors():
    priors=bilby.prior.PriorDict(conversion_function=Double_constraint)
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 60., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amin1 = 0,
                    amax1 = Uniform(0.1,1,'amax1', '$a_{\\rm max,1}$'),
                    sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t,1}$'),
                    zmin1 = Uniform(-1,0.9, 'zmin1', '$z_{\\rm min,1}$'),
                    zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),
                    
                    delta2=Uniform(1,10),
                    mmin2 = Uniform(2., 60., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 100, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
                    mu_a2 = Uniform(0., 1., 'mu_a2', '$mu_{\\rm a,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
                    amax2 = Uniform(0.1,1, 'amax2', '$a_{\\rm max,2}$'),
                    sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t,2}$'),
                    zmin2 = Uniform(-1,0.9, 'zmin2', '$z_{\\rm min,2}$'),
                    zeta2 = Uniform(0,1,'zeta2','$\\zeta_2$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    beta = Uniform(0,8,'beta','$\\beta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7,
                    
                    constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(15)})
    priors.update({'o1':0,'o'+str(15): 0})
    return priors

def Double_priors_reduced():
    priors=bilby.prior.PriorDict()
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 10., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t,1}$'),
                    zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),
                    
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(60., 100, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    mu_a2 = Uniform(0., 1., 'mu_a2', '$mu_{\\rm a,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
                    sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t,2}$'),
                    zeta2 = Uniform(0,1,'zeta2','$\\zeta_2$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    beta = Uniform(0,8,'beta','$\\beta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7,
                    
                    amin1 = 0,
                    amax1 = 1,
                    zmin1 = -1,
                    delta2=0,
                    amin2 = 0,
                    amax2 = 1.,
                    zmin2 = -1.
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(15)})
    priors.update({'o1':0,'o'+str(15): 0})
    return priors

#Single

def Single_constraint(params):
    params['constraint']=0
    params['constraint']+=np.sign(params['amax1']-params['mu_a1'])-1
    
    return params

#priors
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
                    amax1 = Uniform(0.1,1,'amax1', '$a_{\\rm max,1}$'),
                    sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t,1}$'),
                    zmin1 = Uniform(-1,0.9, 'zmin1', '$z_{\\rm min,1}$'),
                    zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),

                    beta = Uniform(0,8,'beta','$\\beta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7,
                    
                    constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    return priors

#Triple

def Triple_constraint(params):
    params['constraint']=0
    
    params['constraint']+=np.sign(1-params['r2']-params['r3'])-1
    params['constraint']+=np.sign(params['mu_a2']-params['mu_a1'])-1
    params['constraint']+=np.sign(params['mu_a3']-params['mu_a2'])-1
    params['constraint']+=np.sign(params['amax1']-params['mu_a1'])-1
    params['constraint']+=np.sign(params['mu_a2']-params['amin2'])-1
    params['constraint']+=np.sign(params['amax2']-params['mu_a2'])-1
    params['constraint']+=np.sign(params['amax2']-params['amin2']-0.1)-1
    
    params['constraint']+=np.sign(params['mu_a3']-params['amin3'])-1
    params['constraint']+=np.sign(params['amax3']-params['mu_a3'])-1
    params['constraint']+=np.sign(params['amax3']-params['amin3']-0.1)-1
    
    params['constraint']+=np.sign(params['mmax3']-params['mmin3']-10)-1
    params['constraint']+=np.sign(params['mmax2']-params['mmin2']-10)-1
    params['constraint']+=np.sign(params['mmax1']-params['mmin1']-10)-1
    
    return params
    
def Triple_priors():
    priors=bilby.prior.PriorDict(conversion_function=Triple_constraint)
    priors.update(dict(
       delta1=Uniform(1,10),
       mmin1 = Uniform(2., 60., 'mmin1', '$m_{\\rm min,1}$'),
       mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
       alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
       mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
       sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
       amin1 = 0,
       amax1 = Uniform(0.1,1,'amax1', '$a_{\\rm max,1}$'),
       sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t,1}$'),
       zmin1 = Uniform(-1,0.9, 'zmin1', '$z_{\\rm min,1}$'),
       zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),
       
       delta2=Uniform(1,10),
       mmin2 = Uniform(2., 60., 'mmin2', '$m_{\\rm min,2}$'),
       mmax2 = Uniform(20., 100, 'mmax2', '$m_{\\rm max,2}$'),
       alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
       amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
       mu_a2 = Uniform(0., 1., 'mu_a2', '$mu_{\\rm a,2}$'),
       sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
       amax2 = Uniform(0.1,1, 'amax2', '$a_{\\rm max,2}$'),
       sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t,2}$'),
       zmin2 = Uniform(-1,0.9, 'zmin2', '$z_{\\rm min,2}$'),
       zeta2 = Uniform(0,1,'zeta2','$\\zeta_2$'),
       r2 = Uniform(0,1, 'r2', '$r_2$'),
       
       delta3=Uniform(1,10),
       mmin3 = Uniform(2., 60., 'mmin3', '$m_{\\rm min,3}$'),
       mmax3 = Uniform(20., 100, 'mmax3', '$m_{\\rm max,3}$'),
       alpha3 = Uniform(-4., 8., 'alpha3', '$\\alpha_3$'),
       amin3 = Uniform(0,0.8, 'amin3', '$a_{\\rm min,3}$'),
       mu_a3 = Uniform(0., 1., 'mu_a3', '$mu_{\\rm a,3}$'),
       sigma_a3 = Uniform(0.05, 0.5, 'sigma_a3', '$\\sigma_{\\rm a,3}$'),
       amax3 = Uniform(0.1,1, 'amax3', '$a_{\\rm max,3}$'),
       sigma_t3 = Uniform(0.1, 4., 'sigma_t3', '$\\sigma_{\\rm t,3}$'),
       zeta3 = Uniform(0,1,'zeta3','$\\zeta_3$'),
       r3 = Uniform(0,1, 'r3', '$r_3$'),

       beta = Uniform(0,8,'beta','$\\beta$'),

       lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
       gamma=2.7,
       
       constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
    ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(15)})
    priors.update({'o1':0,'o'+str(15): 0})
    priors.update({'q'+str(i+1): TG(0,1,-100,100,name='q'+str(i+1))  for i in np.arange(15)})
    priors.update({'q1':0,'q'+str(15): 0})
    return priors
 

def Triple_reduced_constraint(params):
    params['constraint']=0
    
    params['constraint']+=np.sign(1-params['r2']-params['r3'])-1
    params['constraint']+=np.sign(params['mu_a2']-params['mu_a1'])-1
    params['constraint']+=np.sign(params['mu_a3']-params['mu_a2'])-1
    
    params['constraint']+=np.sign(params['mmax3']-params['mmin3']-10)-1
    params['constraint']+=np.sign(params['mmax2']-params['mmin2']-10)-1
    params['constraint']+=np.sign(params['mmax1']-params['mmin1']-10)-1
    
    return params

    
def Triple_priors_reduced():
    priors=bilby.prior.PriorDict(conversion_function=Triple_reduced_constraint)
    priors.update(dict(
       delta1=Uniform(1,10),
       mmin1 = Uniform(2., 60., 'mmin1', '$m_{\\rm min,1}$'),
       mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,1}$'),
       alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
       mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
       sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
       sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t,1}$'),
       zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),
       
       mmin2 = Uniform(2., 60., 'mmin2', '$m_{\\rm min,2}$'),
       mmax2 = Uniform(20., 100, 'mmax2', '$m_{\\rm max,2}$'),
       alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
       mu_a2 = Uniform(0., 1., 'mu_a2', '$mu_{\\rm a,2}$'),
       sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
       sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t,2}$'),
       zeta2 = Uniform(0,1,'zeta2','$\\zeta_2$'),
       r2 = Uniform(0,1, 'r2', '$r_2$'),
       
       mmin3 = Uniform(2., 60., 'mmin3', '$m_{\\rm min,3}$'),
       mmax3 = Uniform(20., 100, 'mmax3', '$m_{\\rm max,3}$'),
       alpha3 = Uniform(-4., 8., 'alpha3', '$\\alpha_3$'),
       mu_a3 = Uniform(0., 1., 'mu_a3', '$mu_{\\rm a,3}$'),
       sigma_a3 = Uniform(0.05, 0.5, 'sigma_a3', '$\\sigma_{\\rm a,3}$'),
       sigma_t3 = Uniform(0.1, 4., 'sigma_t3', '$\\sigma_{\\rm t,3}$'),
       zeta3 = Uniform(0,1,'zeta3','$\\zeta_3$'),
       r3 = Uniform(0,1, 'r3', '$r_3$'),

       beta = Uniform(0,8,'beta','$\\beta$'),
       
       amin1 = 0,
       amax1 = 1,
       zmin1 = -1,
       delta2=0,
       amin2 = 0,
       amax2 = 1.,
       zmin2 = -1.,
       delta3=0,
       amin3 = 0,
       amax3 = 1.,
       zmin3 = -1.,

       lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
       gamma=2.7,
       
       constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
    ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(15)})
    priors.update({'o1':0,'o'+str(15): 0})
    priors.update({'q'+str(i+1): TG(0,1,-100,100,name='q'+str(i+1))  for i in np.arange(15)})
    priors.update({'q1':0,'q'+str(15): 0})
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
def hyper_Triple(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Triple_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,amin3,mu_a3,sigma_a3,amax3,sigma_t3,zmin3,zeta3,r3,beta)*llh_z(z,gamma)
    return hp

#Double
def hyper_Double(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta)*llh_z(z,gamma)
    return hp

def hyper_Double_nosel(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,\
                                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,amin2,mu_a2,sigma_a2,amax2,sigma_t2,zmin2,zeta2,r2,beta)
    return hp

#Single
def hyper_Single(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Single_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta)*llh_z(z,gamma)
    return hp
    
def hyper_Single_nosel(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Single_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,amin1,mu_a1,sigma_a1,amax1,sigma_t1,zmin1,zeta1,beta)
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

inject_dir='/data/home/public/share/LIGO_post_release/data_for_pop_analysis/LVKC_injection/'
if use_ssr:
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

def log_SpinF():
    r1=s1x_inj**2+s1y_inj**2+s1z_inj**2
    r2=s2x_inj**2+s2y_inj**2+s2z_inj**2
    return -2*np.log(4*np.pi)-np.log(r1)-np.log(r2)
logpspin_inj=log_SpinF()
logpact_inj=np.log(act_Uniform(a1_inj,a2_inj,ct1_inj,ct2_inj))

#This selection effect dose not account for spin distribution 
def log_selection_eff(Nobs, mass_model,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdm = np.log(mass_model(m1_inj,m2_inj,**kwargs))
    log_dNds = logpspin_inj
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdm+log_dNds, np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds - logpdraw) - np.log(Ndraw)
    term1 = Nobs*log_N(Tobs,lgR0,gamma)
    term2 = -np.exp(log_Nexp)
    return term1 + term2
    
def Rate_selection_function_with_uncertainty(Nobs, mass_model,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdm = np.log(mass_model(m1_inj,m2_inj,**kwargs))
    log_dNds = logpspin_inj
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdm+log_dNds, np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds - logpdraw) - np.log(Ndraw)
    term1 = Nobs*log_N(Tobs,lgR0,gamma)
    term2 = -np.exp(log_Nexp)
    selection=term1 + term2
    logmu=log_Nexp-log_N(Tobs,lgR0,gamma)
    varsel= np.sum(np.exp(2*(np.log(Tobs)+log_dNdzdmds - logpdraw-log_N(Tobs,lgR0,gamma)- np.log(Ndraw))))-np.exp(2*logmu)/Ndraw
    total_vars=Nobs**2 * varsel / np.exp(2*logmu)
    Neff=np.exp(2*logmu)/varsel
    return selection, total_vars, Neff


        
        
