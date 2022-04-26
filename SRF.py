import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numba import njit,vectorize,prange
import h5py




with h5py.File("SRF.hdf5","r") as h5:
    CS = np.array(h5.get("Cross-Sections/NaI"))
    
    KN_CS = np.array(h5.get("Cross-Sections/Klein-Nishina/KN_CS"))
    Es_KN_CS = np.geomspace(0.001,100,1000)
    
    PPR = np.array(h5.get("COMPTEL/Photopeak Ratio/D2_PPR"))
    Es_PPR = np.geomspace(0.01,100,1000)
    


### Data Extraction Functions

@vectorize
def incoh_cs(E):
    return 1.*continue_log(E,CS[:,0],CS[:,2])

@vectorize
def photoel_cs(E):
    return 1.*continue_log(E,CS[:,0],CS[:,3])
    
@vectorize
def pair_cs(E):
    return 1.*continue_log(E,CS[:,0],CS[:,4]) + continue_log(E,CS[:,0],CS[:,5])

@vectorize
def total_cs(E):
    return 1.*incoh_cs(E) + photoel_cs(E) + pair_cs(E)


@njit
def klein_nishina_diff_energy_cs(E1,E2):
    theta=np.arccos( 1-0.511*( (E1-E2) / (E1*E2) ) )
    if np.isnan(theta):
        return 0.
    P=E2/E1
    return P**2 * (P+1/P-np.sin(theta)**2) * (1/E2**2)


@njit
def compton_cs_E1s(E1, E2, incoh, out):
    for i in range(len(E1)):
        out[i] = klein_nishina_diff_energy_cs(E1[i], E2) / continue_log(E1[i], Es_KN_CS, KN_CS) * incoh[i]
    return out
        
@njit
def compton_cs_E2s(E1, E2, incoh, out):
    for i in range(len(E2)):
        out[i] = klein_nishina_diff_energy_cs(E1, E2[i]) / continue_log(E1, Es_KN_CS, KN_CS) * incoh
    return out

@vectorize
def detector_sig(E): #Standard Deviation of Detector
    return 1/100 * (9.86*E + 4.143*E**2)**(1/2)


### Helper Functions

@njit
def continue_log(value,x_data,y_data):
    if value<x_data[0] or value>=x_data[-1]:
        return 0
    for i in range(len(x_data)-1):
        if value>=x_data[i] and value<x_data[i+1]:
            if y_data[i]==0. or y_data[i+1]==0.:
                return 0.
            s=np.log(y_data[i+1]/y_data[i])/np.log(x_data[i+1]/x_data[i])
            f=y_data[i]/(x_data[i]**s)
            return f*value**s

@njit
def continue_log_x(value,x_data,y_data):
    if value<x_data[0] or value>=x_data[-1]:
        return 0
    for i in range(len(x_data)-1):
        if value>=x_data[i] and value<x_data[i+1]:
            if y_data[i+1]-y_data[i]==0.:
                return y_data[i]
            else:
                A= (y_data[i+1]-y_data[i]) / np.log(x_data[i+1]/x_data[i])
                B = ( np.exp(y_data[i]/A) ) / x_data[i] 
                return A*np.log(B*value)

@njit
def continue_lin(value,x_data,y_data):
    if value<x_data[0] or value>=x_data[-1]:
        return 0
    for i in range(len(x_data)-1):
        if value>=x_data[i] and value<x_data[i+1]:
            return ( y_data[i+1] - y_data[i] ) / ( x_data[i+1] - x_data[i] ) * ( value - x_data[i] ) + y_data[i]

@njit
def geomspace(start, stop, num):
    return np.array([ start * (stop/start) ** ( i/ (num-1) ) for i in range(num) ])


@njit
def lin_int_solver(domain_values,range_values):
    h= (domain_values[-1] - domain_values[0]) / (len(domain_values)-1)
    weights=np.ones(len(domain_values)) * h
    weights[0]/=2
    weights[-1]/=2
    return np.sum(weights*range_values)

@njit
def discrete_delta(domain,value):
    weights=np.zeros(len(domain))
    if value<domain[0] or value>=domain[-1]:
        return weights
    for i in range(len(domain)-1):
        if value>=domain[i] and value<domain[i+1]:
            weights[i] = 1 - (value - domain[i]) / (domain[i+1] - domain[i])
            weights[i+1] = (value - domain[i]) / (domain[i+1] - domain[i])
            return weights / ( (domain[-1] - domain[0]) / (len(domain) - 1) )

@njit
def gauss_sig(E1,E2):
    return (2*np.pi)**(-1/2) * detector_sig(E2)**(-1) * np.exp( -(E1-E2)**2 / (2*detector_sig(E2)**2) )

def fit_func(a,y1,y2):
    return np.sum((y1-a*y2)**2)


### Main Algorithm

@njit
def main_loop_lin(E):
    E_res = 1500
    n = 10
    max_loop = 10
    E_min = 0.001
    E_d_up = 0.52
    T_threshold = 0.001
    T_trial = 13.
    T_abort = 15.
    T_guess_init = 5.
    T_guess_var = 30.
    
    flag = False
    
    # print("INITIALIZING")
    # print(E)
    # print()

    Es = np.linspace(E_min, E, E_res)
    P_T = continue_log_x(E, Es_PPR, PPR)
    incoh = incoh_cs(Es)
    photoel = photoel_cs(Es)
    pair = pair_cs(Es)
    total = total_cs(Es)
    compton_E1s = np.zeros(E_res)
    compton_E2s = compton_cs_E2s( E , Es , incoh[-1] , np.zeros(E_res) )
    
    calc_delta = pair_cs(E) != 0.
    
    if calc_delta:
        E_r = round( (E // E_d_up) + 1. )
        for i in range(E_res):
            if Es[i] >= E_d_up:
                break
        E_res_d = i*E_r+1
        Es_d = np.linspace(E_min, Es[i], E_res_d)
        
        incoh_d = incoh_cs(Es_d)
        photoel_d = photoel_cs(Es_d)
        total_d = total_cs(Es_d)
        compton_E1s_d = np.zeros(E_res_d)
        
    else:
        E_r = 0
        E_res_d=10
        Es_d = np.linspace(E_min, E_d_up, E_res_d)
        
    S = np.zeros((2,n,E_res))
    Q = np.zeros((2,n))
    C = np.zeros((2,n))
    S_d = np.zeros((2,n,E_res_d))
    Q_d = np.zeros((2,n))
    
    Q[0,0] = photoel[-1] / total[-1]
    S[0,0,:] = compton_E2s / total[-1]
    C[0,0] = pair[-1] / total[-1]
    S_d[0,0,:] = discrete_delta(Es_d, 0.511)

    
    T=T_trial
                
    for iteration in range(max_loop):
        
        for loop in range(1,n):
            exp=np.exp(-total*T)
            #exp_d=np.exp(-total_d*T)
            
            Q[0,loop]=lin_int_solver(Es,
                                     S[0,loop-1,:] * (1-exp) * photoel / total )
            Q[1,loop]=(lin_int_solver(Es,
                                      S[0,loop-1,:] * (exp) * photoel )
                       +lin_int_solver(Es,
                                       S[1,loop-1,:] * (1-exp) * photoel / total))
            
            C[0,loop] = lin_int_solver(Es, 
                                       S[0,loop-1,:] * (1-exp) * pair / total)
            C[1,loop] = ( lin_int_solver(Es, 
                                         S[0,loop-1,:] * (exp) * pair )
                         + lin_int_solver(Es,
                                          S[1,loop-1,:] * (1-exp) * pair / total) )
            
            for loop2 in range(E_res):
                compton_E1s=compton_cs_E1s( Es, Es[loop2], incoh, compton_E1s)
                
                S[0,loop,loop2]=lin_int_solver(Es,
                                           S[0,loop-1,:] * (1-exp) * compton_E1s / total)
                S[1,loop,loop2]=(lin_int_solver(Es,
                                            S[0,loop-1,:] * exp * compton_E1s ) 
                              +lin_int_solver(Es,
                                              S[1,loop-1,:] * (1-exp) * compton_E1s / total))
                
        if calc_delta:
            for loop in range(1,n):
                exp_d=np.exp(-total_d*T)
                            
                Q_d[0,loop]=lin_int_solver(Es_d,
                                         S_d[0,loop-1,:] * (1-exp_d) * photoel_d / total_d )
                Q_d[1,loop]=(lin_int_solver(Es_d,
                                          S_d[0,loop-1,:] * (exp_d) * photoel_d )
                           +lin_int_solver(Es_d,
                                           S_d[1,loop-1,:] * (1-exp_d) * photoel_d / total_d))
                
                for loop2 in range(E_res_d):
                    compton_E1s_d=compton_cs_E1s( Es_d, Es_d[loop2], incoh_d, compton_E1s_d)
                                    
                    S_d[0,loop,loop2]=lin_int_solver(Es_d,
                                               S_d[0,loop-1,:] * (1-exp_d) * compton_E1s_d / total_d)
                    S_d[1,loop,loop2]=(lin_int_solver(Es_d,
                                                S_d[0,loop-1,:] * exp_d * compton_E1s_d ) 
                                  +lin_int_solver(Es_d,
                                                  S_d[1,loop-1,:] * (1-exp_d) * compton_E1s_d / total_d))
        
        
        
        if not np.sum( Q[1,:] )==0:
            dT = ( P_T - np.sum(Q[0,:]) - np.sum(C[0,:]) * (np.sum(Q_d[0,:]))**2 ) / ( np.sum(Q[1,:]) + np.sum(C[1,:]) * (np.sum(Q_d[0,:]))**2 + 2 * np.sum(C[0,:]) * np.sum(Q_d[0,:]) * np.sum(Q_d[1,:]) )
            # print (iteration,T,dT)
        else:
            print (iteration,T)
            print("Uh-oh!")
            T=T_guess_init+T_guess_var*np.random.random(1)[0]
            continue

        if abs(dT)<T_threshold:
            print("SUCCESS!",E)
            flag=True
            break
        elif dT<-T:
            T/=2
        else:
            T+=dT
    if not flag:
        print("ABORT!",E)
        T=T_abort
        dT=0
    # print()
    return Q, S, dT, T, P_T, C, Q_d, S_d, Es, Es_d, E_r


@njit
def calc_count_spectrum(E):
    Q, S, dT, T, P_T, C, Q_d, S_d, Es, Es_d, E_r = main_loop_lin(E)
    
    n = len(Q[0,:])
    
    E_res = len(Es)
    h = (Es[-1]-Es[0]) / (E_res-1)
    E_res_inc = 300 ######################################################################### int( 4*D2_E_sig(E) / h )
    E_res2 = E_res+E_res_inc
    Es2 = np.linspace(Es[0], Es[-1]+E_res_inc*h, E_res2)
    
    Phi = S[0,n-1,:] + dT*S[1,n-1,:]
    
    for i in range(n-1):
        Phi += (S[0,i,:] + dT*S[1,i,:]) * np.exp( -(T+dT) * total_cs(Es) )
    
    if not E_r==0:
        E_res_d = len(Es_d)
        h_d = (Es_d[-1]-Es_d[0]) / (E_res_d-1)
        E_res_d_b = round(Es_d[0] / h_d) + 3*E_r
        E_res_d_t = round(Es_d[-1] / h_d) + 3*E_r
        E_res3 = E_res_d + E_res_d_b + E_res_d_t
        Es3 = np.linspace( Es_d[0] - E_res_d_b*h_d, Es_d[-1] + E_res_d_t*h_d, E_res3 )
        
        Q_d_s = np.sum( Q_d[0,:] ) + dT*np.sum( Q_d[1,:] )
        
        Phi_d = S_d[0,n-1,:] + dT*S_d[1,n-1,:]
        
        for i in range(n-1):
            Phi_d += (S_d[0,i,:] + dT*S_d[1,i,:]) * np.exp( -(T+dT) * total_cs(Es_d) )
        
        
        Phi_d_2 = np.zeros(E_res3)
        Phi_d_2[E_res_d_b:E_res_d_b+E_res_d] = Phi_d
            
        Phi_d_2 += discrete_delta(Es3,0.)[:] * Q_d_s
        
        Phi_d_3 = np.zeros(E_res3)
        Phi_d_r = np.zeros(E_res3)
        for i in range(E_res3):
            Phi_d_r[:] = np.zeros(E_res3)
            Phi_d_r[i::-1] = Phi_d_2[:i+1]
            Phi_d_3[i] = lin_int_solver(Es3,Phi_d_r*Phi_d_2)
        
        weights = np.ones(E_res3) * h_d
        weights[0] /= 2
        weights[-1] /= 2
        s_d = 0
        
        for i in range(E_res3):
            s_d += weights[i] * Phi_d_3[i]
            Phi_d_3[i]=0.
            if s_d >= Q_d_s**2:
                break
        
        i_0 = E_res_d_b % E_r
        i_n = (E_res3 - i_0 - 1) // E_r + 1
        i_1 = i_0 + (i_n-1) * E_r
        
        Es4 = np.linspace( Es3[i_0], Es3[i_1], i_n )
        Phi_d_4 = np.zeros( i_n )
        
        if E_r % 2 == 1:
            i_d = E_r//2
            Phi_d_4[0] = np.sum( Phi_d_3[i_0 : i_0 + i_d + 1] ) / (i_d + 1)
            Phi_d_4[-1] = np.sum( Phi_d_3[i_1 - i_d : i_1 + 1] ) / (i_d + 1)
            for i in range(1,i_n-1):
                Phi_d_4[i] = np.sum( Phi_d_3[i_0 + i*E_r - i_d : i_0 + i*E_r + i_d + 1] ) / (2*i_d + 1)
        else:
            i_d = E_r//2
            weights = np.ones(E_r + 1) / E_r
            weights[0] /= 2
            weights[-1] /= 2
            Phi_d_4[0] = np.sum( Phi_d_3[i_0 : i_0 + i_d + 1] * weights[round(E_r/2) : ] )
            Phi_d_4[-1] = np.sum( Phi_d_3[i_1 - i_d : i_1 + 1] * weights[ : round(E_r/2) + 1] )
            for i in range(1,i_n-1):
                Phi_d_4[i] = np.sum( Phi_d_3[i_0 + i*E_r - i_d : i_0 + i*E_r + i_d + 1] * weights[:] )
                
        i_m = E_res_d_b // E_r
        pps = np.sum( C[0,:] ) + dT*np.sum( C[1,:] )
        Phi[:i_n-i_m] += pps*Phi_d_4[i_m:]
    
    e=np.zeros(E_res2)
    for i in range(E_res):
        e[i] = continue_lin( E-Es[i], Es, Phi )
    
    Cc=np.zeros(E_res2)
    B=np.zeros(E_res2)
    for i in range(E_res2):
        for j in range(E_res2):
            B[j] = gauss_sig(Es2[i],Es2[j])
        Cc[i]=lin_int_solver( Es2, B*e )
    Cp=np.zeros(E_res2)
    for i in range(E_res2):
        Cp[i] = gauss_sig(Es2[i], E) * P_T
    Ct=Cc+Cp
    Ct/=lin_int_solver(Es2,Ct)
    return Es2, Ct


@njit(parallel=True)
def calc_cont_spectrum(x_data,y_data):
    E_min=0.5
    samples=300
    E_len_total=1800 ###################################################################################################################################################################
    weights=np.zeros(samples)
    E_samples=np.linspace(E_min,x_data[-1],samples)
    results=np.zeros((2,samples,E_len_total))
    out=np.zeros(len(x_data))
    for i in prange(samples):
        results[0,i,:], results[1,i,:] = calc_count_spectrum(E_samples[i])##############################################################
        weights[i] = continue_lin(E_samples[i], x_data, y_data)
    for i in prange(len(x_data)):
        for j in range(samples):
            out[i] += continue_lin( x_data[i], results[0,j,:], results[1,j,:] ) * weights[j]
    return out



def monochromatic_spectrum():
    fig=plt.figure(figsize=(10,12))
    grid=plt.GridSpec(3, 1,hspace=0.12,wspace=0.12)
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Energy [MeV]",labelpad=25)
    pa.set_ylabel("Counts [Counts/Bin]",labelpad=40)
    
    E_samples=(1.5,5.0,15.0)
    
    
    for i,E in enumerate(E_samples):
        print(E)
        ax=fig.add_subplot(grid[i,0])
        
        ax.set_ylabel("{E} MeV".format(E=E),labelpad=7)
        ax.yaxis.set_label_position("right")
        
        with h5py.File("SRF.hdf5","r") as h5:
            E2s = np.array(h5.get("COMPTEL/Simulated Data/Mono_{E}_Incidence".format(E=int(E*1000))))
            E2s_r = np.array(h5.get("COMPTEL/Simulated Data/Mono_{E}_Counts".format(E=int(E*1000))))
        
        
        Es = np.linspace(0.5, E*1.05, 100)
        xs = Es[:-1] + (Es[-1]-Es[0])/(len(Es)-1)
        
        n, _ = np.histogram(E2s, bins=Es)
        n_r, _, _ = plt.hist(E2s_r, bins=Es, label="MEGAlib Simulation D2 Counts",color="C0")
        
        out = calc_cont_spectrum(xs, n) # mon_spec[i,:] # 
        a=minimize(fit_func,np.amax(n_r)/np.amax(out),(n_r,out)).x #######################################################################
        
        plt.plot(xs,n*np.sum(out*a)/np.sum(n),lw=2.0,label="D2 Incidence Spectrum",c="C2")
        plt.plot(xs,out*a,lw=2.0,label="D2 Count Spectrum",c="C1")
        if i==0:
            plt.legend()

    plt.savefig('monochromatic_final_spectrum_pp.pdf',bbox_inches='tight')







