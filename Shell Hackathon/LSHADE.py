import numpy as np
from scipy import stats
from Farm_Evaluator_Vec import *
import pandas as pd


power_curve  =  loadPowerCurve('Shell_Hackathon Dataset/power_curve.csv')
n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)
wind_inst_freq =  binWindResourceData('Shell_Hackathon Dataset/Wind Data/combined_data.csv')


def get_objective_value(coords,constraint_check=False,year=None,verbose=0,n_wind_instances=n_wind_instances, cos_dir=cos_dir, sin_dir=sin_dir, wind_sped_stacked=wind_sped_stacked, C_t=C_t,wind=1,wind_inst_freq=wind_inst_freq,get_each_turb_aep=1):
    turb_specs = {'Name': 'Anon Name','Vendor': 'Anon Vendor','Type': 'Anon Type','Dia (m)': 100,
                  'Rotor Area (m2)': 7853,'Hub Height (m)': 100,'Cut-in Wind Speed (m/s)': 3.5,'Cut-out Wind Speed (m/s)': 25,
                  'Rated Wind Speed (m/s)': 15,'Rated Power (MW)': 3}
    turb_diam  =  turb_specs['Dia (m)']
    turb_rad   =  turb_diam/2

    if wind==0:
        if year == None:
            mean_AEP = 0
            for y in [2007,2008,2009,2013,2014,2015,2017]:
                wind_inst_freq =  binWindResourceData('Shell_Hackathon Dataset/Wind Data/wind_data_'+str(y)+'.csv')
                mean_AEP += getAEP(turb_rad, coords, power_curve, wind_inst_freq,
                          n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
            mean_AEP /= 7

            if verbose:
                print('Average power produced by the wind farm over 7 years is: ', "%.12f"%(mean_AEP), 'GWh')

            return mean_AEP
        else:
            wind_inst_freq =  binWindResourceData('Shell_Hackathon Dataset/Wind Data/wind_data_'+str(year)+'.csv')
            AEP = getAEP(turb_rad, coords, power_curve, wind_inst_freq,
                          n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)

            if verbose:
                print('Total power produced by the wind farm is: ', "%.12f"%(AEP), 'GWh')

            return AEP

    else:
        #penalty = checkConstraints(coords,turb_diam)

        AEP = getAEP(turb_rad, coords, power_curve, wind_inst_freq,
              n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,get_each_turb_aep)

        return AEP

class LSHADE:

    def __init__(self,Np_ini,d,coords,turb,NFE_max,Np_min,is_fixed_coords=0,fixed_coords=None):
        self.Np_ini = Np_ini
        self.NFE_max = NFE_max
        self.Np_min = Np_min
        self.Np = Np_ini #population size
        self.d = d #total number of possible wind turbine locations in the discrete scenario(100)
        self.turb = turb #Number of turbines to be placed(50)
        self.Obj_vals = [] #Stores the objective values obtained in each generation
        self.F = [] #Stores scaling factors
        self.CR = [] #Stores crossover rates
        self.best_score = 0 #Maximum AEP found until any given instance(generation)
        self.X_all = [] #Stores all the decision vectors corresponding to all generations
        self.coords = coords #Coordinates of 100 discrete locations
        self.mu_f = 0.5 #Scaling factor mean used to generate scaling factor f
        self.mu_cr = 0.5 #Crossover Rate mean used to generate CRs
        self.mutate_vec = [] #Stores all the mutation vectors (v's)
        self.trial_vec = [] #Stores all the trial vectors(u's)
        self.NFE = 0
        self.is_fixed_coords = is_fixed_coords
        if is_fixed_coords:
            self.fixed_coords = fixed_coords



    def initialize(self):
        """
        Np = population size
        d = Dimension of decision vector; Number of discrete location
        turb = Number of turbines to be placed; 50 out of 100 dimensional vector will have 1's
        The function randomly initializes values between 0 and 1. This is the 0th generation population.
        """
        random_values = np.random.random((self.Np,self.d))
        columns = np.argsort(random_values,axis=-1)[:,-self.turb:].flatten()
        rows = np.repeat(list(range(self.Np)),repeats=self.turb)
        X = np.zeros((self.Np,self.d))
        X[rows,columns] = 1
        self.X_all.append(X)
        scores = []
        for x in X:
            scores.append(self.get_objective(x))
        self.Obj_vals.append(np.array(scores))

    def get_coords(self,x):
        """
        x is an array of length d with values between 0 and 1.
        The location of turbines is decided by sorting x and replacing the last 50 values with 1 and remaining as 0.
        returns 50 coordinates out of 100 possible ones based on x.
        """
        locs = np.argsort(x)[-self.turb:]
        co = self.coords[locs.tolist()]
        if self.is_fixed_coords:
            co = np.vstack((co,self.fixed_coords))
        return co

    def get_objective(self,x):
        """
        x is an array of length d with values between 0 and 1.
        returns objective value corresponding to x
        """
        co = self.get_coords(x)
        return get_objective_value(co,get_each_turb_aep=0)

    def update_best_score(self,t):
        """
        Evaluates all the decision vectors corresponding to a given generation and stores in Obj_vals.
        t is used to identify t-th generation.
        Also finds the best objective function value till the current generation.
        """
        score = max(self.Obj_vals[t])
        max_ind = np.argmax(self.Obj_vals[t])
        if score>self.best_score:
            self.best_score = score
            self.best_x = self.X_all[t][max_ind]
            self.best_coords = self.get_coords(self.X_all[t][max_ind])

    def get_p_best(self,t,p=0.05):
        """
        p is the percentile of the best population individuals.
        x_pbest is chosen randomly from Np*p best individuals.
        returns x_pbest
        """
        P = int(self.Np*p)
        inds = np.argsort(self.Obj_vals[t])[-P:]
        p_best = np.random.choice(inds,1)[0]
        return self.X_all[t][p_best]

    def get_f(self):
        """
        Scaling factors for a generation are drawn from cauchy distribution with mean 0.5 and variance 0.1
        """
        F = stats.cauchy.rvs(loc=self.mu_f,scale=np.sqrt(0.1),size=self.Np)
        return F

    def get_cr(self):
        """
        Crossover rates are drawn from normal distribution with mean 0.5 and variance 0.1
        """
        CR = np.random.normal(loc=self.mu_cr,scale=np.sqrt(0.1),size=self.Np)
        return CR

    def parameter_adaptation(self,t,p=0.05):
        if t>=1:
            P = int(self.Np*p)
            inds = np.argsort(self.Obj_vals[t])[-P:].tolist()
            self.mu_f = (self.mu_f + np.mean(self.F[t-1][inds]))/2
            self.mu_cr = (self.mu_cr + np.mean(self.CR[t-1][inds]))/2

    def mutation(self,t):
        """
        generates mutant vector v using scaling factor, x_pbest and current generation samples (X)
        """
        x_pbest = np.array(self.get_p_best(t))
        X = self.X_all[t]
        F = self.get_f()
        r1 = np.random.choice(list(range(self.Np)),size=self.Np).tolist()
        r2 = np.random.choice(list(range(self.Np)),size=self.Np).tolist()
        X_r1, X_r2 = X[r1], X[r2]
        V = np.zeros((self.Np,self.d))
        for i in range(self.Np):
            V[i] = X[i] + F[i] * (x_pbest - X[i]) + F[i] * (X_r1[i]-X_r2[i])
        V[V>1] = (X[V>1]+1)/2 #If v goes above 1 or below 0 bring it back within the range.
        V[V<0] = X[V<0]/2
        self.mutate_vec.append(V)
        self.F.append(F.flatten())

    def crossover(self,t):
        """
        generates trial vectors u using cross over rates and mutant vectors.
        """
        U = np.zeros((self.Np,self.d))
        V = self.mutate_vec[t]
        X = self.X_all[t]
        K = np.random.choice(list(range(self.Np)),size=1)[0]
        CR = self.get_cr()
        self.CR.append(CR)
        random_values = np.random.random((self.Np,self.d))
        for i in range(self.Np):
            for j in range(self.d):
                if j==K or random_values[i,j] <= CR[i]:
                    U[i,j] = V[i,j]
                else:
                    U[i,j] = X[i,j]
        self.trial_vec.append(U)

    def drop_rows(self,X,final_length,t):
        X_1 = X[np.argsort(self.Obj_vals[t+1])[-final_length:].tolist()]
        self.Obj_vals[t+1] = self.Obj_vals[t+1][np.argsort(self.Obj_vals[t+1])[-final_length:].tolist()]
        return X_1

    def selection(self,t):
        """
        The next generation (t+1) vectors are selected based on objective function values for x and u.
        """
        X_1 = np.zeros((self.Np,self.d))
        X = self.X_all[t]
        U = self.trial_vec[t]
        scores = np.zeros(self.Np)
        for i in range(self.Np):
            f_x = self.Obj_vals[t][i]
            f_u = self.get_objective(U[i])
            if f_x>=f_u:
                X_1[i] = X[i]
                scores[i] = f_x
            else:
                X_1[i] = U[i]
                scores[i] = f_u
        self.Obj_vals.append(scores)
        columns = np.argsort(X_1,axis=-1)[:,-self.turb:].flatten()
        rows = np.repeat(list(range(self.Np)),repeats=self.turb)
        X_1[rows,columns] = 1
        self.NFE += self.Np
        self.population_size_reduction()
        X_1 = self.drop_rows(X_1,self.Np,t)
        self.X_all.append(X_1)


    def population_size_reduction(self):
        self.Np = (self.Np_min-self.Np_ini)/self.NFE_max * self.NFE + self.Np_ini
        self.Np = int(self.Np)

    def run(self,run_from_middle=0,starting_x=None):
        if run_from_middle:
            self.X_all.append(starting_x)
        else:
            self.initialize()
        t = 0
        while self.NFE <= self.NFE_max:
            self.update_best_score(t)
            print('{:.4f}'.format(self.best_score))
            self.mutation(t)
            self.crossover(t)
            self.selection(t)
            t += 1
            #self.parameter_adaptation(t)
