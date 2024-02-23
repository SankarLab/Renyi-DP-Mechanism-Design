import jax
import jax.numpy as jnp 
import jax.random as jrandom

from abc import ABC, abstractmethod
from typing import List, Tuple, Union


'''
Notes:

    Only coordinate descent logistic regression is implemented as this time, but all the archetecture needed
    for proximal coordnate descent lasso is here. 

'''
class Abstract_Loss(ABC):

    '''
    This is an awkward class. Its not really a class, because the 
    internal states, i.e. data = (X,y) and regularization,
    are constants. This class is more of a convienent wrapper for
    a bunch of Jax functions. 

    '''
    def objective(self, w: jnp.ndarray, res: jnp.ndarray) -> float:
        return self.eval_objective(w, res) + self.eval_regularizer(w)
        
    def coordinate_gradient(self, 
                            w: jnp.ndarray, 
                            j: int, 
                            res: jnp.ndarray,
                            clip: float) -> float:
        """
        Get total coordinate gradient. It gets the 
        per sample gradinet from loss and clips it. It
        then adds the regularizer gradient to this. Note: 
        I do not clip the regularizer gradient, since it is
        independent of the private data, hence a post processing. 
        """

        #output is a jnp array
        obj_gradient = self.vectorized_per_sample_coordinate_gradient(w, j, res)

        #output is a float
        average_clipped_obj_gradient = jnp.mean(jnp.clip( obj_gradient , a_min = -clip, a_max = clip))

        #output is a float. This is 0 for proximal coordinate GD, and non-zero for coordinate GD
        regularizer_gradient = self.regularizer_coord_gradient(w, j)

        return average_clipped_obj_gradient + regularizer_gradient

    @abstractmethod
    def vectorized_per_sample_coordinate_gradient(self, w: jnp.ndarray, j: int, res: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for an objective coordinate gradient.")
    
    @abstractmethod
    def regularizer_coord_gradient(self, w: jnp.ndarray, j: int) -> float:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for a regularizer coordinate gradient.")
    @abstractmethod
    def vector_residuals(self, w: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for residuals.")
    @abstractmethod 
    def eval_objective(self, w: jnp.ndarray, res: jnp.ndarray) -> float:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for objective.")
    @abstractmethod
    def eval_probs(self, w: jnp.ndarray, res: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for probs.")
    @abstractmethod
    def eval_predictions(self, w: jnp.ndarray, res: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for predictions.")
    @abstractmethod
    def accuracy(self, w: jnp.ndarray, res: jnp.ndarray) -> float:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for accuracy.")
    @abstractmethod 
    def eval_regularizer(self, w: jnp.ndarray) -> float:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for regularizer.")
    @abstractmethod 
    def coord_prox(self, updated_wj: float, stepsize: float) -> float:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for coord_prox.")
    @abstractmethod
    def update_residuals(self, res: jnp.ndarray, diff_w: float, j: int) -> jnp.ndarray:
        raise NotImplementedError(f"{type(self)} has not provided an implementation for updating residuals.")

class Logistic_Loss(Abstract_Loss):
    
    def __init__(self, 
                 data: Tuple,
                 regularization: float) -> None:
        

        self.X_ = data[0]
        self.y_ = data[1]
        self.n_ = data[0].shape[0]
        self.p_ = data[0].shape[1]

        # singular_values = jnp.linalg.svd(self.X_, compute_uv=False)
        # self.lipschitz = 0.25 / self.n_ * jnp.max(singular_values)**2

        #Note: add regularization to coordinate lipschitz
        X2_feat_bound_squared = jnp.mean( jnp.power(self.X_, 2), axis = 0)  #jnp.linalg.norm(self.X_, ord = 2, axis = 0)
        self.vec_coord_lipschitz = 0.25 * X2_feat_bound_squared + regularization      # 0.25 / self.n_ * self.X2_feat_bound_**2
        
        self.regularizer = regularization
        
    def eval_objective(self, w: jnp.ndarray, res: jnp.ndarray) -> float:
        return jnp.mean( jax.nn.softplus(- res) )
        # exp_arr = 1 + jnp.exp(- res)
        # arr = jnp.log(exp_arr)
        # return jnp.mean(arr) 
    
    def eval_probs(self, w: jnp.ndarray, res: jnp.ndarray) -> jnp.ndarray:
        log_probs = jax.nn.log_sigmoid( jnp.dot(self.X_, w) )
        return jnp.exp(log_probs)
        
    def eval_predictions(self, w: jnp.ndarray, res: jnp.ndarray) -> jnp.ndarray:
        probs = self.eval_probs(w, res)

        #since y data is {-1,1}, we do this funky 
        #transformation to map {True,False} -> {1,-1} 
        return jnp.array( (probs > 0.5) ) * 2 - 1

    def accuracy(self, w: jnp.ndarray, res: jnp.ndarray) -> float:
        preds = self.eval_predictions(w, res)
        bool_array = (preds == self.y_ )
        return jnp.mean( jnp.array(bool_array ) )
        
    def eval_regularizer(self, w: jnp.ndarray) -> float:
        return 0.5 * self.regularizer *  jnp.sum(jnp.power(w, 2))
        
    def vector_residuals(self, w: jnp.ndarray) -> jnp.ndarray:
        return self.y_ * jnp.dot(self.X_, w) 
        
    def update_residuals(self, res: jnp.ndarray, diff_w: float, j: int) -> jnp.ndarray:
        X_jcol = jnp.take(self.X_, j, axis = 1)
        return res + self.y_ * X_jcol * diff_w
        
    def regularizer_coord_gradient(self, w: jnp.ndarray, j: int) -> float:
        return self.regularizer * w[j] 
        
    def vectorized_per_sample_coordinate_gradient(self, w: jnp.ndarray, j: int, res: jnp.ndarray):
        # res = self.vector_residuals(w)
        X_jcol = jnp.take(self.X_, j, axis = 1)
        return - self.y_ / (1.0 + jnp.exp(res)) * X_jcol

    def coord_prox(self, updated_wj: float, stepsize: float) -> float:
        return updated_wj



def run_jit_gauss_final(Loss, w_init, clip, \
            sigma_array, learning_rate, epochs, seed):

    #get total iterations. p is the total number of features in training data
    p = Loss.p_
    T = epochs * p
    
    #initialize key for random sampling
    key = jrandom.PRNGKey(seed)
    
    #get j's. Here, it is just {0,1,2, ..., p-1, 0, 1, ... },  T times
    j_indices = jnp.mod(jnp.arange(T, dtype = int), p )
    
    #create learning rate array
    stepsize_array = learning_rate / Loss.vec_coord_lipschitz

    #create clipping aray
    lipalpha = Loss.vec_coord_lipschitz #jnp.power(Loss.vec_coord_lipschitz, alpha_clip)
    lipsum = jnp.sum(lipalpha) 
    clipping_array = clip * jnp.sqrt(lipalpha / lipsum)
    
    #create effective sigma array by doing sigma_new = (sensitivity) * sigma_input
    #because sensitivity of gradient is 2 * lipschitz constant / dataset size
    effective_sigma_array = sigma_array * 2 * clipping_array / Loss.n_

    #this function does one iteration
    def inner_loop(j_temp, tup):
        theta, objs, accuracies, res, key = tup
        
        j = j_indices[j_temp]  #j_indices[j_temp] 
        clip_j = clipping_array[j]
        sigma_j = effective_sigma_array[j]
        theta_j_old = theta[j]

        #if we are at end of epoch, update objective and accuracy list
        objs = jax.lax.cond((j_temp)%p == 0,\
                            lambda: objs.at[jnp.array(j_temp/p, int)].set(Loss.objective(theta, res)),\
                            lambda: objs)
        accuracies = jax.lax.cond((j_temp)%p == 0,\
                            lambda: accuracies.at[jnp.array(j_temp/p, int)].set(Loss.accuracy(theta, res)),\
                            lambda: accuracies)
        
        g_j = Loss.coordinate_gradient(theta, j, res, clip_j)
        
        #add Gaussian noise to g_j
        g_j += jrandom.normal(key = key, shape = (1,))[0] * sigma_j

        #update key
        _, key = jrandom.split(key) 

        #compute jth component of theta
        theta_j_new = Loss.coord_prox(theta[j] - stepsize_array[j] * g_j, stepsize_array[j])
        
        #update jth component of theta
        theta = theta.at[j].set(theta_j_new)

        #update residuals
        diff_theta = theta_j_new - theta_j_old
        res = Loss.update_residuals(res, diff_theta, j) 

        return theta, objs, accuracies, res, key
        
    #initialize for loop
    
    res = Loss.vector_residuals(w_init)
    
    objs = jnp.ones(shape = (epochs+1,)) * -1
    accuracies = jnp.ones(shape = (epochs+1,)) * -1
    
    init = (w_init, objs, accuracies, res, key )
    
    #this fori loop jit compiles. 
    #no need to jit this function. 
    theta, objs, accuracies, res, _ = jax.lax.fori_loop(
      lower=0, upper=T, body_fun=inner_loop, init_val=init)

    objs = objs.at[-1].set(Loss.objective(theta, res))
    accuracies = accuracies.at[-1].set(Loss.accuracy(theta, res))
    return theta, objs, accuracies


def run_jit_rdp_final(Loss, w_init, clip, \
            rdp_noise_params_array, learning_rate, epochs, seed, k):

    #get total iterations. p is the total number of features in training data
    p = Loss.p_
    T = epochs * p
    
    #initialize key for random sampling
    key = jrandom.PRNGKey(seed)
    
    #get j's. Here, it is just {0,1,2, ..., p-1, 0, 1, ... },  T times
    j_indices = jnp.mod(jnp.arange(T, dtype = int), p )
    
    #create learning rate array
    stepsize_array = learning_rate / Loss.vec_coord_lipschitz

    #create clipping aray
    lipalpha = Loss.vec_coord_lipschitz #jnp.power(Loss.vec_coord_lipschitz, alpha_clip)
    lipsum = jnp.sum(lipalpha) 
    clipping_array = clip * jnp.sqrt(lipalpha / lipsum)
    assert jnp.all(clipping_array == clipping_array)

    #set up grid for sampling bins from the RDP noise
    full_pmf = jnp.hstack((rdp_noise_params_array[::-1][:-1], rdp_noise_params_array))
    x_full_pmf = jnp.arange(-len(rdp_noise_params_array) + 1, len(rdp_noise_params_array)) / k

    #this function does one iteration
    def inner_loop(j_temp, tup):
        theta, objs, accuracies, res, key = tup
        
        j = j_indices[j_temp]  #j_indices[j_temp] 
        clip_j = clipping_array[j]
        theta_j_old = theta[j]

        #if we are at end of epoch, update objective and accuracy list
        objs = jax.lax.cond((j_temp)%p == 0,\
                            lambda: objs.at[jnp.array(j_temp/p, int)].set(Loss.objective(theta, res)),\
                            lambda: objs)
        accuracies = jax.lax.cond((j_temp)%p == 0,\
                            lambda: accuracies.at[jnp.array(j_temp/p, int)].set(Loss.accuracy(theta, res)),\
                            lambda: accuracies)
        
        g_j = Loss.coordinate_gradient(theta, j, res, clip_j)
        
        #sample bin
        bin_j = jrandom.choice(key = key, a = x_full_pmf, p = full_pmf, shape = (1,))[0]
    
        #update key before sampling the uniform
        _, key = jrandom.split(key) 
    
        #sample uniform 
        jitter_j = jrandom.uniform(key = key, shape = (1,), minval = -1 / 2 / k, maxval = 1 / 2 / k)[0]
        
        #update key
        _, key = jrandom.split(key) 
        
        #create effective noise array by doing etas_new = (sensitivity) * etas
        #where etas is samples from the RDP mechanism
        #with sensitivity of gradient is 2 * lipschitz constant / dataset size
        eta_j = bin_j + jitter_j
        eta_j = eta_j * 2 * clipping_array[0] / Loss.n_

        #add RDP noise to g_j
        g_j += eta_j
        
        theta_j_new = Loss.coord_prox(theta[j] - stepsize_array[j] * g_j, stepsize_array[j])
        
        #update jth component of theta
        theta = theta.at[j].set(theta_j_new)

        #update residuals
        diff_theta = theta_j_new - theta_j_old
        res = Loss.update_residuals(res, diff_theta, j) 

        return theta, objs, accuracies, res, key
        
    #initialize for loop
    
    res = Loss.vector_residuals(w_init)
    
    objs = jnp.ones(shape = (epochs+1,)) * -1
    accuracies = jnp.ones(shape = (epochs+1,)) * -1
    
    init = (w_init, objs, accuracies, res, key )
    
    #this fori loop jit compiles. 
    #no need to jit this function. 
    theta, objs, accuracies, res, key = jax.lax.fori_loop(
      lower=0, upper=T, body_fun=inner_loop, init_val=init)

    objs = objs.at[-1].set(Loss.objective(theta, res))
    accuracies = accuracies.at[-1].set(Loss.accuracy(theta, res))
    
    return theta, objs, accuracies


#-------------------------------------------------------
#---------------- Historical Functions------------------
#-------------------------------------------------------
'''

NOTES: 

In Jax, running:

-----------------------------------------------
import jax.random as jrandom

key = jrandom.PRNGKey(42)
etas1 = jrandom.normal(key = key, shape = (2,))

key = jrandom.PRNGKey(42)
etas2 = jrandom.normal(key = key, shape = (3,))

print(etas1)
print(etas2)
-----------------------------------------------

gives: 


-----------------------------------------------
[-2.169826    0.46480057]
[ 0.18693547 -1.2806505  -1.5593132 ]
-----------------------------------------------

i.e., the generated random numbers depend on the size
of the requested number of random numbers! This does 
NOT occur in numpy, and is surprising behavior. 

The two functions functions below, run_jit_rdp_fast and 
run_jit_gauss_fast, for T iterations, sample T Gaussians / uniforms
in one call, i.e. shape = (T,). This causes the functions to 
to yeild irreproducible results, in the sense that running
the optimization for 10 vs 12 epochs may yeild completely different
training trajectories, even for fixed random seed.

I changed the names of these functions to "run_jit_XX_fast" because
they run ~3x faster than the default functions, at the cost of no 
reproducibility. 
'''
def run_jit_rdp_fast(Loss, w_init, clip, \
            rdp_noise_params_array, learning_rate, epochs, seed, k):

    '''
    See NOTES above
    '''

    #get total iterations. p is the total number of features in training data
    p = Loss.p_
    T = epochs * p
    
    #initialize key for random sampling
    key = jrandom.PRNGKey(seed)
    
    #get j's. Here, it is just {0,1,2, ..., p-1, 0, 1, ... },  T times
    j_indices = jnp.mod(jnp.arange(T, dtype = int), p )
    
    #create learning rate array
    stepsize_array = learning_rate / Loss.vec_coord_lipschitz

    #create clipping aray
    lipalpha = Loss.vec_coord_lipschitz #jnp.power(Loss.vec_coord_lipschitz, alpha_clip)
    lipsum = jnp.sum(lipalpha) 
    clipping_array = clip * jnp.sqrt(lipalpha / lipsum)
    assert jnp.all(clipping_array == clipping_array)

    #sample RDP noise, uses rdp_noise_params_array and k input values
    full_pmf = jnp.hstack((rdp_noise_params_array[::-1][:-1], rdp_noise_params_array))
    x_full_pmf = jnp.arange(-len(rdp_noise_params_array) + 1, len(rdp_noise_params_array)) / k
    bins = jrandom.choice(key = key, a = x_full_pmf, p = full_pmf, shape = (T,))
    
    #update key before sampling the uniforms
    _, key = jrandom.split(key) 
    
    #sample uniforms 
    jitter = jrandom.uniform(key = key, shape = (T,), minval = -1 / 2 / k, maxval = 1 / 2 / k)
    
    #update key (for best practice, this key is not used later in the code)
    _, key = jrandom.split(key) 
    
    #create effective noise array by doing etas_new = (sensitivity) * etas
    #where etas is samples from the RDP mechanism
    #with sensitivity of gradient is 2 * lipschitz constant / dataset size
    etas = bins + jitter
    etas = etas * 2 * clipping_array[0] / Loss.n_
    
    #this function does one iteration
    def inner_loop(j_temp, tup):
        theta, objs, accuracies, res = tup
        
        j = j_indices[j_temp]  #j_indices[j_temp] 
        clip_j = clipping_array[j]
        eta_j = etas[j]
        theta_j_old = theta[j]

        #if we are at end of epoch, update objective list and accuracy list
        objs = jax.lax.cond((j_temp)%p == 0,\
                            lambda: objs.at[jnp.array(j_temp/p, int)].set(Loss.objective(theta, res)),\
                            lambda: objs)
        accuracies = jax.lax.cond((j_temp)%p == 0,\
                            lambda: accuracies.at[jnp.array(j_temp/p, int)].set(Loss.accuracy(theta, res)),\
                            lambda: accuracies)
        g_j = Loss.coordinate_gradient(theta, j, res, clip_j)
        
        #add RDP noise to g_j
        g_j += eta_j
        
        theta_j_new = Loss.coord_prox(theta[j] - stepsize_array[j] * g_j, stepsize_array[j])
        
        #update jth component of theta
        theta = theta.at[j].set(theta_j_new)

        #update residuals
        diff_theta = theta_j_new - theta_j_old
        res = Loss.update_residuals(res, diff_theta, j) 

        return theta, objs, accuracies, res
        
    
    
    # ---- for loop -----
    
    #initialize for loop
    
    res = Loss.vector_residuals(w_init)
    
    objs = jnp.ones(shape = (epochs+1,)) * -1
    accuracies = jnp.ones(shape = (epochs+1,)) * -1
    
    init = (w_init, objs, accuracies, res)
    
    #this fori loop jit compiles. 
    #no need to jit this function. 
    theta, objs, accuracies, res = jax.lax.fori_loop(
      lower=0, upper=T, body_fun=inner_loop, init_val=init)

    objs = objs.at[-1].set(Loss.objective(theta, res))
    accuracies = accuracies.at[-1].set(Loss.accuracy(theta, res))
    return theta, objs, accuracies


def run_jit_gauss_fast(Loss, w_init, clip, \
            sigma_array, learning_rate, epochs, seed):

    '''
    see NOTES above. 
    '''
    #get total iterations. p is the total number of features in training data
    p = Loss.p_
    T = epochs * p
    
    #initialize key for random sampling
    key = jrandom.PRNGKey(seed)
    
    #get j's. Here, it is just {0,1,2, ..., p-1, 0, 1, ... },  T times
    j_indices = jnp.mod(jnp.arange(T, dtype = int), p )
    
    #sample Gaussian noise
    #noise is scaled to appropriate value in for loop
    etas = jrandom.normal(key = key, shape = (T,))
    
    #update key (for best practice, this key is not used later in the code)
    _, key = jrandom.split(key) 
    
    #create learning rate array
    stepsize_array = learning_rate / Loss.vec_coord_lipschitz

    #create clipping aray
    lipalpha = Loss.vec_coord_lipschitz #jnp.power(Loss.vec_coord_lipschitz, alpha_clip)
    lipsum = jnp.sum(lipalpha) 
    clipping_array = clip * jnp.sqrt(lipalpha / lipsum)
    
    #create effective sigma array by doing sigma_new = (sensitivity) * sigma_input
    #because sensitivity of gradient is 2 * lipschitz constant / dataset size
    effective_sigma_array = sigma_array * 2 * clipping_array / Loss.n_

    #see equation 
    #this function does one iteration
    def inner_loop(j_temp, tup):
        theta, objs, accuracies, res = tup
        
        j = j_indices[j_temp]  #j_indices[j_temp] 
        clip_j = clipping_array[j]
        eta_j = etas[j]
        sigma_j = effective_sigma_array[j]
        theta_j_old = theta[j]

        #if we are at end of epoch, update objective and accuracy list
        objs = jax.lax.cond((j_temp)%p == 0,\
                            lambda: objs.at[jnp.array(j_temp/p, int)].set(Loss.objective(theta, res)),\
                            lambda: objs)
        accuracies = jax.lax.cond((j_temp)%p == 0,\
                            lambda: accuracies.at[jnp.array(j_temp/p, int)].set(Loss.accuracy(theta, res)),\
                            lambda: accuracies)

        g_j = Loss.coordinate_gradient(theta, j, res, clip_j)
        
        #add Gaussian noise to g_j
        g_j += eta_j * sigma_j
        
        theta_j_new = Loss.coord_prox(theta[j] - stepsize_array[j] * g_j, stepsize_array[j])
        
        #update jth component of theta
        theta = theta.at[j].set(theta_j_new)

        #update residuals
        diff_theta = theta_j_new - theta_j_old
        res = Loss.update_residuals(res, diff_theta, j) 

        return theta, objs, accuracies, res
        
    
    
    # ---- for loop -----
    
    #initialize for loop
    
    res = Loss.vector_residuals(w_init)
    
    objs = jnp.ones(shape = (epochs+1,)) * -1
    accuracies = jnp.ones(shape = (epochs+1,)) * -1
    
    init = (w_init, objs, accuracies, res )
    
    #this fori loop jit compiles. 
    #no need to jit this function. 
    theta, objs, accuracies, res = jax.lax.fori_loop(
      lower=0, upper=T, body_fun=inner_loop, init_val=init)

    objs = objs.at[-1].set(Loss.objective(theta, res))
    accuracies = accuracies.at[-1].set(Loss.accuracy(theta, res))
    
    return theta, objs, accuracies
