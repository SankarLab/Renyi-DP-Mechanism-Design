from typing import List, Union
import math
import jax.numpy as jnp

# connect the dots fun stuff 
from dp_accounting.pld.privacy_loss_mechanism import AdditiveNoisePrivacyLoss, AdjacencyType, TailPrivacyLossDistribution, ConnectDotsBounds
from dp_accounting.pld.privacy_loss_distribution import _create_pld_pmf_from_additive_noise, _pld_for_subsampled_mechanism, PrivacyLossDistribution
from dp_accounting.pld import pld_pmf

class DiscreteRDPPrivacyLoss(AdditiveNoisePrivacyLoss):

    def __init__(self,
               p_vec,
               N,
               k,
               r,
               mass_truncation: float = 1e-15,
               sensitivity: int = 1, 
               sampling_prob: float = 1.0,
               adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
        """Initializes the privacy loss of the discrete RDP mechanism.

        Args:
            p_vec, N, k, r: obvious 
            sensitivity: the integer sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)

            sampling_prob: sub-sampling probability, a value in (0,1].

            adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution (does not matter if sampling_prob = 1
        """
        if not isinstance(sensitivity, int):
            raise ValueError(f'Sensitivity is not an integer : {sensitivity}')

        self._mass_truncation = mass_truncation

        self._p_vec = p_vec
        self._final_p = p_vec[-1]
        self._r = r
        self._N = N
        self._k = k
        self.sensitivity = sensitivity
        
        # IMPORTANT: the True below tells Connect the Dots that this is a 
        # continuous mechanism
        super().__init__(sensitivity, False, sampling_prob, adjacency_type)

    def pdf(self, x: int):
        """
        intput: bin number  
        output: pdf value 
        """
        abs_x = abs(x)
        if abs_x > self._N:
            return self._final_p * self._r**(abs_x - self._N)
        return self._p_vec[abs_x]
    
    def get_delta_for_epsilon(
        self, 
        epsilon: Union[float, List[float]]) -> Union[float, List[float]]:
        
        """
        Returns delta(epsilon). Vectorized over epsilon.
        """
        k = self._k
        p_val = self._p_vec
        r = self._r
        N = self._N 
        # integer_shift = self.sensitivity
        final_p = self._final_p
        
        if jnp.isscalar(epsilon):
            # epsilon is a scalar
            epsilon = jnp.asarray([epsilon])[:, jnp.newaxis]
        elif epsilon.size == 1:
            # epsilon is an array with one number in it
            epsilon = jnp.asarray([epsilon])[:, jnp.newaxis]
        elif len(epsilon.shape) == 1:
            # epsilon is a (N) array. Needs to be (N ,1)
            epsilon = jnp.asarray(epsilon)[:, jnp.newaxis]
        
        shifts = jnp.arange(1, self.sensitivity + 1) # (1, 2, ..., sensitivity)
        deltas = jnp.zeros(epsilon.shape)[:,0] #take just first axis 
        for shift in shifts:
            proposed_deltas = self.get_delta_for_epsilon_shift(epsilon, shift)
            deltas = jnp.maximum(deltas, proposed_deltas)
        return deltas
    
    def get_delta_for_epsilon_shift(
        self, 
        epsilon: Union[float, List[float]],
        integer_shift: int) -> Union[float, List[float]]:
        
        """
        Returns delta(epsilon). Vectorized over epsilon.
        """
        k = self._k
        p_val = self._p_vec
        r = self._r
        N = self._N 
        # integer_shift = self.sensitivity
        final_p = self._final_p

        # fix p_val 
        p_val = p_val / k
        final_p = p_val[-1]
        log_r = jnp.log(r)
        exp_eps = jnp.exp(epsilon)

        ## Term 1: both pmfs are in the left geometric tail region
        condition = (-integer_shift * log_r) > epsilon
        t1 = jnp.where(
            condition,
            final_p * (r - exp_eps * r ** (integer_shift + 1)) / (1 - r),
            0.0
        )[:,0]

        ## Term 2: left pmf is not in the left geometric tail, right one is 
        t2 = jnp.sum( jnp.maximum( p_val[N - integer_shift + 1: N + 1] - exp_eps * final_p * jnp.arange(1, integer_shift + 1), 0 ) , axis = 1)

        ## Term 3: both pmfs are not in the geometric tail
        t3_1 = jnp.sum( jnp.maximum( p_val[1:N-integer_shift+1] - exp_eps * p_val[integer_shift+1:N+1], 0 ), axis = 1)
        t3_2 = jnp.sum( jnp.maximum( p_val[0:integer_shift][::-1] - exp_eps * p_val[1:integer_shift+1], 0 ), axis = 1)
        t3_3 = jnp.sum( jnp.maximum( p_val[integer_shift:N] - exp_eps * p_val[0:N-integer_shift], 0 ), axis = 1 )
        t3 = t3_1 + t3_2 + t3_3
    
        ## Term 4: left pmf in right geometric tail, right one is not 
        t4 = jnp.sum( jnp.maximum( final_p * jnp.arange(integer_shift) - exp_eps * p_val[N - integer_shift: N ], 0), axis = 1 )

        ## Term 5: both pmfs in right geometric tail region 
        condition = (integer_shift * log_r) > epsilon
        t5 = jnp.where(
            condition,
            final_p * (r ** (integer_shift)   - exp_eps ) / (1-r),
            0.0
        )[:,0]
        return t1 + t2 + t3 + t4 + t5
        
    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        """
        For REMOVE adjacency type: lower_x_truncation is set such that
            CDF(lower_x_truncation) = 0.5 * mass_truncation_bound, and
            upper_x_truncation is set to be -lower_x_truncation. Finally,
            lower_x_truncation is shifted by -1 * sensitivity.
            Recall that here mu_upper(x) := (1-q).mu(x) + q.mu(x + sensitivity),
            where q=sampling_prob. The truncations chosen above ensure that the tails
            of both mu(x) and mu(x+sensitivity) are smaller than 0.5 *
            exp(log_mass_truncation_bound). This ensures that the considered tails of
            mu_upper are no larger than exp(log_mass_truncation_bound). This is
            computationally cheaper than computing exact tail thresholds for mu_upper.
            
        For ADD adjacency type: lower_x_truncation is set such that
            CDF(lower_x_truncation) = 0.5 * mass_truncation_bound, and
            upper_x_truncation is set to be -lower_x_truncation. Finally,
            upper_x_truncation is shifted by +1 * sensitivity.
            Recall that here mu_upper(x) := mu(x) for any value of sampling_prob.
            The truncations chosen ensures that the tails of mu(x) (and hence of
            mu_upper) are no larger than 0.5 * exp(log_mass_truncation_bound).
            While it was not strictly necessary to shift upper_x_truncation by +1 *
            sensitivity in this case, this choice leads to the same discretized
            privacy loss distribution for both ADD and REMOVE adjacency
            types, in the case where sampling_prob = 1.
        """
        upper_x_truncation = math.ceil( self._N + math.log( self._mass_truncation * (1 - self._r) / self._final_p) )
        lower_x_truncation = -upper_x_truncation

        if self.adjacency_type == AdjacencyType.ADD:
            upper_x_truncation += self.sensitivity
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            lower_x_truncation -= self.sensitivity
        print(f'xlow, xhigh = {lower_x_truncation, upper_x_truncation}')
        return TailPrivacyLossDistribution(
        lower_x_truncation, upper_x_truncation,
        {math.inf: lambda x: 1})

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

        lower_x and upper_x are same as lower_x_truncation and upper_x_truncation
        as given by privacy_loss_tail().

        Returns:
          A ConnectDotsBounds instance containing lower and upper values of x
          to use in connect-the-dots algorithm.
        """
        tail_pld = self.privacy_loss_tail()
        xlow = tail_pld.lower_x_truncation
        xhigh = tail_pld.upper_x_truncation
        losses = [self.privacy_loss(x) for x in range(xlow, xhigh + 1)]
        print(f'epslow, epshigh = {(min(losses), max(losses))}')
        # return ConnectDotsBounds(lower_x=int(tail_pld.lower_x_truncation),
        #                      upper_x=int(tail_pld.upper_x_truncation))
        # return ConnectDotsBounds(
        #     epsilon_upper=self.privacy_loss(tail_pld.lower_x_truncation),
        #     epsilon_lower=self.privacy_loss(tail_pld.upper_x_truncation))
        return ConnectDotsBounds(
            epsilon_upper=max(losses),
            epsilon_lower=min(losses))
        


    def privacy_loss_without_subsampling(self, x: float) -> float:
        """Computes the privacy loss of the discrete RDP mechanism without sub-sampling at a given point.

        Args:
            x: the point at which the privacy loss is computed.

        Returns:
            The privacy loss of the discrete RDP mechanism at integer bin value x,
            which is given as

        For REMOVE adjacency type:
            returns log( f(x + shift) / f(x) )

        For ADD adjacency type:
           Negative of REMOVE

        Raises:
          ValueError: if the privacy loss is undefined.
        """
        if not isinstance(x, int):
            raise ValueError(f'Privacy loss at x is undefined for x = {x}')
        
        L = math.log( self.pdf(x + self.sensitivity) / self.pdf(x) )
        if self.adjacency_type == AdjacencyType.ADD:
            return -L

        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return L
        
    def from_privacy_guarantee():
        return None
    def inverse_privacy_loss_without_subsampling():
        return None
    def noise_cdf():
        return None
    def noise_log_cdf():
        return None
    
    
    
def from_discrete_RDP_mechanism(
    p_vec,
    N: int,
    k: int,
    r: float,
    sensitivity: int = 1,
    pessimistic_estimate: bool = True,
    value_discretization_interval: float = 1e-4,
    sampling_prob: float = 1.0,
    use_connect_dots: bool = True) -> PrivacyLossDistribution:
    """Computes the privacy loss distribution of the Discrete RDP mechanism.

    This method supports two algorithms for constructing the privacy loss
    distribution. One given by the "Privacy Buckets" algorithm and other given by
    "Connect the Dots" algorithm. See Sections 2.1 and 2.2 of supplementary
    material for more details.

    Args:
    p_vec, N, k, r: obvious 
    sensitivity: the sensitivity of function f. (i.e. the maximum absolute
      change in f when an input to a single user changes.)
    pessimistic_estimate: a value indicating whether the rounding is done in
      such a way that the resulting epsilon-hockey stick divergence computation
      gives an upper estimate to the real value.
    value_discretization_interval: the length of the dicretization interval for
      the privacy loss distribution. The values will be rounded up/down to be
      integer multiples of this number. Smaller value results in more accurate
      estimates of the privacy loss, at the cost of increased run-time / memory
      usage.
    sampling_prob: sub-sampling probability, a value in (0,1].
    use_connect_dots: when False (not default), the privacy buckets algorithm will
      be used to construct the privacy loss distribution. When True, the
      connect-the-dots algorithm would be used.

    Returns:
    The privacy loss distribution corresponding to the Discrete Laplace
    mechanism with given parameters.
    """

    def single_discrete_RDP_pld(
        adjacency_type: AdjacencyType) -> pld_pmf.PLDPmf:
        
        return _create_pld_pmf_from_additive_noise(
        DiscreteRDPPrivacyLoss(
            p_vec = p_vec,
            N = N,
            k = k,
            r = r,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type),
        pessimistic_estimate=pessimistic_estimate,
        value_discretization_interval=value_discretization_interval,
        use_connect_dots=use_connect_dots)

    return _pld_for_subsampled_mechanism(single_discrete_RDP_pld,
                                       sampling_prob)
