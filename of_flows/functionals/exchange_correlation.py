import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar
from jax import lax

class LDA(eqx.Module):
    r"""
    Local density approximation (LDA) exchange functional.

    See eq 2.72 from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich

    $$\epsilon_{\text{X}}^{\text{LDA}} = -\frac{3}{4} \left(\frac{3}{\pi}\right)^{1/3} \rho(\boldsymbol{x})^{1/3}$$
    
    Attributes
    ----------
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions. 
    """
    
    dim: Int[Scalar, ""]

    def __init__(self, dim = 3):
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        
        l = -(3/4) * (Ne**(4/3)) * (3/jnp.pi)**(1/3)
        return l * den**(1/3)


class ExchangeCorrelation1D(eqx.Module):
    r"""
    1D exchange-correlation functional.
    
    See eq 7 in https://iopscience.iop.org/article/10.1088/1751-8113/42/21/214021 
    
    $$\epsilon_{\text{XC}} (r_s,\zeta) = \frac{a_\zeta + b_\zeta r_s + c_\zeta r_s^{2}}{1 + d_\zeta r_s + e_\zeta r_s^2 + f_\zeta r_s^3} + 
    \frac{g_\zeta r_s \ln[{r_s + \alpha_\zeta r_s^{\beta_\zeta} }]}{1 + h_\zeta r_s^2}$$
    
    Attributes
    ----------
    dim : Int[Scalar, ""]
        Dimension of the system, default is 1 dimension. 
    """
    
    dim: Int[Scalar, ""]

    def __init__(self, dim = 1):
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch d"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        rs = 1 / (2 * Ne * den)
        a0 = -0.8862269
        b0 = -2.1414101
        c0 = 0.4721355
        d0 = 2.81423
        e0 = 0.529891
        f0 = 0.458513
        g0 = -0.202642
        h0 = 0.470876
        alpha0 = 0.104435
        beta0 = 4.11613
        
        n1 = a0 + b0 * rs + c0 * rs**2 
        d1 = 1 + d0 * rs + e0 * rs**2 + f0 * rs**3 
        f1 = n1 / d1 
        n2 = g0 * rs * jnp.log(rs + alpha0 * rs**beta0)
        d2 = 1 + h0 * rs**2 
        f2 = n2 / d2 
        return Ne * (f1 + f2)


class CorrelationVWN(eqx.Module):
    r"""
    VWN correlation functional.
    
    See original paper eq 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also text after eq 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    $$\epsilon_{\text{C}}^{\text{VWN}} = \frac{A}{2} \left\{ \ln\left(\frac{y^2}{Y(y)}\right) + \frac{2b}{Q} \tan^{-1} \left(\frac{Q}{2y + b}\right) +
    - \frac{b y_0}{Y(y_0)} \left[\ln\left(\frac{(y-y_0)^2}{Y(y)}\right) + \frac{2(b+2y_0)}{Q}\tan^{-1}  \left(\frac{Q}{2y+b}\right) \right] \right\}$$
    
    Attributes
    ----------
    clip_cte : Float[Array, ""], 
        Float for numerical stability. 
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions. 
    """
    
    clip_cte: Float[Array, ""]
    dim: Int[Scalar, ""]

    def __init__(self, clip_cte = 1e-30, dim = 3):
        self.clip_cte = clip_cte
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        A = 0.0621814
        b = 3.72744
        c = 12.9352
        x0 = -0.10498
        
        den_clipped = jnp.where(den > self.clip_cte, den, 0.0)
        log_den = jnp.log2(jnp.clip(den_clipped, a_min=self.clip_cte))
        log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
        log_x = log_rs / 2
        rs = 2.**log_rs
        x = 2.**log_x

        X = 2.**(2. * log_x) + 2.**(log_x + jnp.log2(b)) + c
        X0 = x0**2 + b * x0 + c
        Q = jnp.sqrt(4 * c - b**2)

        e_PF = (
            A / 2. * (
                2. * jnp.log(x)
                - jnp.log(X)
                + 2. * b / Q * jnp.arctan(Q / (2. * x + b))
                - b * x0 / X0 * (
                    jnp.log((x - x0)**2. / X) 
                    + 2. * (2. * x0 + b) / Q * jnp.arctan(Q / (2. * x + b))
                )
            )
        )
        return Ne * e_PF


class CorrelationPW92(eqx.Module):
    r"""
    PW92 correlation functional.
    
    See eq 10 in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.13244

    $$\epsilon_{\text{C}}^{\text{PW92}} = -2A(1+\alpha_1 r_s) \ln \left[{1 + \frac{1}{2A(\beta_1 r_s^{1/2} + \beta_2 r_s + \beta_3 r_s^{3/2} + \beta_4 r_s^{2})}}\right]$$
    
    Attributes
    ----------
    clip_cte : Float[Array, ""], 
        Float for numerical stability. 
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions. 
    """
    
    clip_cte: Float[Array, ""]
    dim: Int[Scalar, ""]

    def __init__(self, clip_cte = 1e-30, dim = 3):
        self.clip_cte = clip_cte
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        A_ = 0.031091
        alpha1 = 0.21370
        beta1 = 7.5957
        beta2 = 3.5876
        beta3 = 1.6382
        beta4 = 0.49294

        log_den = jnp.log2(jnp.clip(den, a_min=self.clip_cte))
        log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_den / 3.0
        
        brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
        ars = 2 ** (log_rs + jnp.log2(alpha1))
        brs = 2 ** (log_rs + jnp.log2(beta2))
        brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
        brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))

        e_PF = -2 * A_ * (1 + ars) * jnp.log(
            1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2)
        )
        return Ne * e_PF


class B88Exchange(eqx.Module):
    r"""
    B88 exchange functional.

    See eq 8 in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.38.3098
    See also https://github.com/ElectronicStructureLibrary/libxc/blob/4bd0e1e36347c6d0a4e378a2c8d891ae43f8c951/maple/gga_exc/gga_x_b88.mpl#L22

    $$\epsilon_{\text{X}}^{\text{B88}} = -\beta \frac{X^2}{\left(1 + 6 \beta X \sinh^{-1}(X) \right)} \rho(\boldsymbol{x})^{1/3}$$
    
    Attributes
    ----------
    clip_cte : Float[Array, ""], 
        Float for numerical stability. 
    beta : Float[Array, ""],
        Beta parameter. 
    dim : Int[Scalar, ""]
        Dimension of the system, default is 3 dimensions. 
    """
    
    clip_cte: Float[Array, ""]
    beta: Float[Array, ""]
    dim: Int[Scalar, ""]

    def __init__(self, clip_cte = 1e-30, beta = 0.0042, dim = 3):
        self.clip_cte = clip_cte
        self.beta = beta
        self.dim = dim 
    
    def __call__(
        self,
        den: Float[Array, "batch"],
        score: Float[Array, "batch dim"],
        Ne: Int[Scalar, ""],
    ) -> Float[Array, "batch"]:
        den_clipped = jnp.clip(den, a_min=self.clip_cte)
        log_den = jnp.log2(den_clipped)

        score_sqr = jnp.einsum('ij,ij->i', score, score)
        den_sqr = den_clipped * den_clipped
        grad_den_norm_sq = lax.expand_dims(score_sqr, (1,)) * den_sqr

        log_grad_den_norm = jnp.log2(jnp.clip(grad_den_norm_sq, a_min=self.clip_cte)) / 2
        log_x_sigma = log_grad_den_norm - 4 / 3.0 * log_den
        x_sigma = 2**log_x_sigma

        # Eq 2.78 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
        b88_e = -(
            self.beta
            * 2 ** (
                4 * log_den / 3
                + 2 * log_x_sigma
                - jnp.log2(1 + 6 * self.beta * x_sigma * jnp.arcsinh(x_sigma))
            )
        )
        return b88_e * Ne**(2/3)