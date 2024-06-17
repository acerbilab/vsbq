functions {
    real ep_normal_lpdf(real y,  real C, real a, real b) {
        return C * normal_lpdf(y | a, b);
    }
    
    real log_likelihood(int noise_level, int n, real[] s_vest, real[] s_vis, int[] obs, real[] theta) {
        real sigma_vis = exp(theta[noise_level]);
        real sigma_vest = exp(theta[4]);
        real kappa = exp(theta[6]);
        real lambd = inv_logit(theta[5]);
        
        real b = sigma_vest/sigma_vis;
        real cumm = 0;
        for (i in 1:n) { 
            real a_plus = (s_vest[i]  - s_vis[i] + kappa)/sigma_vis;
            real a_minus = (s_vest[i]  - s_vis[i] - kappa)/sigma_vis;
            
            real p_resp = lambd/2 + (1-lambd) * (Phi(a_plus / sqrt(1+b^2)) - Phi(a_minus / sqrt(1+b^2)));
            
            cumm += bernoulli_lpmf(obs[i] | 1-p_resp);
        }
        return cumm;
    }
}
data {
    int n1;
    int n2;
    int n3;
    
    real s_vest1[n1];
    real s_vest2[n2];
    real s_vest3[n3];
    
    real s_vis1[n1];
    real s_vis2[n2];
    real s_vis3[n3];
    
    int obs1[n1];
    int obs2[n2];
    int obs3[n3];
    
    real C;
}
parameters {
    real theta[6];
}
model {
    theta[1] ~ ep_normal(C, log(5), log(4)); // log_sigma_vis[1]
    theta[2] ~ ep_normal(C, log(5), log(4)); // log_sigma_vis[2]
    theta[3] ~ ep_normal(C, log(5), log(4)); // log_sigma_vis[3]
    theta[4] ~ ep_normal(C, log(5), log(4)); // log_sigma_vest
    theta[6] ~ ep_normal(C, log(5), log(4)); // log_kappa
    theta[5] ~ ep_normal(C, logit(0.02), log(2)); // logit_lambd
            
    target += log_likelihood(1, n1, s_vest1, s_vis1, obs1, theta);
    target += log_likelihood(2, n2, s_vest2, s_vis2, obs2, theta);
    target += log_likelihood(3, n3, s_vest3, s_vis3, obs3, theta);
}
generated quantities {
    real lpd = 0;
    lpd += ep_normal_lpdf(theta[1] | C, log(5), log(4)); // log_sigma_vis[1]
    lpd += ep_normal_lpdf(theta[2] | C, log(5), log(4)); // log_sigma_vis[2]
    lpd += ep_normal_lpdf(theta[3] | C, log(5), log(4)); // log_sigma_vis[3]
    lpd += ep_normal_lpdf(theta[4] | C, log(5), log(4)); // log_sigma_vest
    lpd += ep_normal_lpdf(theta[6] | C, log(5), log(4)); // log_kappa
    lpd += ep_normal_lpdf(theta[5] | C, logit(0.02), log(2)); // logit_lambd
    lpd += log_likelihood(1, n1, s_vest1, s_vis1, obs1, theta);
    lpd += log_likelihood(2, n2, s_vest2, s_vis2, obs2, theta);
    lpd += log_likelihood(3, n3, s_vest3, s_vis3, obs3, theta);
}