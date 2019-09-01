function [alpha betal betar] = estimateaggdparam(vec)


gam   = 0.2:0.001:10;
r_gam = ((gamma(2./gam)).^2)./(gamma(1./gam).*gamma(3./gam));


leftstd            = sqrt(mean((vec(vec<0)).^2));
rightstd           = sqrt(mean((vec(vec>0)).^2));

gammahat           = leftstd/rightstd;
rhat               = (mean(abs(vec)))^2/mean((vec).^2);
rhatnorm           = (rhat*(gammahat^3 +1)*(gammahat+1))/((gammahat^2 +1)^2);
[min_difference, array_position] = min((r_gam - rhatnorm).^2);
alpha              = gam(array_position);

betal              = leftstd *sqrt(gamma(1/alpha)/gamma(3/alpha));
betar              = rightstd*sqrt(gamma(1/alpha)/gamma(3/alpha));


