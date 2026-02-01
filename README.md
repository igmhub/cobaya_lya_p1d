# Lya P1D likelihood for Cobaya

External likelihood for [Cobaya](https://github.com/CobayaSampler/cobaya) containing Lya P1D cosmological constraints from multiple analyses, including the results from the DESI DR1 analysis (Chaves-Montero et al. 2026, https://arxiv.org/abs/2601.21432).

## How to use

First clone this repository:

```console
$ git clone https://github.com/igmhub/cobaya_lya_p1d.git
```

Then install it:

```console
$ pip install -e .
```

If you want to evaluate the likelihood for the best-fitting DESI DR1 results, you call the likelihood within Cobaya as follows:

```yaml
cobaya_lya_p1d.cobaya_lya_p1d.Cobaya_lya_p1d:
  params:
    delta2star_mean: 0.379
    delta2star_std: 0.032
    nstar_mean: -2.309
    nstar_std: 0.019
    correlation: -0.1738
    zstar: 3.0
    kstar_kms: 0.009
    speed: 30000
```

To check that everything works as expected, you can run:

```console
$ cobaya-run cobaya_lya_p1d/example.yaml
```

See Equations 1 and 2 from Chaves-Montero et al. 2026 for the definitions of delta2star and nstar, and the [Cobaya documentation](https://cobaya.readthedocs.io/en/latest/) for instructions on installing and using Cobaya.

### If you use this likelihood please cite Chaves-Montero et al. 2026 (https://arxiv.org/abs/2601.21432)
