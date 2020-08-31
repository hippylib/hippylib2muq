import pymuqSamplingAlgorithms as ms
import seaborn
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def _get_name(x):
    if isinstance(x, ms.CrankNicolsonProposal):
        return 'pcn'
    elif isinstance(x, ms.MALAProposal):
        return 'mala'
    elif isinstance(x, ms.MHKernel):
        return 'mh'
    elif isinstance(x, ms.DRKernel):
        return 'dr'


def print_methodDict(method_list):
    """Print the method list in dictionary format for MCMC simulation

    :param method_list: discription of MCMC methods

    """
    print('{0:18s} {1:10s} {2:10s} {3:10s}'.format('Method', 'Kernel', 'Proposal',
                                                   'Beta or Step-size'))
    print('-'*58)
    for mName, method in method_list.items():
        
        kern = method['Sampler'].Kernels()[0]
        kernName = _get_name(kern)
        if kernName == 'dr':
            props = kern.Proposals()
            print('{0:18s} {1:10s}'.format(mName, kernName), end=' ')
            for i, prop in enumerate(props):
                propName = _get_name(prop)
                if i != 0:
                    print(' '*30, end='')
                if propName == 'pcn':
                    print('{0:10s} {1:10.1e}'.format(propName,
                          method['Options']['Beta']))
                elif propName == 'mala':
                    print('{0:10s} {1:10.1e}'.format(propName,
                                                     method['Options']['StepSize']))
        else:
            prop = kern.Proposal()
            propName = _get_name(prop)
            print('{0:18s} {1:10s}'.format(mName, kernName), end=' ')
            if propName == 'pcn':
                print('{0:10s} {1:10.1e}'.format(propName,
                                                 method['Options']['Beta']))
            elif propName == 'mala':
                print('{0:10s} {1:10.1e}'.format(propName,
                                                 method['Options']['StepSize']))


def print_qoiResult(method_list, qoi_dataset):
    sep = '\n' + '=' * 51 + '\n'
    print(sep, "Summary of convergence diagnostics (single chain)", sep)
    print('{0:18s} {1:>6s} {2:^6s} {3:>6s} {4:>7s}'.format('Method', 'E[QOI]',
                                                           'AR', 'ESS', 'ES/min'))
    print("-" * 47)
    for mName, qoi_result in qoi_dataset.items():
        eqoi = qoi_result['qoi'].mean()

        ess = qoi_result['ess']
        esm = ess / method_list[mName]['ElapsedTime'] * 60.0

        kern = method_list[mName]['Sampler'].Kernels()[0]
        kernName = _get_name(kern)
        if kernName == 'dr':
            ar = method_list[mName]['AcceptRate'].sum()
        else:
            ar = method_list[mName]['AcceptRate']

        print('{0:18s} {1:>6.3f} {2:>6.3f} {3:>6.1f} {4:>6.1f}'.format(mName, eqoi,
                                                                       ar, ess, esm))


def plot_qoiResult(method_list, qoi_dataset, max_lag=None):
    fig, axes = plt.subplots(nrows=len(method_list), ncols=3, figsize=(12, 20))
    colorlist = ['c', 'orange', 'b', 'g', 'r']

    firstkey = list(method_list.keys())[0]
    nsamps = method_list[firstkey]['Options']['NumSamples'] - \
             method_list[firstkey]['Options']['BurnIn'] + 1
    if max_lag == None:
        max_lag = nsamps // 10

    for i, mName in enumerate(qoi_dataset):
        samps = qoi_dataset[mName]['qoi']

        axes[i, 0].plot(samps, 'o', alpha=0.1, c=colorlist[i])
        axes[i, 0].set_xlim((0, nsamps))
        axes[i, 0].set_ylim((-3, 3))
        axes[i, 0].set_ylabel(mName, rotation=90, size='large')

        acf, conf = sm.tsa.stattools.acf(samps, nlags=max_lag, alpha=.1, fft=False)
        xlags = np.arange(acf.size)
        conf0 = conf.T[0]
        conf1 = conf.T[1]

        axes[i, 1].fill_between(xlags, conf0, conf1, alpha=0.1, color=colorlist[i])
        axes[i, 1].plot((0, max_lag), (0, 0), 'k--')
        axes[i, 1].plot(xlags, acf, linewidth=2, color=colorlist[i])
        axes[i, 1].set_xlim((0, max_lag))
        axes[i, 1].set_ylim((-0.2, 1.0))

        seaborn.distplot(samps, ax=axes[i, 2], norm_hist=True, color=colorlist[i])
        axes[i, 2].set_xlim((-3, 3))

    axes[0, 0].set_title('QOI trace')
    axes[0, 1].set_title('Autocorrelation')
    axes[0, 2].set_title('PDF of posterior QOI')
    axes[-1, 0].set_xlabel('number of samples')
    axes[-1, 1].set_xlabel('lag')
    axes[-1, 2].set_xlabel('QOI')
    plt.show()
