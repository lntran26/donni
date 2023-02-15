'''
Library module of custom dadi models
'''
from dadi import Numerics, PhiManip, Integration, Spectrum


def out_of_africa(params, ns, pts):
    '''
    Gutenkunst et al. 2009 Out-of-Africa model.
    '''
    nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, \
        mAfB, mAfEu, mAfAs, mEuAs, \
        TAf, TB, TEuAs = params
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(
        phi, xx, TB, nu1=nuAf, nu2=nuB, m12=mAfB, m21=mAfB)

    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)

    def nuEu_func(t):
        return nuEu0 * (nuEu/nuEu0) ** (t/TEuAs)
    def nuAs_func(t):
        return nuAs0 * (nuAs/nuAs0) ** (t/TEuAs)
    phi = Integration.three_pops(phi, xx, TEuAs, nu1=nuAf, nu2=nuEu_func,
                                 nu3=nuAs_func, m12=mAfEu, m13=mAfAs,
                                 m21=mAfEu, m23=mEuAs, m31=mAfAs, m32=mEuAs)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
out_of_africa.__param_names__ = ['nuAf', 'nuB', 'nuEu0',
                                 'nuEu', 'nuAs0', 'nuAs',
                                 'mAfB', 'mAfEu', 'mAfAs', 'mEuAs',
                                 'TAf', 'TB', 'TEuAs']


def out_of_africa_no_mig(params, ns, pts):
    '''
    Gutenkunst et al. 2009 Out-of-Africa model with migration rates set to 0.
    '''
    nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, TAf, TB, TEuAs = params
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(
        phi, xx, TB, nu1=nuAf, nu2=nuB, m12=0, m21=0)

    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)

    def nuEu_func(t):
        return nuEu0 * (nuEu/nuEu0) ** (t/TEuAs)
    def nuAs_func(t):
        return nuAs0 * (nuAs/nuAs0) ** (t/TEuAs)
    phi = Integration.three_pops(phi, xx, TEuAs, nu1=nuAf, nu2=nuEu_func,
                                 nu3=nuAs_func, m12=0, m13=0,
                                 m21=0, m23=0, m31=0, m32=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
out_of_africa_no_mig.__param_names__ = ['nuAf', 'nuB', 'nuEu0',
                                        'nuEu', 'nuAs0', 'nuAs',
                                        'TAf', 'TB', 'TEuAs']
