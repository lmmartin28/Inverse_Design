#!/usr/bin/env python
# coding: utf-8
"""
date: 06/2023
@author: P. Wiecha

helper functions for deep learning inverse design tutorials
"""
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras


import PyMoosh as pym


# data loading
def load_reflection_spectra_data(path_h5, save_scalers=True, test_fraction=0.05):
    import h5py
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    with h5py.File(path_h5) as f_read:
        R_spec = np.array(f_read['R'], dtype=np.float32)
        mat = np.array(f_read['mat'], dtype=np.float32)
        thick = np.array(f_read['thick'], dtype=np.float32)
        wavelengths = np.array(f_read['wavelengths'], dtype=np.float32)

    # if necessary, add a channel dimension to the spectra (keras: channels last)
    if R_spec.shape[-1] != 1:
        R_spec = np.expand_dims(R_spec, -1)

    #  separately standardize permittivities and thicknesses
    scaler_mat = StandardScaler().fit(mat)
    scaler_thick = StandardScaler().fit(thick)

    # save the scalers using pickle
    if save_scalers:
        pickle.dump([scaler_mat, scaler_thick],
                    open('{}_scalers.pkl'.format(os.path.splitext(path_h5)[0]), 'wb'))

    # apply scaler and combine materials and thicknesses
    mat_thick = scale_and_combine_Rspec_data(
        mat, thick, scaler_mat, scaler_thick)

    # split into training and test datasets. Set random state for a reproducible splitting
    x_train, x_test, y_train, y_test = train_test_split(
        mat_thick, R_spec, test_size=test_fraction, random_state=2)

    return x_train, x_test, y_train, y_test


def scale_and_combine_Rspec_data(mat, thick, scaler_mat, scaler_thick):
    """apply scaler transforms and concatenate materials and thicknesses"""

    #  separately standardize permittivities and thicknesses
    mat = scaler_mat.transform(mat)
    thick = scaler_thick.transform(thick)

    # concatenate materials and thicknesses. keras: last dimension is channel
    mat_thick = np.stack([mat, thick], axis=-1)

    return mat_thick


def inverse_scale_mat_thick(mat, thick, scaler_mat, scaler_thick):
    """mat and thick are predicted, normalized values. return their inverse transforms"""
    mat_physical = scaler_mat.inverse_transform(mat)
    thick_physical = scaler_thick.inverse_transform(thick)

    return mat_physical, thick_physical

# =============================================================================
# residual blocks
# =============================================================================


def residual_block(x_in, N_filter, kernel_size=3, strides=1,
                   conv_layer=keras.layers.Conv1D, alpha=0.3,
                   with_BN=False):
    """resnet block, default: 1D convolutions"""

    # residual connection
    if x_in.shape[-1] != N_filter or strides != 1:
        # if input!=output dimension: add BN/ReLU/conv. into shortcut
        conv_shortcut = conv_layer(
            filters=N_filter, kernel_size=1, strides=strides, padding='same')(x_in)
    else:
        # if input==output dimension: use bare input as shortcut
        conv_shortcut = x_in

    # convolutional path
    x = x_in

    x = conv_layer(filters=N_filter, kernel_size=1, strides=1,
                   padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)

    x = conv_layer(filters=N_filter, kernel_size=kernel_size,
                   strides=strides, padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)

    x = conv_layer(filters=N_filter, kernel_size=1, strides=1,
                   padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)

    # add residual and main and apply a further activation
    x = keras.layers.Add()([x, conv_shortcut])
    x = keras.layers.LeakyReLU(alpha)(x)

    return x


def id_map_conv_block(x_in, N_filter, kernel_size=3, strides=1, alpha=0.3, with_BN=False):
    """resnet identity mapping block, default: 1D convolutions"""
    x = x_in
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)

    # residual connection
    if x_in.shape[-1] != N_filter or strides != 1:
        # if input!=output dimension: add BN/ReLU/conv. into shortcut
        conv_shortcut = keras.layers.Conv1D(
            filters=N_filter, kernel_size=1, strides=strides, padding='same')(x_in)
    else:
        # if input==output dimension: use bare input as shortcut
        conv_shortcut = x_in

    # main convolution path
    x = keras.layers.Conv1D(filters=N_filter, kernel_size=1,
                            strides=strides, padding='same', use_bias=not with_BN)(x)

    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)
    x = keras.layers.Conv1D(filters=N_filter, kernel_size=kernel_size,
                            strides=1, padding='same', use_bias=not with_BN)(x)

    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)
    x = keras.layers.Conv1D(filters=N_filter, kernel_size=1,
                            strides=1, padding='same', use_bias=not with_BN)(x)

    # add shortcut and main path
    x = keras.layers.Add()([x, conv_shortcut])
    return x


# =============================================================================
# re-calculate reflectivity of prediction
# =============================================================================
def calc_R(thicknesses, materials, wavelengths,
           n_env=1.0, theta_incidence=0, polarization=0):
    """
    theta_incidence (rad): 0=normal incidence
    polarization: 0=s, 1=p
    """
    # we add semi-inifinity bottom and top vacuum layers
    thicknesses = [0.] + list(thicknesses) + [0.]
    materials = [1.] + list(materials) + [1.]
    mat_indices = np.arange(len(thicknesses))

    # create the structure
    struct = pym.Structure(materials, mat_indices, thicknesses, verbose=False)

    # spectrum
    R = np.zeros(len(wavelengths))
    for i, wl in enumerate(wavelengths):
        _r, _t, R[i], _T = pym.coefficient(
            struct, wl, theta_incidence, polarization)

    return R


def batch_calc_R(thicknesses_many, materials_many, wavelengths,
                 n_env=1.0, theta_incidence=0, polarization=0, verbose=True):
    from tqdm import tqdm

    """batch process several layer-stack reflectivity calculations"""
    all_R = []
    if verbose:
        print('pymoosh: calculating R for {} samples...'.format(len(thicknesses_many)))
    for [_t, _m] in tqdm(zip(thicknesses_many, materials_many)):
        _R = calc_R(_t, _m, wavelengths, n_env, theta_incidence, polarization)
        all_R.append(_R)

    return np.array(all_R)


def calc_R_for_network_designs(designs_predict, scaler_mat, scaler_thick,
                               wavelengths=np.linspace(500, 1500, 64)):

    # scaler inverse transform
    thick_physical, mat_physical, wavelengths = get_design_from_network_output(
        designs_predict, scaler_mat, scaler_thick, wavelengths)
    # pymoosh calculation
    y_recalc = batch_calc_R(thick_physical, mat_physical, wavelengths)

    return y_recalc


def get_design_from_network_output(
        y_predict,  scaler_mat=None, scaler_thick=None,
        wavelengths=np.linspace(500, 1500, 64)):

    _mat = y_predict[..., 0]
    _thick = y_predict[..., 1]

    # scaler inverse transform: normalized datarange --> physical datarange
    if scaler_mat is not None:
        materials, thicknesses = inverse_scale_mat_thick(
            _mat, _thick, scaler_mat, scaler_thick)

    return thicknesses, materials, wavelengths


# =============================================================================
#  plot a layer stack
# =============================================================================
def plot_stack(materials, thicknesses, lim_eps_colors=[2, 4.5]):
    # give sub- and superstrate finite thickness
    _thick = [0] + list(thicknesses) + [0]
    _thick[0] = 50
    _thick[-1] = 50
    _mats = [1] + list(materials) + [1]  # air environment

    # reverse order so that top layer is on top in plot
    _thick = _thick[::-1]
    _mats = _mats[::-1]

    # define colors for layers of different ref.indices
    if any(isinstance(x, str) for x in _mats):
        colors = ['.3'] + ['C{}'.format(i)
                           for i in range(len(_thick)-2)] + ['.3']
    else:
        cmap = plt.cm.jet
        colors = ['.3'] + [cmap((n-lim_eps_colors[0])/lim_eps_colors[1])
                           for n in _mats[1:-1]] + ['.3']

    for i, di in enumerate(_thick):
        d0 = np.sum(_thick[:i])
        n = _mats[i]

        if type(n) is not str:
            n = str(np.round(n, 2))

        if i < len(_thick)-1:
            plt.axhline(d0+di, color='k', lw=1)
        plt.axhspan(d0, d0+di, color=colors[i], alpha=0.5)
        if len(_thick)-1 > i >= 1:
            plt.text(0.05, d0+di/2, 'eps={:}'.format(n),
                     ha='left', va='center', fontsize=8)
            plt.text(0.95, d0+di/2, 'd={}nm'.format(int(np.round(di))),
                     ha='right', va='center', fontsize=8)
        else:
            plt.text(0.1, d0+di/2, 'eps={:}'.format(n),
                     ha='left', va='center', fontsize=8)
    plt.ylabel('D (nm)')
    plt.xticks([])


def plot_stack_from_raw_predict(designs_predict, scaler_mat, scaler_thick, lim_eps_colors=[2, 4.5]):
    if designs_predict.ndim == 2:
        designs_predict = designs_predict[None, ...]

    if len(designs_predict) > 1:
        print("Warning! Got more than 1 designs to plot. Plotting first one.")

    # scaler inverse transform
    thick_physical, mat_physical, wavelengths = get_design_from_network_output(
        designs_predict, scaler_mat, scaler_thick)

    plot_stack(mat_physical[0], thick_physical[0], lim_eps_colors)


# =============================================================================
#  benchmark a model and plot results
# =============================================================================
def plot_benchmark_R_samples(y_predict, y_test, N_plot=(2, 3), labels=['fwd-net', 'pymoosh'], random_order=True,
                             plot_design=False, y_designs=None, scaler_mat=None, scaler_thick=None):
    # plot samples
    if N_plot is not None:
        plt.figure(figsize=(2.5 * (1 + 1*plot_design) * N_plot[1],
                            2 * N_plot[0]))
        if random_order:
            idx_rnd = np.random.choice(
                len(y_test), size=np.prod(N_plot), replace=False)
        else:
            idx_rnd = np.arange(np.prod(N_plot))
        for i, index in enumerate(idx_rnd):
            # plot geometry (optional)
            if plot_design:
                plt.subplot(N_plot[0], (1 + 1*plot_design) *
                            N_plot[1], 2*i + 2)
                plot_stack_from_raw_predict(
                    y_designs[index], scaler_mat, scaler_thick)
            # plot spectra
            plt.subplot(N_plot[0], (1 + 1*plot_design) *
                        N_plot[1], i * (1 + 1*plot_design) + 1)
            plt.plot(y_predict[index], label=labels[0], linewidth=1)
            plt.plot(y_test[index], label=labels[1], linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_benchmark_R_stat(y_predict, y_test):
    # plot statistics
    err = np.squeeze(y_predict) - np.squeeze(y_test)
    err_mean = np.mean(err, axis=-1)
    err_max = np.max(np.abs(err), axis=-1)

    plt.figure(figsize=(8, 3))
    plt.subplot(121, title='average absolute error')
    plt.hist(err_mean, bins=50)
    plt.xlim(-np.quantile(np.abs(err_mean), 0.99),
             np.quantile(np.abs(err_mean), 0.99))  # don't plot outliers

    plt.subplot(122, title='spectrum peak absolute error')
    plt.hist(err_max, bins=50)
    plt.xlabel('absolute reflectivity error', x=0)
    plt.xlim(-0.01, np.quantile(err_max, 0.995))  # don't plot outliers

    plt.tight_layout()
    plt.show()


def plot_benchmark_R_model(y_predict, y_test, N_plot=(2, 3), labels=['fwd-net', 'pymoosh']):
    plot_benchmark_R_stat(y_predict, y_test)
    plot_benchmark_R_samples(y_predict, y_test, N_plot, labels)


# =============================================================================
# Neural Adjoint
# =============================================================================
# --- losses
def L2_designtarget(y_target, y_pred):
    # design loss: Reproduce target spectrum
    return keras.backend.mean(keras.backend.square(y_pred - y_target), axis=(1, 2))


def loss_param_constraint(x_design):
    # parameter constraint loss: penalize design parameters outside normalization range
    return keras.backend.mean(tf.keras.activations.relu((keras.backend.abs(x_design) - 1.5), alpha=0.0))


# --- regularized neural adjoint gradient evaluation
@tf.function  # compile function to tensorflow static graph --> ~2x faster
def train_step_neural_adjoint(input_geo, input_target, weight_constraint_geo, fwd_model):
    with tf.GradientTape() as tape:
        tape.watch(input_geo)
        # predict the scattering by passing latent through generator and fwd model
        pred = fwd_model(input_geo)

        # calc fitness of physics response
        loss1 = L2_designtarget(input_target, pred)
        loss2 = loss_param_constraint(input_geo)
        fitness = loss1 + loss2*weight_constraint_geo

    # latent-value-wise gradients with respect to the loss, optimize `input_z`
    gradients = tape.gradient(fitness, input_geo)
    grad_rescaled = 1E6 * gradients / len(input_target)
    return grad_rescaled  # return the gradients


# helper: status text
def status_text_NA(inv_design_geo, input_target, weight_constraint_geo, fwd_model):
    y_pred = fwd_model(inv_design_geo)
    L2_loss_target = L2_designtarget(input_target, y_pred)
    loss_constr = loss_param_constraint(inv_design_geo)
    loss_total = L2_loss_target + loss_constr*weight_constraint_geo

    status_text = "best: T-loss {:.2e}, constraint_loss {:.2e}, total {:.2e}".format(
        np.min(L2_loss_target), np.min(loss_constr), np.min(loss_total))

    return status_text

# helper: sorting results by loss


def sort_NA_results(inv_design_geo, input_target, weight_constraint_geo, fwd_model):
    # --- end of loop: sort results as function of loss
    y_pred = fwd_model(inv_design_geo)
    L2_loss_target = L2_designtarget(input_target, y_pred)
    loss_constr = loss_param_constraint(inv_design_geo)
    loss_total = L2_loss_target + loss_constr*weight_constraint_geo

    idx_sort = np.argsort(L2_loss_target, axis=0)
    inv_design_sorted = inv_design_geo.numpy()[idx_sort]
    del inv_design_geo

    return loss_total, inv_design_sorted


#  neural adjoint main function
def do_NA(init_geo, design_target, fwd_model, optimizer, 
          N_epoch, weight_constraint_geo=0.0):
    # --- initialize variables
    N_population = len(init_geo)
    init_geo = tf.convert_to_tensor(init_geo, dtype=tf.float32)
    # tf variable --> optimizer can modify
    inv_design_geo = tf.Variable(init_geo)

    # copy target design for each geometry
    target_spec_Ntimes = np.tile(design_target, (N_population, 1, 1))
    target_spec_Ntimes = tf.convert_to_tensor(
        target_spec_Ntimes, dtype=tf.float32)

    # --- the actual optimization loop
    # input_geo = inv_design_geo
    input_target = target_spec_Ntimes
    pbar = tqdm(range(N_epoch))
    for i in pbar:
        # calc. gradients; optimizer to adjust the designs
        grad_rescaled = train_step_neural_adjoint(
            inv_design_geo, input_target, weight_constraint_geo, fwd_model)
        optimizer.apply_gradients(zip([grad_rescaled], [inv_design_geo]))

        # status printing
        if i % 5 == 0:
            status_text = status_text_NA(
                inv_design_geo, input_target, weight_constraint_geo, fwd_model)
            pbar.set_description(status_text)

    loss_total, inv_design_sorted = sort_NA_results(
        inv_design_geo, input_target, weight_constraint_geo, fwd_model)

    return loss_total, inv_design_sorted


# %%
# =============================================================================
# nano-structure scattering calculation via pyGDM
# these tools require pygdm2.
# https://homepages.laas.fr/pwiecha/pygdm_doc/
# =============================================================================
def img_to_geo(geo_img, step, H):
    from pyGDM2 import structures

    geo2d = np.squeeze(geo_img)
    if len(geo2d.shape) == 3:
        geo2d = geo_img[..., 0]
    geo2d = 255 * (geo2d + 1) / 2

    geo = structures.image_to_struct(
        geo2d, useDarkPixel=False,
        nm_per_pixel=step, stepsize=step, H=H
    )

    return geo


def gen_sim(geo, wavelengths=np.linspace(500, 1000, 50), material=None):
    """setup pyGDM simulation"""
    from pyGDM2 import core
    from pyGDM2 import structures
    from pyGDM2 import tools
    from pyGDM2 import propagators
    from pyGDM2 import materials
    from pyGDM2 import fields
    if material is None:
        material = materials.dummy(3.5 + 0.01j)

    # Setup incident field
    field_generator = fields.planewave
    kwargs = dict(theta=[0.0, 90])

    efield = fields.efield(
        field_generator, wavelengths=wavelengths, kwargs=kwargs)

    # environment - vacuum
    dyads = propagators.DyadsQuasistatic123(1)

    # Setup structure
    step = tools.get_step_from_geometry(geo)
    struct = structures.struct(step, geo, material)

    # ---------- Simulation initialization
    sim = core.simulation(struct, efield, dyads)

    return sim


def calc_scatspec(sim, method='lu', verbose=False):
    from pyGDM2 import tools
    from pyGDM2 import linear
    sim.scatter(method=method, verbose=verbose)

    # spectrum via tools
    wl, spec_ext0 = tools.calculate_spectrum(sim, 0, linear.extinct)
    wl, spec_ext90 = tools.calculate_spectrum(sim, 1, linear.extinct)

    return spec_ext0[:, 0], spec_ext90[:, 0]


def calc_scattering_pygdm(geo_img, method='lu',
                          wavelengths=np.linspace(500, 1000, 50),
                          verbose=False):
    geo = img_to_geo(geo_img, step=30, H=5)
    sim = gen_sim(geo, wavelengths)
    sc_recalc = np.stack(calc_scatspec(sim, method=method,
                         verbose=verbose), axis=-1)
    return sc_recalc


def apply_best_threshold(geo, T_target, model_fwd, N_thresholds=25):
    """use the forward net to find best geometry threshold"""
    
    all_geo = []
    all_mse = []
    all_Npert = []
    
    test_thresholds = np.linspace(-1, 1, N_thresholds)
    pbar = tqdm(test_thresholds)
    for th in pbar:
        geo_th = geo.copy()
        geo_th[geo_th < th] = -1
        geo_th[geo_th >= th] = 1
        N_pert = np.count_nonzero(geo_th>0, axis=(1,2))[...,0]

        # calculate loss wrt target
        T_p_th = model_fwd.predict(geo_th, verbose=False)
        mse = np.square(T_p_th - T_target[None,...]).mean(axis=(1,2))
        
        all_mse.append(mse)
        all_geo.append(geo_th)
        all_Npert.append(N_pert)
        
        pbar.set_description("optimizing pattern binarization...")
    
    geo_best = np.array(all_geo)[
                np.argmin(all_mse, axis=0),   # best threshold
                np.arange(len(geo_th)),       # each geometry
                ...]
    threshold_values = test_thresholds[np.argmin(all_mse, axis=0)]
    return geo_best, threshold_values


def plot_scattering_design(geo_design, geo_val, sc_target, sc_norm,
                           sc_fwd=None, sc_recalc='recalc',
                           wavelengths=np.linspace(500, 1000, 50),
                           pygdm_method='lu', verbose=1):
    if sc_recalc == 'recalc':
        sc_recalc = calc_scattering_pygdm(
            geo_design, method=pygdm_method, wavelengths=wavelengths, verbose=verbose)

    plt.figure(figsize=(6, 6))

    plt.subplot(211, title='scattering sections')
    if sc_recalc is not None:
        plt.plot(wavelengths, sc_recalc, label=['X-sim', 'Y-sim'])
    plt.gca().set_prop_cycle(None)
    if sc_fwd is not None:
        plt.plot(wavelengths, sc_norm*sc_fwd,
                 dashes=[2, 4], label=['X-fwd', 'Y-fwd'])
    plt.gca().set_prop_cycle(None)
    plt.plot(wavelengths, sc_norm*sc_target,
             dashes=[1, 1], lw=1, label=['X-target', 'Y-target'])

    plt.legend()
    plt.xlabel('wavelength (nm)')
    plt.ylabel('SCS (nm^2)')

    plt.subplot(223, title='neural adjoint')
    plt.imshow(geo_design)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224, title='used reference')
    plt.imshow(geo_val)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
