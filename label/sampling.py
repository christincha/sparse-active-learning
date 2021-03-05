"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from sklearn.utils import shuffle

import logging
import numpy as np
import os
import random


logger = logging.getLogger('UST')

def get_BALD_acquisition(y_T):

	expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + 1e-10), axis=-1), axis=0)
	expected_p = np.mean(y_T, axis=0)
	entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
	return (entropy_expected_p - expected_entropy)

def sample_by_bald_difficulty(X, y_mean, y_var, y, num_samples, num_classes, y_T):
    logger.info("Sampling by difficulty BALD acquisition function")
    BALD_acq = get_BALD_acquisition(y_T)
    p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
    p_norm = p_norm / np.sum(p_norm)
    indices = np.random.choice(len(X), num_samples, p=p_norm, replace=False)
    X_s = [X[i] for i in indices]
    y_s = y[indices]
    w_s = y_var[indices][:, 0]
    return X_s, y_s, w_s


def sample_by_bald_easiness( X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by easy BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), (1. - BALD_acq)/np.sum(1. - BALD_acq))
	p_norm = p_norm / np.sum(p_norm)
	logger.info (p_norm[:10])
	indices = np.random.choice(len(X), num_samples, p=p_norm, replace=False)
	X_s = [X[i] for i in indices]
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_s, w_s


def sample_by_bald_class_easiness(X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by easy BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)
	logger.info (BALD_acq)
	samples_per_class = num_samples // num_classes
	X_s_input_ids = []
	y_s, w_s = np.array([]), np.array([])
	for label in range(1, num_classes+1):
		cla_id = np.where(y==label)[0]
		X_input_tmp = [X[i] for i in cla_id]
		y_ = y[cla_id]
		y_var_ = y_var[cla_id, :]
		# p = y_mean[y == label]
		p_norm = BALD_acq[y==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		if len(X_input_tmp) < samples_per_class:
			logger.info ("Sampling with replacement.")
			replace = True
		else:
			replace = False
		indices = np.random.choice(len(X_input_tmp), samples_per_class, p=p_norm, replace=replace)
		X_input_tmp =[X_input_tmp[i] for i in indices]
		X_s_input_ids.extend(X_input_tmp)
		y_s = np.hstack((y_s, y_[indices]))
		w_s= np.hstack((w_s, y_var_[indices, label-1]))
	return X_s_input_ids, np.asarray(y_s), np.asarray(w_s)

def sample_by_bald_class_difficulty(X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by easy BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	logger.info (BALD_acq)
	samples_per_class = num_samples // num_classes
	X_s_input_ids = []
	y_s, w_s = np.array([]), np.array([])
	for label in range(1,num_classes+1):
		cla_id = np.where(y==label)
		X_input_tmp = [X[i] for i in cla_id]
		y_ = y[y==label]
		y_var_ = y_var[y == label]
		# p = y_mean[y == label]
		p_norm = BALD_acq[y==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		if len(X_input_tmp) < samples_per_class:
			logger.info ("Sampling with replacement.")
			replace = True
		else:
			replace = False
		indices = np.random.choice(len(X_input_tmp), samples_per_class, p=p_norm, replace=replace)
		X_input_tmp =[X_input_tmp[i] for i in indices]
		X_s_input_ids.extend(X_input_tmp[indices])
		y_s = np.hstack((y_s, y_[indices]))
		w_s= np.hstack((y_s, y_var_[indices][:,label-1]))
	return X_s_input_ids, np.asarray(y_s), np.asarray(w_s)