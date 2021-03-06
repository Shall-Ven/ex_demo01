"""
This module provide the attack method for JSMA's implement.
"""
from __future__ import division

from builtins import range
import logging
import random
import numpy as np

from base import Attack


class SaliencyMapAttack(Attack):
    """
    Implements the Saliency Map Attack.
    The Jacobian-based Saliency Map Approach (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """

    def _apply(self,
               adversary,
               max_iter=2000,
               fast=True,
               theta=0.1,
               max_perturbations_per_pixel=7,
               two_pix=False):
        """
        Apply the JSMA attack.
        Args:
            adversary(Adversary): The Adversary object.
            max_iter(int): The max iterations.
            fast(bool): Whether evaluate the pixel influence on sum of residual classes.
            theta(float): Perturbation per pixel relative to [min, max] range.
            max_perturbations_per_pixel(int): The max count of perturbation per pixel.
            two_pix(bool): Whether to select two pixels from salience map which 
                is used in original paper.
        Return:
            adversary: The Adversary object.
        """
        assert adversary is not None

        #目前不支持双点
        two_pix=False



        if not adversary.is_targeted_attack or (adversary.target_label is None):
            target_labels = self._generate_random_target(
                adversary.original_label)
        else:
            target_labels = [adversary.target_label]

        for target in target_labels:
            original_image = np.copy(adversary.original)

            # the mask defines the search domain
            # each modified pixel with border value is set to zero in mask
            mask = np.ones_like(original_image)

            # count tracks how often each pixel was changed
            counts = np.zeros_like(original_image)

            labels = list(range(self.model.num_classes()))
            adv_img = original_image.copy()
            min_, max_ = self.model.bounds()

            for step in range(max_iter):
                adv_img = np.clip(adv_img, min_, max_)
                logit=self.model.predict(adv_img)
                adv_label = np.argmax(logit)
                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary

                # stop if mask is all zero
                if not any(mask.flatten()):
                    return adversary

                logging.info('step = {}, original_label = {}, adv_label={} target logit={}'.format(step, adversary.original_label, adv_label,logit[adversary.target_label]))

                # get pixel location with highest influence on class
                idx, p_sign = self._saliency_map(adv_img, target, labels, 
                            mask, fast=fast, two_pix=two_pix)
                if two_pix:
                    input_dimension = adv_img.shape[0]
                    idx = idx[0]
                    idx1 = idx // input_dimension
                    idx2 = idx % input_dimension
                    # apply perturbation
                    adv_img[idx1] += -p_sign * theta * (max_ - min_)
                    adv_img[idx2] += -p_sign * theta * (max_ - min_)
                    # tracks number of updates for each pixel
                    counts[idx1] += 1
                    counts[idx2] += 1
                    # remove pixel from search domain if it hits the bound
                    if adv_img[idx1] <= min_ or adv_img[idx1] >= max_:
                        mask[idx1] = 0
                    if adv_img[idx2] <= min_ or adv_img[idx2] >= max_:
                        mask[idx2] = 0
                    # remove pixel if it was changed too often
                    if counts[idx1] >= max_perturbations_per_pixel:
                        mask[idx1] = 0
                    if counts[idx2] >= max_perturbations_per_pixel:
                        mask[idx2] = 0
                else:
                    # apply perturbation


                    adv_img[idx] += p_sign * theta * (max_ - min_)
                    # tracks number of updates for each pixel
                    counts[idx] += 1
                    # remove pixel from search domain if it hits the bound
                    if adv_img[idx] <= min_ or adv_img[idx] >= max_:
                        logging.info('adv_img[idx] {} is over'.
                                     format(adv_img[idx]))
                        mask[idx] = 0
                    # remove pixel if it was changed too often
                    if counts[idx] >= max_perturbations_per_pixel:
                        logging.info('adv_img[idx] {} is over max_perturbations_per_pixel'.
                                     format(adv_img[idx]))
                        mask[idx] = 0

                adv_img = np.clip(adv_img, min_, max_)
            
            # if attack remains unsuccessful within max iteration
            if step == max_iter - 1:
                return adversary

    def _generate_random_target(self, original_label):
        """
        Draw random target labels all of which are different and not the original label.
        Args:
            original_label(int): Original label.
        Return:
            target_labels(list): random target labels
        """
        num_random_target = 1
        num_classes = self.model.num_classes()
        assert num_random_target <= num_classes - 1

        target_labels = random.sample(list(range(num_classes)), num_random_target + 1)
        target_labels = [t for t in target_labels if t != original_label]
        target_labels = target_labels[:num_random_target]

        return target_labels

    def _saliency_map(self, image, target, labels, mask, fast=False, two_pix=True):
        """
        Get pixel location with highest influence on class.
        Args:
            image(numpy.ndarray): Image with shape (height, width, channels).
            target(int): The target label.
            labels(int): The number of classes of the output label.
            mask(list): Each modified pixel with border value is set to zero in mask.
            fast(bool): Whether evaluate the pixel influence on sum of residual classes.
            two_pix(bool): Whether to select two pixels from salience map which 
                is used in original paper.
        Return:
            idx: The index of optimal pixel (idx = idx1 * input_dimension +
                idx2 in two pix setting)
            pix_sign: The direction of perturbation
        """
        # pixel influence on target class
        alphas = self.model.gradient(image, target) * mask

        # pixel influence on sum of residual classes(don't evaluate if fast == True)
        if fast:
            betas = -np.ones_like(alphas)
        else:
            betas = np.sum([
                self.model.gradient(image, label) * mask - alphas
                for label in labels
            ], 0)

        if two_pix:
            # construct two pix matrix to treat it as one pix
            alphas_col = np.expand_dims(alphas, axis=0)
            alphas_mat = alphas_col.transpose(1, 0) + alphas_col
            # we want to select two different piexl
            np.fill_diagonal(alphas_mat, 0)
            alphas_mat = alphas_mat.reshape(-1)
            betas_col = np.expand_dims(betas, axis=0)
            betas_mat = betas_col.transpose(1, 0) + betas_col
            # we want to select two different piexl
            np.fill_diagonal(betas_mat, 0)
            betas_mat = betas_mat.reshape(-1)
            # compute saliency map (take into account both pos. & neg.
            # perturbations)
            sal_map = np.abs(alphas_mat) * np.abs(betas_mat) * np.sign(
                alphas_mat * betas_mat)
            # find optimal pixel & direction of perturbation
            idx = np.argmin(sal_map)
            idx = np.unravel_index(idx, sal_map.shape)
            pix_sign = np.sign(alphas_mat)[idx]
        else:
            # compute saliency map (take into account both pos. & neg.
            # perturbations)
            sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)
            # find optimal pixel & direction of perturbation
            idx = np.argmin(sal_map)
            idx = np.unravel_index(idx, mask.shape)
            pix_sign = np.sign(alphas)[idx]

        return idx, pix_sign


JSMA = SaliencyMapAttack