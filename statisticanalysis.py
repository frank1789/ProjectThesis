#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod

import matplotlib

# This needs to be done *before* importing pyplot or pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BaseHistoryAnalysis(ABC):
    def __init__(self):
        super(BaseHistoryAnalysis, self).__init__()

    @abstractmethod
    def generate_plot(self, epochs, history, namedir) -> None:
        pass

    @abstractmethod
    def prepare_folder(self, namedir) -> str:
        pass


class MaskRCNNAnalysis(BaseHistoryAnalysis):
    def __init__(self):
        super().__init__()

    def generate_plot(self, epochs, history, namedir) -> None:
        """
        function that extracts the information of the training history from the keras model to the specific Mask - R CNN.

        Parameters
        ----------
        :param epochs: (int) number of epochs.
        :param history: (object) history from keras model.
        :param namedir: (str) name of directory where save plot.
        """
        save_directory = self.prepare_folder(namedir)
        generate_name = (lambda name: "{:s}_loss.png".format(name))

        # export train-valid loss
        train_valid = os.path.join(save_directory, generate_name("train_valid"))
        plt.plot(len(history["loss"]), history["loss"], label="Train loss")
        plt.plot(len(history["val_loss"]), history["val_loss"], label="Valid loss")
        plt.legend()
        plt.savefig(train_valid)
        plt.close()

        # export train-valid class
        class_valid = os.path.join(save_directory, generate_name("train_valid_class"))
        plt.plot(len(history["mrcnn_class_loss"]), history["mrcnn_class_loss"], label="Train class ce")
        plt.plot(len(history["val_mrcnn_class_loss"]), history["val_mrcnn_class_loss"], label="Valid class ce")
        plt.legend()
        plt.savefig(class_valid)
        plt.close()

        # export  train-valid box
        box_valid = os.path.join(save_directory, generate_name("train_valid_box"))
        plt.plot(len(history["mrcnn_bbox_loss"]), history["mrcnn_bbox_loss"], label="Train box loss")
        plt.plot(len(history["val_mrcnn_bbox_loss"]), history["val_mrcnn_bbox_loss"], label="Valid box loss")
        plt.legend()
        plt.savefig(box_valid)
        plt.close()

    def prepare_folder(self, namedir) -> str:
        """
        function that checks if the destination folder exists to save the plots. If it does not exist, it constructs it
        and returns the path as a string.

        Parameters
        ----------
        :param namedir: (str) folder's name.

        Return
        ------
        :return: (str) path.
        """
        plot_folder = os.path.join("result_plot", namedir)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        return plot_folder


class HistoryAnalysis:
    @staticmethod
    def plot_history(history, namefile):
        """
        Collects the history, returned from training the model and creates two charts:
        A plot of accuracy on the training and validation datasets over training epochs.
        A plot of loss on the training and validation datasets over training epochs.

        Parameters
        ----------
        :param history: (dict) from keras fit
        :param namefile: (str) set name save file

        Return
        ------
        :return: plt(object) plot
        """
        # make new directory
        new_dir = 'ResultPlot'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # plot's name
        loss_plot = os.path.join(new_dir, "{:s}_loss.png".format(namefile))
        acc_plot = os.path.join(new_dir, "{:s}_accuracy.png".format(namefile))

        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return
        else:
            pass

        # As loss always exists
        epochs = range(1, len(history.history[loss_list[0]]) + 1)

        # Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # save figure loss
        plt.savefig(loss_plot)

        # Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
        for l in val_acc_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # save figure loss
        plt.savefig(acc_plot)
        return plt
