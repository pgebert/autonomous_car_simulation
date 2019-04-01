import os
import ujson
import plot as pl
import pandas as pd

class Logger:
    def __init__(self, cfg):

        self.data = []
        self.cfg = cfg

        # load json data
        log_path = os.path.join(self.cfg.log_dir, self.cfg.log_file)
        if os.path.isfile(log_path) and os.stat(log_path).st_size != 0:
            with open(log_path, 'r') as f:
                self.data = ujson.load(f)

        # append current configuration
        msg = (
            "BatchSize: " + str(cfg.batch_size)
            + ", Epochs: " + str(cfg.train_epochs)
            + ", TestRate: " + str(cfg.test_rate)
            + ", Optimizer: " + str(cfg.optimizer)
        )
        self.data.append({"cfg": msg})
        # add attributes for loss and test
        self.data[-1]["loss"] = []
        self.data[-1]["test"] = []

        with open(log_path, 'w+') as jsonFile:
            ujson.dump(self.data, jsonFile)

    def logLoss(self, data):
        """
        Writes loss values to the json log file.

        Args:
            data - tuple containing the epoch and the loss value
        """
        log_path = os.path.join(self.cfg.log_dir, self.cfg.log_file)

        self.data[-1]["loss"].append(data)
        with open(log_path, 'w+') as jsonFile:
            ujson.dump(self.data, jsonFile)
        
        # Update plot
        if self.cfg.auto_plot:
            out_path = os.path.join(self.cfg.log_dir, self.cfg.plot_file)
            pl.plot(data=self.data, out_path=out_path)


    def logTest(self, data):
        """
        Writes test values to the json log file.

        Args:
            data - tuple containing the epoch and the test accurancy
        """
        log_path = os.path.join(self.cfg.log_dir, self.cfg.log_file)

        self.data[-1]["test"].append(data)
        with open(log_path, 'w+') as jsonFile:
            ujson.dump(self.data, jsonFile)

        # Update plot
        if self.cfg.auto_plot:
            out_path = os.path.join(self.cfg.log_dir, self.cfg.plot_file)
            pl.plot(data=self.data, out_path=out_path)

    def logCrossValidation(self, data):
        """
        Writes the average result from cross validation to the json log file.

        Args:
            data (int) - the cross validation result
        """
        log_path = os.path.join(self.cfg.log_dir, self.cfg.log_file)
        self.data[-1]["crossvalidation"] = data
        with open(log_path, 'w+') as jsonFile:
            ujson.dump(self.data, jsonFile)