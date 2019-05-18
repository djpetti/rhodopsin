import abc
import os
import signal

from . import menu
from . import params


class ExperimentBase(abc.ABC):
    """ Base class for experiments that defines the API. """

    def __init__(self, save_file="experiment.rhp", hyperparams=None,
                 status=None):
        """
        Args:
          save_file: File in which to save the model data.
          hyperparams: Optional custom hyperparameters to use.
          status: Optional custom status parameters to use. """
        self.__save_file = save_file

        # Create hyperparameters.
        self.__params = hyperparams
        if self.__params is None:
            self.__params = params.HyperParams()

        # Create status parameters.
        self.__status = status
        if self.__status is None:
            self.__status = params.Status()

        # Add default status parameters.
        self.__status.add_if_not_set("iterations", 0)

        # Register the signal handler.
        signal.signal(signal.SIGINT, self._handle_signal)

        # Create the menu tree.
        self.__menus = menu.MenuTree()
        main_menu = menu.MainMenu(self.__params, self.__status)
        adjust_menu = menu.AdjustMenu(self.__params, self.__status)
        status_menu = menu.StatusMenu(self.__params, self.__status)
        self.__menus.add_menu(main_menu)
        self.__menus.add_menu(adjust_menu)
        self.__menus.add_menu(status_menu)

        # Run custom initialization code.
        self._init_experiment()

        # Check for an existing model.
        if self._model_exists(self.__save_file):
            load_menu = menu.LoadModelMenu(self.__params, self.__status,
                                           self.__save_file)
            load_menu.show()

            # Check what was selected.
            if load_menu.should_load():
                # Load the model.
                self._load_model(self.__save_file)

    @abc.abstractmethod
    def _handle_signal(self, signum, frame):
        """ Handles the user hitting Ctrl+C. This is supposed to bring up the
        menu.
        Args:
          signum: The signal number that triggered this.
          frame: Current stack frame. """
        pass

    def _show_main_menu(self):
        """ Show the main menu. """
        self.__menus.show("main")

    def _checkpoint(self):
        """
        Saves the model at this point.
        """
        self._save_model(self.__save_file)

    def _init_experiment(self):
        """ Runs any custom initialization code for the experiment. This will be
        run right after we've configured parameters and hyperparameters, and
        before we've attempted to load the model. By default, it does nothing.
        """
        pass

    @abc.abstractmethod
    def _run_training_step(self):
        """ Runs a single training iteration. This is meant to be overidden by a
        subclass. """
        pass

    @abc.abstractmethod
    def _run_testing_step(self):
        """ Runs a single testing iteration. This is meant to be overidden by a
        subclass. """
        pass

    def _save_model(self, save_file):
        """ Saves the model. By default, it does nothing. It should be
        implemented by a subclass.
        Args:
          save_file: The path at which to save the model. """
        pass

    def _load_model(self, save_file):
        """ Loads a model from disk. If _save_model() is used, this must be
        implemented by a subclass. Note that this is not an abstract method,
        because if save_model is not used, it need not be implemented either.
        Args:
          save_file: The path from which to load the model. """
        raise NotImplementedError(
            "_load_model() must be implemented by subclass.")

    @classmethod
    def _model_exists(cls, save_file):
        """ Checks if a saved model exists. By default, it just checks if
        save_path exists, but it can be overridden to allow for more
        sophisticated functionality.
        Args:
          save_file: The possible path to the saved model. """
        return os.path.exists(save_file)

    @abc.abstractmethod
    def train(self):
        """ Runs the training procedure to completion. """
        pass

    def get_params(self):
        """
        Returns:
          The hyperparameters being used for this experiment. """
        return self.__params

    def get_status(self):
        """
        Returns:
          The status parameters being used for this experiment. """
        return self.__status
