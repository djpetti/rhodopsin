import abc

from .delegating_callback import DelegatingCallback
from .experiment_base import ExperimentBase


class KerasExperiment(ExperimentBase):
    """
    An experiment meant to better integrate with Keras.
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: Will be forwarded to the base class.
        :param kwargs: Will be forwarded to the base class.
        """
        super().__init__(*args, **kwargs)

        # If true, indicates that we should enter the menu at the end of the
        # next epoch.
        self.__enter_menu = False

        # Users should add this callback to their model so that the Rhodopsin
        # menu works.
        self.__callback = DelegatingCallback(on_epoch_end=self.__on_epoch_end)

    def __on_epoch_end(self, epoch, logs=None):
        """
        Function to be called at the end of an epoch.
        :param epoch: The epoch number.
        :param logs: The log dictionary from Keras.
        """
        self._update_after_epoch(epoch, logs=logs)

        if not self.__enter_menu:
            # Nothing to do.
            return

        # Enter the menu.
        self._show_main_menu()
        self.__enter_menu = False

        # Update the model.
        self._update_after_menu()
        # Save the model.
        self._checkpoint()

    @abc.abstractmethod
    def _update_after_menu(self):
        """
        This function is run after the user exits from a menu. It should
        perform any reconfiguration that is necessary due to the user's actions.
        """
        pass

    @abc.abstractmethod
    def _update_after_epoch(self, epoch, logs=None):
        """
        This function is run after an epoch finishes. It should update any
        status values that require this.
        :param epoch: The epoch number.
        :param logs: The log dictionary from Keras.
        """
        pass

    def _handle_signal(self, signum, frame):
        """ Handles the user hitting Ctrl+C. This is supposed to bring up the
        menu.
        Args:
          signum: The signal number that triggered this.
          frame: Current stack frame. """
        if self.__enter_menu:
            # Already entering the menu.
            return

        # Give some immediate feedback.
        print("Signal caught, entering menu after current epoch.")

        # Indicate that we want to enter the menu on the next iteration.
        self.__enter_menu = True

    def _run_testing_step(self):
        """
        Does nothing, because we assume that, for Keras, all testing is handled
        internally by the call to fit().
        """
        pass

    def get_callback(self):
        """
        Gets the callback that should be added to the model to make Rhodopsin
        work.
        :returns: The callback.
        """
        return self.__callback

    def train(self):
        """
        Runs the training procedure to completion.
        """
        # For Keras, we assume that there is only one training step, and that
        # it does everything, including testing.
        self._run_training_step()
