from .experiment_base import ExperimentBase


class Experiment(ExperimentBase):
    """ Facilitates user control of deep learning experiments by providing a CLI
    that can be used to evaluate the model and set hyperparameters during
    training. """

    def __init__(self, testing_interval, **kwargs):
        """
        Args:
          testing_interval: How many training iterations to run for every
          testing iteration.
          kwargs: Will be forwarded to the base class constructor.
        """
        super().__init__(**kwargs)

        # Whether we want to enter the menu as soon as we can.
        self.__enter_menu = False
        self.__testing_interval = testing_interval

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
        print("Signal caught, entering menu after current iteration.")

        # Indicate that we want to enter the menu on the next iteration.
        self.__enter_menu = True

    def train(self):
        """ Runs the training procedure to completion. """
        while True:
            # Run training and testing iterations.
            for i in range(0, self.__testing_interval):
                if self.__enter_menu:
                    # Show the menu.
                    self._show_main_menu()
                    self.__enter_menu = False

                    # Save after the user adjusts something.
                    self._checkpoint()

                self._run_training_step()

                # Update the iteration counter.
                iterations = self.__status.get_value("iterations")
                self.__status.update("iterations", iterations + 1)

            self._run_testing_step()
            # Save the model after each testing iteration.
            self._checkpoint()
