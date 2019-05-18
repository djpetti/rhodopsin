try:
    from tensorflow.keras.callbacks import Callback
except ImportError:
    raise ImportError("Please install TensorFlow to use Keras integration"
                      " features.")


class DelegatingCallback(Callback):
    """
    A simple Keras callback that delegates to something else.
    """

    def __init__(self, on_epoch_end=None):
        """
        :param on_epoch_end: Function to call at the end of an epoch. Will be
        called with the same arguments as the Keras Callback.on_epoch_end()
        method.
        """
        super().__init__()

        if on_epoch_end:
            self.on_epoch_end = on_epoch_end
