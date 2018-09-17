import unittest

from .. import params


class TestParams(unittest.TestCase):
  """ Tests for the Params superclass. """

  def setUp(self):
    # Create instance to test with.
    self._params = params.HyperParams()

  def test_add(self):
    """ Tests that add() works under normal conditions. """
    self._params.add("param1", 0)
    self._params.add("param2", 1)

    # They should be both in there.
    self.assertEqual(0, self._params.get_value("param1"))
    self.assertEqual(1, self._params.get_value("param2"))

  def test_add_twice(self):
    """ Tests that add() does not allow us to add the same thing twice. """
    self._params.add("param", 0)

    self.assertRaises(NameError, self._params.add, "param", 0)

  def test_add_if_not_set(self):
    """ Tests that add_if_not_set() works under normal conditions. """
    self._params.add_if_not_set("param", 0)
    # It should be in there now.
    self.assertEqual(0, self._params.get_value("param"))

    # Try adding it again.
    self._params.add_if_not_set("param", 1)
    # It should have done nothing.
    self.assertEqual(0, self._params.get_value("param"))

  def test_update(self):
    """ Tests that update() works under normal conditions. """
    # Add a parameter.
    self._params.add("param", 0)
    self.assertEqual(0, self._params.get_value("param"))

    # Update the value.
    self._params.update("param", 1)
    self.assertEqual(1, self._params.get_value("param"))

    # Update again.
    self._params.update("param", 2)
    self.assertEqual(2, self._params.get_value("param"))

  def test_update_bad_param(self):
    """ Tests that update() handles a bad parameter name. """
    # Try to update a parameter that doesn't exist.
    self.assertRaises(NameError, self._params.update, "param", 0)

  def test_get_value(self):
    """ Tests that get_value() works under normal conditions. """
    # Add a value.
    self._params.add("param", 0)

    # Get the value.
    self.assertEqual(0, self._params.get_value("param"))

  def test_get_value_bad_param(self):
    """ Tests that get_value() handles a bad parameter name. """
    # Try to get a parameter that doesn't exist.
    self.assertRaises(NameError, self._params.get_value, "param")

  def test_get_all(self):
    """ Tests that get_all() works under normal conditions. """
    self.assertListEqual([], self._params.get_all())

    self._params.add("param1", 0)
    names = self._params.get_all()

    self.assertEqual(1, len(names))
    self.assertIn("param1", names)

    # Add another.
    self._params.add("param2", 1)

    names = self._params.get_all()
    self.assertEqual(2, len(names))
    self.assertIn("param1", names)
    self.assertIn("param2", names)

  def test_get_changed(self):
    """ Tests that get_changed() works under normal conditions. """
    self.assertListEqual([], self._params.get_changed())

    # Add a new parameter.
    self._params.add("param1", 0)

    # It should be marked as changed.
    changed = self._params.get_changed()
    self.assertEqual(1, len(changed))
    self.assertIn("param1", changed)

    # It should be marked as changed no longer.
    self.assertListEqual([], self._params.get_changed())

    # Update the parameter.
    self._params.update("param1", 1)
    # Add a new one.
    self._params.add_if_not_set("param2", 2)

    # Both should be marked as changed.
    changed = self._params.get_changed()
    self.assertEqual(2, len(changed))
    self.assertIn("param1", changed)
    self.assertIn("param2", changed)

    # Update something without changing it.
    self._params.update("param1", 1)

    # It should not be marked as changed.
    self.assertListEqual([], self._params.get_changed())

    # Try the same thing with an addition.
    self._params.add_if_not_set("param1", 3)
    self.assertListEqual([], self._params.get_changed())

class TestStatus(TestParams):
  """ Tests for the Status class. We inherit from TestParams because Status
  should be a true subtype of Params, so everything that works on the superclass
  should also work on the subclass. """

  def setUp(self):
    # Create instance for testing.
    self._params = params.Status()

  def test_update_length_limit(self):
    """ Tests that update() correctly limits the amount of history that we
    collect. """
    self._params.add("status", 0)

    # Add a lot of historical data.
    max_len = params.Status._MAX_HISTORY_LEN
    for i in range(0, max_len + 1):
      self._params.update("status", i)

    # It should have dropped the old data points.
    history = self._params.get_history("status")
    self.assertEqual(max_len, len(history))
    # The latest datapoints should still be there.
    self.assertEqual(max_len, history[-1])

  def test_get_history(self):
    """ Tests that get_history() works under normal conditions. """
    # Create a status value and add some historical data.
    self._params.add("status", 0)
    self._params.update("status", 1)
    self._params.update("status", 2)
    self._params.update("status", 3)

    # Get the history for this parameter.
    history = self._params.get_history("status")
    self.assertListEqual([0, 1, 2, 3], history)

  def test_history_bad_param(self):
    """ Tests that get_history() handles a bad parameter name correctly. """
    self.assertRaises(NameError, self._params.get_history, "status")
