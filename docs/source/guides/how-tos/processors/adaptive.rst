How to implement an adaptive transformer?
#########################################

Adaptive transformers are stateful processors that can be updated with labeled training data
during operation. They're used for online/incremental machine learning where the model
learns continuously from streaming data.

.. contents:: On this page
   :local:
   :depth: 2


Overview
--------

``BaseAdaptiveTransformer`` extends ``BaseStatefulTransformer`` with:

- ``partial_fit(message)`` — Update the model with labeled training data
- ``partial_fit_transform(message)`` — Train then run inference in one call
- Async variants: ``apartial_fit``, ``apartial_fit_transform``

The key distinction from regular transformers is that adaptive transformers maintain
learnable parameters that are updated via ``partial_fit``, separate from the inference
path in ``_process``.


Implementing an Adaptive Transformer
------------------------------------

Here's a minimal example:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.baseproc import BaseAdaptiveTransformer, processor_state
   from ezmsg.util.messages.axisarray import AxisArray

   class MyModelSettings(ez.Settings):
       learning_rate: float = 0.01

   @processor_state
   class MyModelState:
       weights: np.ndarray | None = None

   class MyAdaptiveModel(BaseAdaptiveTransformer[
       MyModelSettings,   # Settings type
       AxisArray,         # Input message type
       AxisArray,         # Output message type
       MyModelState,      # State type
   ]):
       def _reset_state(self, message: AxisArray) -> None:
           """Called when input structure changes. Initialize model here."""
           n_features = message.data.shape[-1]
           self._state.weights = np.zeros(n_features)

       def partial_fit(self, message: AxisArray) -> None:
           """Update model with labeled data. Label is in message.attrs['trigger'].value"""
           X = message.data
           y = message.attrs["trigger"].value
           # Update weights using your learning algorithm
           error = y - X @ self._state.weights
           self._state.weights += self.settings.learning_rate * X.T @ error

       def _process(self, message: AxisArray) -> AxisArray:
           """Run inference (prediction) on unlabeled data."""
           predictions = message.data @ self._state.weights
           return replace(message, data=predictions)


Training vs Inference
---------------------

Adaptive transformers have separate paths for training and inference:

**Training only** — use ``partial_fit()``:

.. code-block:: python

   processor.partial_fit(labeled_message)  # Updates model, returns None

**Inference only** — use ``__call__()`` or ``send()``:

.. code-block:: python

   result = processor(unlabeled_message)  # Runs _process, returns prediction

**Training + Inference** — use ``partial_fit_transform()``:

.. code-block:: python

   result = processor.partial_fit_transform(labeled_message)  # Train then predict

The ``partial_fit_transform`` method is useful when you want to train on a sample
and immediately get a prediction for it (e.g., for monitoring training progress
or chaining a training step across multiple components).


Sample Messages
---------------

Training data is passed as an ``AxisArray`` with a ``SampleTriggerMessage`` in
``attrs["trigger"]``. The trigger contains the label/target value:

.. code-block:: python

   from ezmsg.baseproc import SampleTriggerMessage
   from ezmsg.util.messages.util import replace

   # Create a labeled sample for training
   sample_message = replace(
       feature_array,  # AxisArray with input features
       attrs={"trigger": SampleTriggerMessage(
           timestamp=0.0,
           value=label,  # The target/label (can be any type)
       )}
   )

   # Train the model
   processor.partial_fit(sample_message)

The ``SampleTriggerMessage`` fields:

- ``timestamp`` — When the sample was captured
- ``value`` — The label/target value (numpy array, scalar, dict, etc.)
- ``period`` — Optional tuple of (start, end) times for the sample window


Using with ezmsg Units
----------------------

``BaseAdaptiveTransformerUnit`` wraps an adaptive transformer for use in ezmsg pipelines.
It provides four streams:

- ``INPUT_SIGNAL`` — Unlabeled data for inference
- ``INPUT_SAMPLE`` — Labeled data for training
- ``OUTPUT_SIGNAL`` — Inference results (from ``INPUT_SIGNAL``)
- ``OUTPUT_SAMPLE`` — Training + inference results (from ``INPUT_SAMPLE``)

.. code-block:: python

   from ezmsg.baseproc import BaseAdaptiveTransformerUnit

   class MyModelUnit(BaseAdaptiveTransformerUnit[
       MyModelSettings,
       AxisArray,
       AxisArray,
       MyAdaptiveModel,
   ]):
       SETTINGS = MyModelSettings

In a pipeline:

.. code-block:: python

   # Connect unlabeled data for inference
   (data_source.OUTPUT_SIGNAL, model_unit.INPUT_SIGNAL),
   (model_unit.OUTPUT_SIGNAL, prediction_consumer.INPUT_SIGNAL),

   # Connect labeled samples for training (+ inference output)
   (sample_source.OUTPUT_SAMPLE, model_unit.INPUT_SAMPLE),
   (model_unit.OUTPUT_SAMPLE, training_monitor.INPUT_SIGNAL),


Complete Example
----------------

Here's a complete example of an adaptive linear regressor:

.. code-block:: python

   import numpy as np
   import ezmsg.core as ez
   from ezmsg.baseproc import (
       BaseAdaptiveTransformer,
       BaseAdaptiveTransformerUnit,
       SampleTriggerMessage,
       processor_state,
   )
   from ezmsg.util.messages.axisarray import AxisArray
   from ezmsg.util.messages.util import replace


   class LinearRegressorSettings(ez.Settings):
       learning_rate: float = 0.001
       regularization: float = 0.01


   @processor_state
   class LinearRegressorState:
       weights: np.ndarray | None = None
       bias: float = 0.0


   class LinearRegressorTransformer(BaseAdaptiveTransformer[
       LinearRegressorSettings,
       AxisArray,
       AxisArray,
       LinearRegressorState,
   ]):
       def _hash_message(self, message: AxisArray) -> int:
           # Reset state if input shape changes
           return hash(message.data.shape[-1])

       def _reset_state(self, message: AxisArray) -> None:
           n_features = message.data.shape[-1]
           self._state.weights = np.zeros(n_features)
           self._state.bias = 0.0

       def partial_fit(self, message: AxisArray) -> None:
           X = message.data
           y = message.attrs["trigger"].value

           # Simple gradient descent update
           pred = X @ self._state.weights + self._state.bias
           error = pred - y

           grad_w = X.T @ error / len(X) + self.settings.regularization * self._state.weights
           grad_b = np.mean(error)

           self._state.weights -= self.settings.learning_rate * grad_w
           self._state.bias -= self.settings.learning_rate * grad_b

       def _process(self, message: AxisArray) -> AxisArray:
           if self._state.weights is None:
               # Not yet initialized — return empty
               return replace(message, data=np.array([]))

           predictions = message.data @ self._state.weights + self._state.bias
           return replace(message, data=predictions)


   class LinearRegressorUnit(BaseAdaptiveTransformerUnit[
       LinearRegressorSettings,
       AxisArray,
       AxisArray,
       LinearRegressorTransformer,
   ]):
       SETTINGS = LinearRegressorSettings


Tips
----

1. **Initialize in _reset_state**: Create your model/weights in ``_reset_state``, which
   is called when input structure changes (detected via ``_hash_message``).

2. **Handle uninitialized state**: Your ``_process`` may be called before any training.
   Return a sensible default (empty array, zeros, etc.).

3. **Use partial_fit_transform for closed-loop**: When you need predictions immediately
   after training (e.g., for adaptive control), use ``partial_fit_transform``.

4. **Separate streams in Units**: Use ``INPUT_SIGNAL`` for high-rate inference data and
   ``INPUT_SAMPLE`` for lower-rate training samples. They can run at different rates.
