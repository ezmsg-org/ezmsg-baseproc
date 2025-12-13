ezmsg.baseproc
==============

Base processor classes and utilities for building message-processing components in the `ezmsg <https://www.ezmsg.org>`_ framework.

Overview
--------

``ezmsg-baseproc`` provides abstract base classes for creating message processors that can be used both standalone and within ezmsg pipelines. The package offers a consistent pattern for building:

* **Processors** - Transform input messages to output messages
* **Producers** - Generate output messages without requiring input
* **Consumers** - Accept input messages without producing output
* **Transformers** - A specific type of processor with typed input/output
* **Stateful variants** - Processors that maintain state across invocations
* **Adaptive transformers** - Transformers that can be trained via ``partial_fit``
* **Composite processors** - Chain multiple processors together efficiently

All base classes support both synchronous and asynchronous operation, making them suitable for offline analysis and real-time streaming applications.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install ezmsg-baseproc

Or install the latest development version:

.. code-block:: bash

   pip install git+https://github.com/ezmsg-org/ezmsg-baseproc@main

Dependencies
^^^^^^^^^^^^

Core dependencies:

* ``ezmsg`` - Core messaging framework
* ``typing-extensions`` - Extended typing support

Quick Start
-----------

For general ezmsg tutorials and guides, visit `ezmsg.org <https://www.ezmsg.org>`_.

Here's a simple example of creating a custom transformer:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit

   class MySettings(ez.Settings):
       scale: float = 1.0

   class MyTransformer(BaseTransformer[MySettings, float, float]):
       def _process(self, message: float) -> float:
           return message * self.settings.scale

   # Use standalone
   transformer = MyTransformer(scale=2.0)
   result = transformer(5.0)  # Returns 10.0

   # Or wrap in an ezmsg Unit
   class MyUnit(BaseTransformerUnit[MySettings, float, float, MyTransformer]):
       SETTINGS = MySettings

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/ProcessorsBase
   guides/how-tos/processors/content-processors
   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
