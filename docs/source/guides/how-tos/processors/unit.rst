How to turn a processor into an ezmsg Unit?
###########################################

To convert a processor to an ``ezmsg`` Unit, you can follow these steps:

1. **Define the Processor**: Create a class that inherits from the appropriate processor base class (e.g., ``BaseTransformer``, ``BaseStatefulTransformer``, etc.).
2. **Implement the Processing Logic**: Override the necessary methods to implement the processing logic.
3. **Define Input and Output Ports**: Use the ``ezmsg`` port system to define input and output ports for the processor.
4. **Register the Unit**: Use the ``ezmsg`` registration system to register the processor as an ``ezmsg`` Unit.

(under construction)
