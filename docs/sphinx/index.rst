PyQuifer Documentation
======================

**PyQuifer** is a PyTorch library for oscillatory consciousness, implementing
neural dynamics, learning rules, and consciousness metrics that power Mizuki's "soul".

.. note::
   PyQuifer implements the idea that **consciousness emerges from synchronized
   oscillations at the edge of chaos**.

Core Principles
---------------

1. **Oscillatory Binding**: Information is bound through phase synchronization (Kuramoto model)
2. **Cross-Frequency Coupling**: Hierarchical processing via theta-gamma coupling
3. **Criticality**: Optimal computation occurs at the edge of chaos
4. **Intrinsic Motivation**: Novelty, mastery, and coherence drive learning

Quick Start
-----------

.. code-block:: python

   from pyquifer import LearnableKuramotoBank, ConsciousnessMonitor

   # Create oscillator bank
   bank = LearnableKuramotoBank(num_oscillators=32, dt=0.01)

   # Step the oscillators
   phases = bank(steps=50)
   coherence = bank.get_order_parameter()  # 0.0 (chaos) to 1.0 (sync)

   # Monitor consciousness
   monitor = ConsciousnessMonitor(state_dim=256)
   result = monitor(brain_activity)
   print(f"Consciousness level: {result['consciousness_level']}")

Installation
------------

.. code-block:: bash

   cd PyQuifer
   pip install -e .

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/index

.. toctree::
   :maxdepth: 2
   :caption: Module Groups

   modules/foundation
   modules/consciousness
   modules/learning
   modules/dynamics
   modules/embodiment
   modules/social

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
