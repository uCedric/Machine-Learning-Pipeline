"""Driving (inbound) adapters: the entry points that drive the application.

These translate the outside world (a directory of image files, a Kafka topic)
into calls on the application use cases. They are wired to concrete dependencies
by the composition roots in :mod:`bootstrap`.

Submodules are imported by their composition root, never here, so an entry point
only pays for the dependencies it actually uses.
"""
