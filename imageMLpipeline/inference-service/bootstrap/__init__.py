"""Composition roots: read configuration, build adapters, wire the use cases.

This is the only layer allowed to know about both the application ports and the
concrete adapters that implement them. Each module exposes a ``main()`` runnable
with ``python -m bootstrap.<name>``.
"""
