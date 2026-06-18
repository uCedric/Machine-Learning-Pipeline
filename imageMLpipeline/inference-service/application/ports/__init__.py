"""Ports: the abstract interfaces the application layer depends on.

Driven (outbound) adapters in :mod:`adapters.outbound` implement these. Every
port stays free of any concrete SDK so the application core can be exercised
without infrastructure.
"""
