"""Print the rows currently in the Iceberg inference-results table.

    docker compose run --rm inference-service python -m bootstrap.show_results
"""
from __future__ import annotations

from pyiceberg.catalog.sql import SqlCatalog

from config.settings import Settings


def main() -> None:
    cfg = Settings.from_env().iceberg
    catalog = SqlCatalog(
        cfg.catalog_name,
        **{
            "uri": cfg.sql_uri,
            "warehouse": cfg.warehouse,
            "s3.endpoint": cfg.s3_endpoint,
            "s3.access-key-id": cfg.s3_access_key,
            "s3.secret-access-key": cfg.s3_secret_key,
            "s3.path-style-access": "true",
            "s3.region": cfg.s3_region,
        },
    )
    table = catalog.load_table((cfg.namespace, cfg.table))
    print(table.scan().to_arrow())


if __name__ == "__main__":
    main()
