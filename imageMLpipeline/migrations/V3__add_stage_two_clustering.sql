-- Flyway V3: stage-two defect-clustering backbones.
-- Registers the 'resnet50' clustering model type and seeds two inference_model
-- rows: ResNet50 as the deployed/validated backbone (is_valid = true) and DINOv2
-- as a not-yet-valid candidate (is_valid = false) awaiting the evaluate-service.
-- The stage-two service builds only a backbone with a valid version, so ResNet50
-- runs today and DINOv2 is promoted later by flipping is_valid.
--
-- Cluster assignments land in the existing cluster_results table (see
-- V2__add_cluster_results.sql); model_id distinguishes which backbone produced
-- the row, so no schema change is needed here.

-- 'dinov2' already exists from V2; register the 'resnet50' clustering type.
INSERT INTO model_type (code)
VALUES ('resnet50')
ON CONFLICT (code) DO NOTHING;

-- ResNet50 clustering: the deployed, validated stage-two backbone.
INSERT INTO inference_model
    (model_id, model_type, bucket, major_version, minor_version, patch_version, is_valid)
VALUES
    ('b7c9a1e2-3d4f-4a5b-9c6d-7e8f9a0b1c2d', 'resnet50', 'models', 1, 0, 0, true)
ON CONFLICT (model_id) DO NOTHING;

-- DINOv2 clustering: a candidate, not yet validated by the evaluate-service.
INSERT INTO inference_model
    (model_id, model_type, bucket, major_version, minor_version, patch_version, is_valid)
VALUES
    ('c8dab2f3-4e5a-4b6c-8d7e-9f0a1b2c3d4e', 'dinov2', 'models', 1, 0, 0, false)
ON CONFLICT (model_id) DO NOTHING;
