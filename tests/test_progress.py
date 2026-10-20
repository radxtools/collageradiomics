import logging

import pytest
from collageradiomics import Collage


def test_progress_reports_ordered_stages(image_and_mask_3d, caplog):
    events = []
    with caplog.at_level(logging.INFO, logger="collageradiomics"):
        Collage(
            *image_and_mask_3d,
            svd_radius=1,
            num_unique_angles=8,
            progress_callback=lambda *event: events.append(event),
        ).execute()
    stages = list(dict.fromkeys(stage for stage, _, _ in events))
    assert stages == ["orientation", "haralick-theta", "haralick-phi"]
    for stage in stages:
        updates = [(done, total) for name, done, total in events if name == stage]
        assert updates[0][0] == 0
        assert updates[-1][0] == updates[-1][1]
        assert [done for done, _ in updates] == sorted(done for done, _ in updates)
    assert "100%" in caplog.text


def test_callback_can_cancel_without_retaining_stale_output(image_and_mask_2d):
    cancelled = False

    def cancel(stage, done, total):
        if cancelled:
            raise RuntimeError("cancelled")

    collage = Collage(*image_and_mask_2d, svd_radius=1, progress_callback=cancel)
    collage.execute()
    cancelled = True
    with pytest.raises(RuntimeError, match="cancelled"):
        collage.execute()
    with pytest.raises(RuntimeError, match="execute"):
        collage.get_single_feature_output(0)
